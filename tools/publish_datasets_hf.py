"""Package TinyPerson & VisDrone2019-DET as verified .tar.zst bundles and push to Hugging Face Hub.

Preserves:
1. ``tinyperson-yolo-v2``: official ``erase_with_uncertain`` train/val rasters + raw ``test`` rasters,
   plus normalized ``annotations/tiny_set_{train,val,test}.json`` matching ``images/{split}/<basename>``.
2. ``visdrone-yolo``: official 10-class train/val/test-dev splits, plus rich COCO JSONs
   (``visdrone_val_coco.json``, ``visdrone_test_dev_coco.json``) extracted via Polars from the
   verbatim VisDrone Parquet files with ``iscrowd=1`` on ``ignored regions`` and ``occlusion`` intact.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import orjson
import polars as pl
import yaml
import zstandard
from huggingface_hub import HfApi

from tinydet.evaluation.tinyperson_benchmark import subset_annotations_to_split
from tinydet.utils.runtime import DEFAULT_HF_DATASET_REPO, ensure_hf_auth

REPO_ROOT = Path(__file__).resolve().parents[1]

VISDRONE_10_CLASSES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]


def _sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _compress_dir_tar_zst(src_dir: Path, archive_out: Path, threads: int = -1, level: int = 3, dereference: bool = False) -> str:
    archive_out.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstandard.ZstdCompressor(level=level, threads=threads)
    with (
        archive_out.open("wb") as raw_out,
        cctx.stream_writer(raw_out) as compressor,
        tarfile.open(fileobj=compressor, mode="w|", dereference=dereference) as tar,
    ):
        tar.add(src_dir, arcname=src_dir.name)
    digest = _sha256_file(archive_out)
    archive_out.with_name(f"{archive_out.name}.sha256").write_text(f"{digest}  {archive_out.name}\n")
    return digest


def _normalize_tinyperson_json(src_json: Path, split_sub: str, dst_json: Path) -> int:
    payload = orjson.loads(src_json.read_bytes())
    images = []
    for img in payload["images"]:
        basename = Path(str(img["file_name"])).name
        images.append({**img, "file_name": f"images/{split_sub}/{basename}"})
    out = {**payload, "images": images}
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_bytes(orjson.dumps(out))
    return len(images)


def _build_visdrone_coco_from_parquet(parquet_path: Path, split_sub: str, dst_json: Path) -> dict[str, int]:
    df = pl.read_parquet(parquet_path, columns=["id", "width", "height", "objects"])
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    ignored_count = 0
    positive_count = 0

    for img_idx, row in enumerate(df.iter_rows(named=True), start=1):
        stem = str(row["id"])
        width = int(row["width"])
        height = int(row["height"])
        images.append({
            "id": img_idx,
            "file_name": f"images/{split_sub}/{stem}.jpg",
            "width": width,
            "height": height,
        })
        objs = row["objects"]
        cats = objs["category"]
        bboxes = objs["bbox"]
        scores = objs["score"]
        truncs = objs["truncation"]
        occs = objs["occlusion"]
        for cat, bbox, score_flag, trunc, occ in zip(cats, bboxes, scores, truncs, occs):
            c_int = int(cat)
            w, h = float(bbox[2]), float(bbox[3])
            if w <= 0.0 or h <= 0.0:
                continue
            if c_int == 0:
                # Official VisDrone ignored region -> represent as crowd ignore across class 1..10
                ignored_count += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": 1,
                    "bbox": [float(bbox[0]), float(bbox[1]), w, h],
                    "area": w * h,
                    "iscrowd": 1,
                    "ignore": True,
                    "truncation": int(trunc),
                    "occlusion": int(occ),
                })
                ann_id += 1
            elif 1 <= c_int <= 10 and int(score_flag) != 0:
                positive_count += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_idx,
                    "category_id": c_int,
                    "bbox": [float(bbox[0]), float(bbox[1]), w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "ignore": False,
                    "truncation": int(trunc),
                    "occlusion": int(occ),
                })
                ann_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": VISDRONE_10_CLASSES,
    }
    dst_json.parent.mkdir(parents=True, exist_ok=True)
    dst_json.write_bytes(orjson.dumps(payload))
    return {"images": len(images), "positive_boxes": positive_count, "ignored_boxes": ignored_count}


def prepare_and_upload(repo_id: str, out_dir: Path, skip_upload: bool = False) -> dict[str, Any]:
    token, user = ensure_hf_auth()
    if not skip_upload and not token:
        raise RuntimeError("HF_TOKEN is required to upload datasets to Hugging Face Hub.")
    print(f"[hf-publish] Authenticated user: {user} | target dataset repo: {repo_id}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"repo_id": repo_id, "archives": {}}

    # 1. Prepare TinyPerson bundle (in a lightweight staging overlay without mutating workspace source)
    tp_src = REPO_ROOT / "datasets" / "derived" / "tinyperson-yolo-v2"
    if not tp_src.is_dir():
        raise FileNotFoundError(f"Missing {tp_src}")

    with tempfile.TemporaryDirectory(prefix="pkg-tp-") as tmp:
        stage_tp = Path(tmp) / "tinyperson-yolo-v2"
        stage_tp.mkdir(parents=True)
        for sub in ("images", "labels"):
            os.symlink((tp_src / sub).resolve(), stage_tp / sub, target_is_directory=True)
        if (tp_src / "manifest.json").is_file():
            shutil.copy2(tp_src / "manifest.json", stage_tp / "manifest.json")

        tp_yaml = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: "person"},
        }
        (stage_tp / "data.yaml").write_text(yaml.safe_dump(tp_yaml, sort_keys=False))

        ann_out = stage_tp / "annotations"
        ann_out.mkdir(exist_ok=True)
        raw_train_json = tp_src / "annotations" / "tiny_set_train.json"
        raw_test_json = tp_src / "annotations" / "tiny_set_test.json"
        subset_annotations_to_split(raw_train_json, stage_tp / "data.yaml", "train", ann_out / "_train_sub.json")
        subset_annotations_to_split(raw_train_json, stage_tp / "data.yaml", "val", ann_out / "_val_sub.json")
        n_tp_train = _normalize_tinyperson_json(ann_out / "_train_sub.json", "train", ann_out / "tiny_set_train.json")
        n_tp_val = _normalize_tinyperson_json(ann_out / "_val_sub.json", "val", ann_out / "tiny_set_val.json")
        n_tp_test = _normalize_tinyperson_json(raw_test_json, "test", ann_out / "tiny_set_test.json")
        (ann_out / "_train_sub.json").unlink()
        (ann_out / "_val_sub.json").unlink()

        tp_archive = out_dir / "tinyperson-yolo-v2.tar.zst"
        print(f"[hf-publish] Compressing {tp_archive.name} (dereferencing image/label symlinks)...", flush=True)
        tp_sha = _compress_dir_tar_zst(stage_tp, tp_archive, threads=-1, level=3, dereference=True)
        summary["archives"][tp_archive.name] = {
            "sha256": tp_sha,
            "size_mb": round(tp_archive.stat().st_size / 1024**2, 2),
            "splits": {"train": n_tp_train, "val": n_tp_val, "test": n_tp_test},
        }
        print(f"[hf-publish] Built {tp_archive.name}: {summary['archives'][tp_archive.name]}", flush=True)

    # 2. Prepare VisDrone bundle
    vd_src = REPO_ROOT / "datasets" / "derived" / "visdrone-yolo"
    vd_pq_dir = REPO_ROOT / "datasets" / "tiny-object-detection" / "visdrone2019-det" / "detection"
    if not vd_src.is_dir() or not vd_pq_dir.is_dir():
        raise FileNotFoundError(f"Missing VisDrone sources: {vd_src} or {vd_pq_dir}")

    with tempfile.TemporaryDirectory(prefix="pkg-vd-") as tmp:
        stage_vd = Path(tmp) / "visdrone-yolo"
        stage_vd.mkdir(parents=True)
        for sub in ("images", "labels"):
            os.symlink((vd_src / sub).resolve(), stage_vd / sub, target_is_directory=True)

        vd_yaml = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: c["name"] for i, c in enumerate(VISDRONE_10_CLASSES)},
        }
        (stage_vd / "data.yaml").write_text(yaml.safe_dump(vd_yaml, sort_keys=False))

        vd_ann_out = stage_vd / "annotations"
        val_stats = _build_visdrone_coco_from_parquet(
            vd_pq_dir / "validation-00000-of-00001.parquet",
            "val",
            vd_ann_out / "visdrone_val_coco.json",
        )
        test_stats = _build_visdrone_coco_from_parquet(
            vd_pq_dir / "test-00000-of-00001.parquet",
            "test",
            vd_ann_out / "visdrone_test_dev_coco.json",
        )

        vd_archive = out_dir / "visdrone-yolo.tar.zst"
        print(f"[hf-publish] Compressing {vd_archive.name}...", flush=True)
        vd_sha = _compress_dir_tar_zst(stage_vd, vd_archive, threads=-1, level=3, dereference=True)
        summary["archives"][vd_archive.name] = {
            "sha256": vd_sha,
            "size_mb": round(vd_archive.stat().st_size / 1024**2, 2),
            "val_coco": val_stats,
            "test_dev_coco": test_stats,
        }
        print(f"[hf-publish] Built {vd_archive.name}: {summary['archives'][vd_archive.name]}", flush=True)

    manifest_file = out_dir / "manifest.json"
    manifest_file.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

    if not skip_upload:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        print(f"[hf-publish] Uploading bundle folder {out_dir} -> hf://datasets/{repo_id} (private=True)...", flush=True)
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Publish verified TinyPerson & VisDrone2019-DET .tar.zst benchmark bundles",
        )
        print(f"[hf-publish] Upload complete: https://huggingface.co/datasets/{repo_id}", flush=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--staging-dir", type=Path, default=Path("/tmp/tinydet-hf-bundle"))
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    prepare_and_upload(args.repo_id, args.staging_dir, skip_upload=args.skip_upload)


if __name__ == "__main__":
    main()
