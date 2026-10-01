"""Comparison plots from completed Heavy-Bottom runs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from tinydet.analysis.heavy_bottom import BINS

STAGES = ("C2", "C3", "C4", "C5")


def _plot_lines(series, ylabel, output):
    fig, axis = plt.subplots(figsize=(7, 4))
    for name, values in series.items():
        axis.plot(range(len(values)), values, marker="o", label=name)
    axis.set_xlabel("Backbone stage")
    axis.set_xticks(range(4), STAGES)
    axis.set_ylabel(ylabel)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def generate_report(runs: dict[str, Path], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, run in runs.items():
        metrics = run / "metrics"
        data = {}
        for key, filename in (("detection", "val.json"), ("size", "size_val.json")):
            file = metrics / filename
            if file.is_file():
                data[key] = json.loads(file.read_text())
        diagnostic = run / "diagnostics" / "heavy_bottom" / "metrics"
        for key, filename in (("activation", "activation_ratio_summary.csv"),
                              ("erf50", "erf50_summary.csv"), ("erf90", "erf90_summary.csv")):
            file = diagnostic / filename
            if file.is_file():
                with file.open() as source:
                    data[key] = list(csv.DictReader(source))
        summary[name] = data
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    comparison = []
    for name, run in runs.items():
        data = summary[name]
        row = {"name": name, "precision": data.get("detection", {}).get("precision"),
               "recall": data.get("detection", {}).get("recall"),
               "AP50": data.get("detection", {}).get("AP50"),
               "AP50-95": data.get("detection", {}).get("AP")}
        for bin_name in BINS:
            row[f"AP{bin_name}"] = data.get("size", {}).get(bin_name, {}).get("AP50-95")
        checkpoint = run / "checkpoints" / "best.pt"
        if checkpoint.is_file():
            model = YOLO(str(checkpoint)).model
            row["parameters"] = sum(p.numel() for p in model.parameters())
            row["gflops_640"] = get_flops(model, imgsz=640)
        train_file = run / "metrics" / "train.csv"
        if train_file.is_file():
            with train_file.open() as source:
                epochs = [{key.strip(): value for key, value in row.items()}
                          for row in csv.DictReader(source)]
            if epochs:
                column = next((key for key in epochs[0] if "mAP50-95" in key), None)
                row["final_epoch"] = int(float(epochs[-1]["epoch"]))
                row["best_epoch"] = (int(float(max(epochs, key=lambda item: float(item[column]))["epoch"]))
                                     if column else None)
                row["final_epoch_val_ap"] = float(epochs[-1][column]) if column else None
        config_file = run / "config.yaml"
        if config_file.is_file() and row.get("final_epoch") is not None:
            budget = yaml.safe_load(config_file.read_text())["train"]["epochs"]
            row["early_stop_epoch"] = row["final_epoch"] if row["final_epoch"] < budget else None
        metadata_file = run / "metadata.json"
        if metadata_file.is_file():
            metadata = json.loads(metadata_file.read_text())
            row["training_seconds"] = metadata.get("training_seconds")
            row["peak_vram_gb"] = metadata.get("peak_vram_gb")
        comparison.append(row)
    if comparison:
        keys = list(dict.fromkeys(key for row in comparison for key in row))
        with (output / "comparison.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(comparison)
    if all("size" in value for value in summary.values()):
        fig, axis = plt.subplots(figsize=(8, 4))
        for name, data in summary.items():
            axis.plot(BINS, [data["size"][bin_name]["AP50-95"] for bin_name in BINS], marker="o", label=name)
        axis.set_ylabel("AP50:95")
        axis.set_xlabel("Max object side in input pixels")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "ap_by_size.png", dpi=160)
        plt.close(fig)
    for key, label in (("activation", "Object/background activation ratio"),
                       ("erf50", "ERF50 area fraction"), ("erf90", "ERF90 area fraction")):
        if all(key in data for data in summary.values()):
            series = {}
            for name, data in summary.items():
                rows = [row for row in data[key] if row["size_bin"] in BINS[:2]]
                series[name] = [sum(float(row["mean"]) for row in rows if row["stage"] == stage) /
                                max(sum(row["stage"] == stage for row in rows), 1) for stage in STAGES]
            _plot_lines(series, label, output / f"{key}.png")
    profiles = {row["name"]: row for row in comparison}
    if all("detection" in data for data in summary.values()):
        fig, axis = plt.subplots(figsize=(7, 4))
        for name, run in runs.items():
            axis.scatter(profiles[name]["parameters"] / 1e6,
                         summary[name]["detection"]["AP"], label=name)
        axis.set_xlabel("Parameters (millions)")
        axis.set_ylabel("AP50:95")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "parameters_vs_ap.png", dpi=160)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(7, 4))
        for name, run in runs.items():
            axis.scatter(profiles[name]["gflops_640"], summary[name]["detection"]["AP"], label=name)
        axis.set_xlabel("GFLOPs at 640 px")
        axis.set_ylabel("AP50:95")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "gflops_vs_ap.png", dpi=160)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    curves = 0
    for name, run in runs.items():
        file = run / "metrics" / "train.csv"
        if not file.is_file():
            continue
        with file.open() as source:
            rows = [{key.strip(): value for key, value in row.items()}
                    for row in csv.DictReader(source)]
        column = next((key for key in rows[0] if "mAP50-95" in key), None) if rows else None
        if column:
            axis.plot([int(float(row["epoch"])) for row in rows],
                      [float(row[column]) for row in rows], label=name)
            curves += 1
    if curves:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation AP50:95")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)
    return output
