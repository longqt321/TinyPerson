import argparse
import subprocess
from pathlib import Path

from tinydet.utils.runtime import default_hf_runs_repo, ensure_hf_auth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download selected run artifacts")
    parser.add_argument("run", help="experiment/run-id")
    parser.add_argument(
        "--source",
        choices=["modal", "hf"],
        default="modal",
        help="Artifact source: Modal volume or Hugging Face Hub",
    )
    parser.add_argument("--hf-repo", help="Hugging Face model repo ID (for --source hf)")
    parser.add_argument("--include", nargs="+", choices=["metadata.json", "config.yaml", "environment.txt", "metrics", "diagnostics", "checkpoints", "evaluation"], default=["metadata.json", "config.yaml", "environment.txt", "metrics"])
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    parts = Path(args.run).parts
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        parser.error("run must be experiment/run-id")
    destination = args.output.joinpath(*parts)
    destination.mkdir(parents=True, exist_ok=True)
    if args.source == "hf":
        from huggingface_hub import snapshot_download

        token, username = ensure_hf_auth()
        repo_id = args.hf_repo or default_hf_runs_repo(username)
        if not repo_id:
            parser.error("Pass --hf-repo or set HF_TOKEN / TINYDET_HF_RUNS_REPO for --source hf")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            allow_patterns=[f"{args.run}/**"],
            local_dir=args.output,
        )
        print(f"Downloaded {args.run} from hf://{repo_id} -> {destination}")
    else:
        for item in args.include:
            remote_item = "evaluation-val" if item == "evaluation" else item
            target = destination / remote_item if item in {"metrics", "diagnostics", "checkpoints", "evaluation"} else destination / item
            subprocess.run(["modal", "volume", "get", "tiny-person-runs", f"/{args.run}/{remote_item}", str(target)], check=True)
