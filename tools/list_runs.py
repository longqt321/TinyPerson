import argparse
from pathlib import Path

from tinydet.research.results import (
    collect_runs,
    load_comparisons,
    render_markdown,
    write_csv,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List downloaded experiment runs")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--config", type=Path, help="Experiment config with comparisons")
    parser.add_argument("--markdown", type=Path, help="Write a Markdown results table")
    args = parser.parse_args()
    rows = collect_runs(args.root)
    if args.csv:
        write_csv(args.csv, rows)
    if args.markdown:
        if args.config is None:
            parser.error("--markdown requires --config")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            render_markdown(rows, load_comparisons(args.config), args.root)
        )
    for row in rows:
        print(" ".join(f"{key}={value}" for key, value in row.items()))
