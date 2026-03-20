from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlpaperdetector.baseline import load_model, load_rows, split_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model.json.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.csv or dataset.jsonl.")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--threshold", type=float, help="Override model threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, metadata = load_model(args.model)
    threshold = args.threshold if args.threshold is not None else float(metadata.get("threshold", 0.5))
    rows = load_rows(args.dataset)
    if args.split == "all":
        selected_rows = rows
    else:
        selected_rows = split_rows(rows).get(args.split, [])
    metrics = model.evaluate(selected_rows, threshold=threshold)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
