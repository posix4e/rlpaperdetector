from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlpaperdetector.baseline import NaiveBayesRetractionModel, load_rows, save_model, split_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure-Python retraction baseline model.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset.csv or dataset.jsonl.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing strength.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.dataset)
    split_map = split_rows(rows)
    train_rows = split_map.get("train", [])
    validation_rows = split_map.get("validation", [])
    test_rows = split_map.get("test", [])
    if not train_rows:
        raise SystemExit("Training split is empty. Build a dataset with split assignments first.")

    model = NaiveBayesRetractionModel.train(train_rows, alpha=args.alpha)
    metrics = {
        "train": model.evaluate(train_rows, threshold=args.threshold),
        "validation": model.evaluate(validation_rows, threshold=args.threshold),
        "test": model.evaluate(test_rows, threshold=args.threshold),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": str(args.dataset),
        "threshold": args.threshold,
        "alpha": args.alpha,
        "metrics": metrics,
    }
    save_model(args.output_dir / "model.json", model, metadata)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
