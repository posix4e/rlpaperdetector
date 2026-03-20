from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rlpaperdetector.baseline import clean_text, load_model, load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict retraction risk with the baseline model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model.json.")
    parser.add_argument("--input-file", type=Path, help="CSV or JSONL file containing title/abstract/journal/publication_year columns.")
    parser.add_argument("--title", default="")
    parser.add_argument("--abstract", default="")
    parser.add_argument("--journal", default="")
    parser.add_argument("--publication-year", default="")
    parser.add_argument("--threshold", type=float, help="Override model threshold.")
    parser.add_argument("--output", type=Path, help="Optional output file for batch predictions.")
    return parser.parse_args()


def collect_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.input_file:
        return load_rows(args.input_file)
    return [
        {
            "title": clean_text(args.title),
            "abstract": clean_text(args.abstract),
            "journal": clean_text(args.journal),
            "publication_year": clean_text(args.publication_year),
        }
    ]


def main() -> int:
    args = parse_args()
    model, metadata = load_model(args.model)
    threshold = args.threshold if args.threshold is not None else float(metadata.get("threshold", 0.5))
    rows = collect_rows(args)
    predictions = []
    for row in rows:
        result = model.predict(row, threshold=threshold)
        predictions.append(
            {
                "pmid": clean_text(row.get("pmid")),
                "doi": clean_text(row.get("doi")),
                "score": result["score"],
                "predicted_label": result["label"],
                "title": clean_text(row.get("title")),
            }
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv":
            with args.output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()) if predictions else ["pmid", "doi", "score", "predicted_label", "title"])
                writer.writeheader()
                writer.writerows(predictions)
        else:
            with args.output.open("w", encoding="utf-8") as handle:
                for prediction in predictions:
                    handle.write(json.dumps(prediction, ensure_ascii=True) + "\n")
    else:
        print(json.dumps(predictions[0] if len(predictions) == 1 else predictions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
