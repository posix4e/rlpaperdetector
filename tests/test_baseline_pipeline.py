from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpaperdetector import baseline
from rlpaperdetector import eval_baseline
from rlpaperdetector import predict_baseline
from rlpaperdetector import train_baseline


class BaselinePipelineTests(unittest.TestCase):
    def make_dataset(self, path: Path) -> None:
        rows = [
            {"label": "1", "split": "train", "title": "Image duplication in western blot paper", "abstract": "duplicated panels and fabricated figure", "journal": "Cell Reports", "publication_year": "2020", "pmid": "1", "doi": "", "matched_positive_pmid": "", "group_id": "1"},
            {"label": "0", "split": "train", "title": "Prospective cohort of hypertension outcomes", "abstract": "clinical follow up and survival analysis", "journal": "Hypertension", "publication_year": "2020", "pmid": "2", "doi": "", "matched_positive_pmid": "1", "group_id": "1"},
            {"label": "1", "split": "validation", "title": "Retraction for manipulated microscopy images", "abstract": "image manipulation and missing raw data", "journal": "Nature Medicine", "publication_year": "2021", "pmid": "3", "doi": "", "matched_positive_pmid": "", "group_id": "3"},
            {"label": "0", "split": "validation", "title": "Genome wide association study of asthma", "abstract": "replication cohort and statistical genetics", "journal": "Nature Genetics", "publication_year": "2021", "pmid": "4", "doi": "", "matched_positive_pmid": "3", "group_id": "3"},
            {"label": "1", "split": "test", "title": "Paper with falsified tumor images", "abstract": "falsified images and unavailable originals", "journal": "Cancer Cell", "publication_year": "2022", "pmid": "5", "doi": "", "matched_positive_pmid": "", "group_id": "5"},
            {"label": "0", "split": "test", "title": "Randomized trial of migraine treatment", "abstract": "double blind placebo controlled efficacy", "journal": "Neurology", "publication_year": "2022", "pmid": "6", "doi": "", "matched_positive_pmid": "5", "group_id": "5"},
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def test_train_eval_predict_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "dataset.csv"
            model_dir = tmp / "model"
            prediction_path = tmp / "predictions.jsonl"
            self.make_dataset(dataset_path)

            with unittest.mock.patch.object(sys, "argv", ["train_baseline.py", "--dataset", str(dataset_path), "--output-dir", str(model_dir)]):
                self.assertEqual(train_baseline.main(), 0)

            model_path = model_dir / "model.json"
            metrics_path = model_dir / "metrics.json"
            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())

            model, metadata = baseline.load_model(model_path)
            self.assertIn("metrics", metadata)
            test_metrics = model.evaluate(baseline.split_rows(baseline.load_rows(dataset_path))["test"])
            self.assertGreaterEqual(test_metrics["accuracy"], 0.5)

            with unittest.mock.patch.object(sys, "argv", ["predict.py", "--model", str(model_path), "--input-file", str(dataset_path), "--output", str(prediction_path)]):
                self.assertEqual(predict_baseline.main(), 0)

            predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(predictions), 6)
            self.assertIn("score", predictions[0])

            with unittest.mock.patch.object(sys, "argv", ["eval.py", "--model", str(model_path), "--dataset", str(dataset_path), "--split", "test"]):
                self.assertEqual(eval_baseline.main(), 0)


if __name__ == "__main__":
    unittest.main()
