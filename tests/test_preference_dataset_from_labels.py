from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpaperdetector import preference_dataset_from_labels


class PreferenceDatasetFromLabelsTests(unittest.TestCase):
    def test_builds_candidates_and_axolotl_exports_from_labeled_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset = tmp / "dataset.csv"
            output_dir = tmp / "preferences"
            rows = [
                {"label": "1", "pmid": "1", "doi": "10.1/1", "title": "Observational neonatal mortality analysis", "abstract": "This observational study reports subgroup mortality differences with limited adjustment.", "journal": "PNAS", "publication_year": "2020", "source": "retraction_watch_pubmed", "split": "train"},
                {"label": "0", "pmid": "2", "doi": "10.1/2", "title": "Randomized migraine therapy trial", "abstract": "This randomized trial reports prespecified endpoints and sensitivity analyses.", "journal": "Neurology", "publication_year": "2021", "source": "pubmed_matched_negative", "split": "validation"},
            ]
            with dataset.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            argv = [
                "build_rw_preferences.py",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output_dir),
                "--max-papers",
                "2",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(preference_dataset_from_labels.main(), 0)

            raw_rows = [json.loads(line) for line in (output_dir / "raw" / "candidate_responses.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(raw_rows), 8)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["scored_rows"], 8)
            self.assertGreaterEqual(summary["dpo_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
