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

from rlpaperdetector import dataset_builder as db


FIXTURE_CSV = PROJECT_ROOT / "tests" / "fixtures" / "retraction_watch_fixture.csv"


class FakePubMedClient:
    def __init__(self, email: str, delay_seconds: float = 0.34) -> None:
        self.email = email
        self.delay_seconds = delay_seconds

    def esearch(self, term: str, retmax: int) -> list[str]:
        if '"Journal A"[TA]' in term:
            return ["333333"]
        if '"Journal B"[TA]' in term:
            return ["444444"]
        return []

    def efetch(self, pmids: list[str]) -> dict[str, db.PubMedRecord]:
        catalog = {
            "111111": db.PubMedRecord(
                pmid="111111",
                doi="10.1000/original1",
                title="Positive one",
                abstract="Abstract one",
                journal="Journal A",
                publication_year=2020,
                publication_types=["Journal Article", "Retracted Publication"],
            ),
            "222222": db.PubMedRecord(
                pmid="222222",
                doi="10.1000/original2",
                title="Positive two",
                abstract="Abstract two",
                journal="Journal B",
                publication_year=2021,
                publication_types=["Journal Article", "Retracted Publication"],
            ),
            "333333": db.PubMedRecord(
                pmid="333333",
                doi="10.1000/negative1",
                title="Negative one",
                abstract="Control abstract one",
                journal="Journal A",
                publication_year=2020,
                publication_types=["Journal Article"],
            ),
            "444444": db.PubMedRecord(
                pmid="444444",
                doi="10.1000/negative2",
                title="Negative two",
                abstract="Control abstract two",
                journal="Journal B",
                publication_year=2021,
                publication_types=["Journal Article"],
            ),
        }
        return {pmid: catalog[pmid] for pmid in pmids if pmid in catalog}


class DatasetBuilderTests(unittest.TestCase):
    def test_normalize_pmid_filters_zero(self) -> None:
        self.assertIsNone(db.normalize_pmid("0"))
        self.assertEqual(db.normalize_pmid("PMID: 12345"), "12345")

    def test_main_builds_expected_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "processed"
            argv = [
                "build_dataset.py",
                "--rw-csv",
                str(FIXTURE_CSV),
                "--output-dir",
                str(output_dir),
                "--negatives-per-positive",
                "1",
                "--crossref-email",
                "ci@example.org",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(db, "PubMedClient", FakePubMedClient):
                    exit_code = db.main()

            self.assertEqual(exit_code, 0)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["positives_written"], 2)
            self.assertEqual(summary["negatives_written"], 2)
            self.assertEqual(summary["rows_written"], 4)
            self.assertEqual(summary["split_counts"]["train"], 2)
            self.assertEqual(summary["split_counts"]["validation"], 0)
            self.assertEqual(summary["split_counts"]["test"], 2)

            with (output_dir / "dataset.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 4)
            positive_rows = [row for row in rows if row["label"] == "1"]
            negative_rows = [row for row in rows if row["label"] == "0"]
            self.assertEqual({row["pmid"] for row in positive_rows}, {"111111", "222222"})
            self.assertEqual({row["pmid"] for row in negative_rows}, {"333333", "444444"})
            self.assertEqual({row["matched_positive_pmid"] for row in negative_rows}, {"111111", "222222"})
            self.assertEqual({row["split"] for row in rows}, {"train", "test"})

            hf_dir = output_dir / "hf"
            self.assertTrue((hf_dir / "train.jsonl").exists())
            self.assertTrue((hf_dir / "validation.jsonl").exists())
            self.assertTrue((hf_dir / "test.jsonl").exists())
            self.assertTrue((hf_dir / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
