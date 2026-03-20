from __future__ import annotations

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

from rlpaperdetector import monitor_subjects
from rlpaperdetector.dataset_builder import PubMedRecord


class FakeMonitorClient:
    def __init__(self, email: str, delay_seconds: float = 0.34) -> None:
        self.email = email
        self.delay_seconds = delay_seconds

    def esearch(self, term: str, retmax: int, retstart: int = 0) -> list[str]:
        if "oncology" in term:
            return ["100", "101"][retstart : retstart + retmax]
        if "neurology" in term:
            return ["101", "102"][retstart : retstart + retmax]
        return []

    def efetch(self, pmids: list[str]) -> dict[str, PubMedRecord]:
        catalog = {
            "100": PubMedRecord("100", "10.1/100", "High-risk oncology paper", "image duplication and raw data missing", "Journal A", 2026, ["Journal Article"]),
            "101": PubMedRecord("101", "10.1/101", "Benign oncology paper", "prospective cohort with external validation", "Journal B", 2026, ["Journal Article"]),
            "102": PubMedRecord("102", "10.1/102", "Neurology paper", "manipulated microscopy images", "Journal C", 2026, ["Journal Article"]),
        }
        return {pmid: catalog[pmid] for pmid in pmids if pmid in catalog}


class FakeModel:
    def predict(self, row: dict[str, str], threshold: float = 0.5) -> dict[str, float | int]:
        text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
        score = 0.95 if "image" in text or "manipulated" in text else 0.15
        return {"score": score, "label": 1 if score >= threshold else 0}


class MonitorSubjectsTests(unittest.TestCase):
    def test_monitor_scores_new_subject_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            query_file = tmp / "queries.txt"
            query_file.write_text("oncology\nneurology\n", encoding="utf-8")
            state_file = tmp / "state.json"
            state_file.write_text(json.dumps({"seen_pmids": ["101"]}), encoding="utf-8")
            output_dir = tmp / "out"

            argv = [
                "monitor_subjects.py",
                "--model",
                str(tmp / "model.json"),
                "--query-file",
                str(query_file),
                "--output-dir",
                str(output_dir),
                "--state-file",
                str(state_file),
                "--crossref-email",
                "ci@example.org",
                "--days-back",
                "2",
                "--end-date",
                "2026-03-20",
                "--min-score",
                "0.2",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(monitor_subjects, "PubMedClient", FakeMonitorClient):
                    with mock.patch.object(monitor_subjects, "load_model", return_value=(FakeModel(), {"threshold": 0.5})):
                        self.assertEqual(monitor_subjects.main(), 0)

            rows = [json.loads(line) for line in (output_dir / "scored_papers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["pmid"] for row in rows], ["100", "102"])
            self.assertGreater(rows[0]["score"], 0.9)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["new_pmids"], 2)
            self.assertEqual(summary["scored_rows"], 2)

            state = json.loads(state_file.read_text(encoding="utf-8"))
            self.assertEqual(state["seen_pmids"], ["100", "101", "102"])


if __name__ == "__main__":
    unittest.main()
