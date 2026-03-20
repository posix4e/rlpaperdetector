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

from rlpaperdetector import probe_set
from rlpaperdetector.dataset_builder import PubMedRecord


class FakeProbeClient:
    def __init__(self, email: str, delay_seconds: float = 0.34) -> None:
        self.email = email

    def efetch(self, pmids: list[str]) -> dict[str, PubMedRecord]:
        catalog = {
            "32817561": PubMedRecord(
                pmid="32817561",
                doi="10.1073/pnas.1913405117",
                title="Physician-patient racial concordance and disparities in birthing mortality for newborns.",
                abstract="Observational analysis of newborn mortality outcomes by physician race.",
                journal="Proc Natl Acad Sci U S A",
                publication_year=2020,
                publication_types=["Journal Article"],
            )
        }
        return {pmid: catalog[pmid] for pmid in pmids if pmid in catalog}


class ProbeSetTests(unittest.TestCase):
    def test_builds_probe_prompt_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "probe.jsonl"
            argv = [
                "build_probe_set.py",
                "--pmid",
                "32817561",
                "--output",
                str(output),
                "--crossref-email",
                "ci@example.org",
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch.object(probe_set, "PubMedClient", FakeProbeClient):
                    self.assertEqual(probe_set.main(), 0)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["pmid"], "32817561")
            self.assertIn("strict retraction judge", rows[0]["messages"][0]["content"].lower())


if __name__ == "__main__":
    unittest.main()
