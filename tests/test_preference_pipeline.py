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

from rlpaperdetector import preference_pipeline


FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "candidate_responses.jsonl"
RUBRIC = PROJECT_ROOT / "configs" / "retraction_rubric.json"


class PreferencePipelineTests(unittest.TestCase):
    def test_pipeline_builds_scored_and_preference_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "preferences"
            argv = [
                "build_preference_data.py",
                "--input",
                str(FIXTURE),
                "--output-dir",
                str(output_dir),
                "--rubric",
                str(RUBRIC),
                "--min-score-gap",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(preference_pipeline.main(), 0)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["scored_rows"], 4)
            self.assertGreaterEqual(summary["dpo_pairs"], 2)
            self.assertGreaterEqual(summary["orpo_pairs_expanded"], summary["orpo_pairs"])

            scored_rows = [json.loads(line) for line in (output_dir / "scored" / "all.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(scored_rows), 4)
            top_scores = {row["prompt_id"]: row["rubric_score"] for row in scored_rows if "limited adjustment" in row["response"] or "prespecified endpoints" in row["response"]}
            self.assertEqual(top_scores["paper-1"], 5)
            self.assertEqual(top_scores["paper-2"], 5)

            dpo_rows = [json.loads(line) for line in (output_dir / "axolotl" / "dpo_all.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(all("score_gap" in row for row in dpo_rows))
            self.assertTrue(all(row["chosen"]["role"] == "assistant" for row in dpo_rows))

            orpo_rows = [json.loads(line) for line in (output_dir / "axolotl" / "orpo_all.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(all(isinstance(row["chosen"], list) for row in orpo_rows))

            sft_rows = [json.loads(line) for line in (output_dir / "axolotl" / "sft_all.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(sft_rows), 2)


if __name__ == "__main__":
    unittest.main()
