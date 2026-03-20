from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpaperdetector import benchmark_eval


class BenchmarkEvalTests(unittest.TestCase):
    def test_collapse_prompt_rows_keeps_unique_prompt_ids(self) -> None:
        rows = [
            {"prompt_id": "a", "response": "one"},
            {"prompt_id": "a", "response": "two"},
            {"prompt_id": "b", "response": "three"},
        ]
        collapsed = benchmark_eval.collapse_prompt_rows(rows)
        self.assertEqual([row["prompt_id"] for row in collapsed], ["a", "b"])

    def test_summarize_results_scores_predictions(self) -> None:
        rubric = {"decision_tokens": ["<KEEP>", "<RETRACT>"]}
        prompts = [
            {
                "prompt_id": "p1",
                "gold_decision": "<RETRACT>",
                "paper_title": "Observational mortality analysis",
                "paper_abstract": "Observational study with confounding and subgroup claims.",
                "split": "test",
            }
        ]
        responses = [
            {
                "prompt_id": "p1",
                "response": "The observational design and confounding risk suggest caution and review.\n<RETRACT>",
            }
        ]
        rows, summary = benchmark_eval.summarize_results(prompts, "model", responses, rubric)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["graded_decision"], "<RETRACT>")
        self.assertGreaterEqual(summary["average_rubric_score"], 4.0)
        self.assertEqual(summary["decision_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
