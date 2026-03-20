from __future__ import annotations

import base64
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlpaperdetector import hf_benchmark_jobs


class HuggingFaceBenchmarkJobsTests(unittest.TestCase):
    def test_submit_benchmark_job_embeds_script_and_optional_secret(self) -> None:
        captured = {}

        def fake_run_job(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(id="job-456", url="https://huggingface.co/jobs/test/job-456")

        fake_hub = types.SimpleNamespace(run_job=fake_run_job)
        original_module = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = fake_hub
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                script_path = Path(tmpdir) / "benchmark_eval.py"
                script_path.write_text("print('benchmark')\n", encoding="utf-8")
                rubric_path = Path(tmpdir) / "rubric.json"
                rubric_path.write_text("{\"decision_tokens\": [\"<KEEP>\", \"<RETRACT>\"]}\n", encoding="utf-8")
                job = hf_benchmark_jobs.submit_benchmark_job(
                    benchmark_script_path=script_path,
                    rubric_path=rubric_path,
                    hf_pref_repo_id="user/prefs",
                    hf_model_repo_id="user/model",
                    base_model_id="Qwen/test",
                    anthropic_model="claude-sonnet-4-20250514",
                    sample_size=50,
                    seed=7,
                    max_new_tokens=128,
                    token="hf-secret",
                    flavor="a10g-small",
                    timeout="4h",
                    namespace="user",
                    anthropic_api_key="anth-secret",
                )

            self.assertEqual(job.id, "job-456")
            self.assertEqual(captured["image"], hf_benchmark_jobs.AXOLOTL_CLOUD_IMAGE)
            self.assertEqual(captured["env"]["HF_PREF_REPO_ID"], "user/prefs")
            self.assertEqual(captured["env"]["HF_MODEL_REPO_ID"], "user/model")
            self.assertEqual(captured["env"]["BASE_MODEL_ID"], "Qwen/test")
            self.assertEqual(captured["env"]["ANTHROPIC_MODEL"], "claude-sonnet-4-20250514")
            self.assertEqual(captured["env"]["SAMPLE_SIZE"], "50")
            decoded = base64.b64decode(captured["env"]["BENCHMARK_SCRIPT_B64"]).decode("utf-8")
            self.assertEqual(decoded, "print('benchmark')\n")
            rubric_decoded = base64.b64decode(captured["env"]["RUBRIC_B64"]).decode("utf-8")
            self.assertIn("decision_tokens", rubric_decoded)
            self.assertEqual(captured["secrets"]["HF_TOKEN"], "hf-secret")
            self.assertEqual(captured["secrets"]["ANTHROPIC_API_KEY"], "anth-secret")
        finally:
            if original_module is None:
                del sys.modules["huggingface_hub"]
            else:
                sys.modules["huggingface_hub"] = original_module


if __name__ == "__main__":
    unittest.main()
