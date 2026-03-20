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

from rlpaperdetector import hf_jobs


class HuggingFaceJobsTests(unittest.TestCase):
    def test_build_training_job_script_includes_expected_steps(self) -> None:
        script = hf_jobs.build_training_job_script()
        self.assertIn("snapshot_download", script)
        self.assertIn("axolotl train train_config.yaml", script)
        self.assertIn("api.upload_folder", script)

    def test_submit_training_job_passes_embedded_config_and_secrets(self) -> None:
        captured = {}

        def fake_run_job(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(id="job-123", url="https://huggingface.co/jobs/test/job-123")

        fake_hub = types.SimpleNamespace(run_job=fake_run_job)
        original_module = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = fake_hub
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "train.yaml"
                config_path.write_text("base_model: test/model\n", encoding="utf-8")
                job = hf_jobs.submit_training_job(
                    config_path=config_path,
                    hf_pref_repo_id="user/prefs",
                    hf_model_repo_id="user/model",
                    token="secret-token",
                    private_repo=True,
                    flavor="a10g-small",
                    timeout="8h",
                    namespace="user",
                )

            self.assertEqual(job.id, "job-123")
            self.assertEqual(captured["image"], hf_jobs.AXOLOTL_CLOUD_IMAGE)
            self.assertEqual(captured["command"][0:2], ["bash", "-lc"])
            self.assertEqual(captured["env"]["HF_PREF_REPO_ID"], "user/prefs")
            self.assertEqual(captured["env"]["HF_MODEL_REPO_ID"], "user/model")
            self.assertEqual(captured["env"]["HF_PRIVATE_REPO"], "true")
            decoded = base64.b64decode(captured["env"]["AXOLOTL_CONFIG_B64"]).decode("utf-8")
            self.assertEqual(decoded, "base_model: test/model\n")
            self.assertEqual(captured["secrets"], {"HF_TOKEN": "secret-token"})
            self.assertEqual(captured["namespace"], "user")
            self.assertEqual(captured["token"], "secret-token")
        finally:
            if original_module is None:
                del sys.modules["huggingface_hub"]
            else:
                sys.modules["huggingface_hub"] = original_module


if __name__ == "__main__":
    unittest.main()
