from __future__ import annotations

import argparse
import base64
import os
import time
from pathlib import Path


AXOLOTL_CLOUD_IMAGE = "axolotlai/axolotl-cloud:main-latest"
TERMINAL_JOB_STAGES = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


def encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_training_job_script() -> str:
    return """set -euo pipefail
cd /workspace
mkdir -p preferences outputs
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_PREF_REPO_ID"],
    repo_type="dataset",
    local_dir="preferences",
    token=os.environ["HF_TOKEN"],
)
PY
python - <<'PY'
import base64
import os
from pathlib import Path
import yaml

config_text = base64.b64decode(os.environ["AXOLOTL_CONFIG_B64"]).decode("utf-8")
cfg = yaml.safe_load(config_text)
for dataset in cfg.get("datasets", []):
    configured_path = dataset.get("path")
    if configured_path:
        dataset["path"] = str(Path("preferences") / "axolotl" / Path(configured_path).name)
    elif dataset.get("data_files"):
        local_files = []
        for path in dataset.get("data_files", []):
            filename = Path(path).name
            local_files.append(str(Path("preferences") / "axolotl" / filename))
        dataset["data_files"] = local_files
        dataset["path"] = local_files[0]
Path("train_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY
axolotl train train_config.yaml
python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id=os.environ["HF_MODEL_REPO_ID"],
    repo_type="model",
    private=os.environ["HF_PRIVATE_REPO"].lower() == "true",
    exist_ok=True,
)
api.upload_folder(
    repo_id=os.environ["HF_MODEL_REPO_ID"],
    repo_type="model",
    folder_path="outputs",
    commit_message="Upload Axolotl training outputs from Hugging Face Job",
)
PY
"""


def submit_training_job(
    *,
    config_path: Path,
    hf_pref_repo_id: str,
    hf_model_repo_id: str,
    token: str,
    private_repo: bool,
    flavor: str,
    timeout: str,
    namespace: str | None = None,
):
    try:
        from huggingface_hub import run_job
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Install it with `pip install huggingface_hub`.") from exc

    return run_job(
        image=AXOLOTL_CLOUD_IMAGE,
        command=["bash", "-lc", build_training_job_script()],
        env={
            "HF_PREF_REPO_ID": hf_pref_repo_id,
            "HF_MODEL_REPO_ID": hf_model_repo_id,
            "HF_PRIVATE_REPO": "true" if private_repo else "false",
            "AXOLOTL_CONFIG_B64": encode_file_base64(config_path),
        },
        secrets={"HF_TOKEN": token},
        flavor=flavor,
        timeout=timeout,
        labels={
            "project": "rlpaperdetector",
            "task": "axolotl-train",
            "config": config_path.stem,
        },
        namespace=namespace,
        token=token,
    )


def wait_for_job(job_id: str, token: str, namespace: str | None = None, poll_seconds: int = 60):
    try:
        from huggingface_hub import inspect_job
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Install it with `pip install huggingface_hub`.") from exc

    while True:
        job = inspect_job(job_id=job_id, namespace=namespace, token=token)
        if job.status is not None and job.status.stage in TERMINAL_JOB_STAGES:
            return job
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit an Axolotl training job to Hugging Face Jobs.")
    parser.add_argument("--hf-pref-repo-id", required=True, help="Dataset repo id that contains the built preference package.")
    parser.add_argument("--hf-model-repo-id", required=True, help="Model repo id to upload trained outputs to.")
    parser.add_argument("--train-config", type=Path, required=True, help="Local Axolotl config file to embed into the remote job.")
    parser.add_argument("--flavor", default="a10g-small", help="Hugging Face Jobs hardware flavor.")
    parser.add_argument("--timeout", default="8h", help="Job timeout, for example `8h` or `14400`.")
    parser.add_argument("--namespace", default="", help="Optional Hugging Face namespace to submit the job under.")
    parser.add_argument("--private-repo", action="store_true", help="Create the output model repo as private if needed.")
    parser.add_argument("--wait", action="store_true", help="Wait for the remote job to finish and return non-zero on failure.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="Hugging Face token. Defaults to HF_TOKEN.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.token:
        raise SystemExit("Missing Hugging Face token. Pass --token or set HF_TOKEN.")
    if not args.train_config.exists():
        raise SystemExit(f"Config file does not exist: {args.train_config}")

    job = submit_training_job(
        config_path=args.train_config,
        hf_pref_repo_id=args.hf_pref_repo_id,
        hf_model_repo_id=args.hf_model_repo_id,
        token=args.token,
        private_repo=args.private_repo,
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace or None,
    )
    print(f"HF job submitted: {job.id}")
    print(f"HF job URL: {job.url}")
    if args.wait:
        final_job = wait_for_job(job.id, token=args.token, namespace=args.namespace or None)
        stage = final_job.status.stage if final_job.status is not None else "UNKNOWN"
        print(f"HF job completed with stage: {stage}")
        return 0 if stage == "COMPLETED" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
