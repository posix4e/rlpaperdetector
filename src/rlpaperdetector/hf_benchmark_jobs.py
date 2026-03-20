from __future__ import annotations

import argparse
import base64
import os
import re
from pathlib import Path

from rlpaperdetector.hf_jobs import AXOLOTL_CLOUD_IMAGE, TERMINAL_JOB_STAGES, wait_for_job


def encode_file_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9=_-]+", "_", value).strip("_")
    return cleaned[:256] or "value"


def build_benchmark_job_script() -> str:
    return """set -euo pipefail
cd /workspace
python -m pip install anthropic
mkdir -p benchmark_data benchmark_outputs
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_PREF_REPO_ID"],
    repo_type="dataset",
    local_dir="benchmark_data",
    token=os.environ["HF_TOKEN"],
)
snapshot_download(
    repo_id=os.environ["HF_MODEL_REPO_ID"],
    repo_type="model",
    local_dir="benchmark_model_repo",
    token=os.environ["HF_TOKEN"],
)
PY
python - <<'PY'
import base64
import os
from pathlib import Path

script_text = base64.b64decode(os.environ["BENCHMARK_SCRIPT_B64"]).decode("utf-8")
Path("benchmark_job.py").write_text(script_text, encoding="utf-8")
rubric_text = base64.b64decode(os.environ["RUBRIC_B64"]).decode("utf-8")
Path("benchmark_rubric.json").write_text(rubric_text, encoding="utf-8")
model_root = Path("benchmark_model_repo")
candidates = []
for marker in ("adapter_config.json", "config.json"):
    candidates.extend(model_root.rglob(marker))
selected = None
for path in candidates:
    if "checkpoint" in path.as_posix():
        continue
    selected = path.parent
    break
if selected is None:
    for path in candidates:
        selected = path.parent
        break
if selected is None:
    raise SystemExit("Could not find a loadable model directory inside benchmark_model_repo")
Path("fine_tuned_model_path.txt").write_text(str(selected), encoding="utf-8")
PY
python benchmark_job.py \
  --input benchmark_data/scored/test.jsonl \
  --rubric benchmark_rubric.json \
  --output-dir benchmark_outputs \
  --fine-tuned-model-id "$(cat fine_tuned_model_path.txt)" \
  --base-model-id "$BASE_MODEL_ID" \
  --anthropic-model "$ANTHROPIC_MODEL" \
  --sample-size "$SAMPLE_SIZE" \
  --seed "$SEED" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --anthropic-api-key "${ANTHROPIC_API_KEY:-}"
python - <<'PY'
import os
from datetime import datetime, timezone
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
path_in_repo = "benchmarks/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
api.upload_folder(
    repo_id=os.environ["HF_MODEL_REPO_ID"],
    repo_type="model",
    folder_path="benchmark_outputs",
    path_in_repo=path_in_repo,
    commit_message="Upload benchmark outputs from Hugging Face Job",
)
print(f"benchmark_path={path_in_repo}")
PY
"""


def submit_benchmark_job(
    *,
    benchmark_script_path: Path,
    rubric_path: Path,
    hf_pref_repo_id: str,
    hf_model_repo_id: str,
    base_model_id: str,
    anthropic_model: str,
    sample_size: int,
    seed: int,
    max_new_tokens: int,
    token: str,
    flavor: str,
    timeout: str,
    namespace: str | None = None,
    anthropic_api_key: str = "",
):
    try:
        from huggingface_hub import run_job
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required. Install it with `pip install huggingface_hub`.") from exc

    secrets = {"HF_TOKEN": token}
    if anthropic_api_key:
        secrets["ANTHROPIC_API_KEY"] = anthropic_api_key

    return run_job(
        image=AXOLOTL_CLOUD_IMAGE,
        command=["bash", "-lc", build_benchmark_job_script()],
        env={
            "HF_PREF_REPO_ID": hf_pref_repo_id,
            "HF_MODEL_REPO_ID": hf_model_repo_id,
            "BASE_MODEL_ID": base_model_id,
            "ANTHROPIC_MODEL": anthropic_model,
            "SAMPLE_SIZE": str(sample_size),
            "SEED": str(seed),
            "MAX_NEW_TOKENS": str(max_new_tokens),
            "BENCHMARK_SCRIPT_B64": encode_file_base64(benchmark_script_path),
            "RUBRIC_B64": encode_file_base64(rubric_path),
        },
        secrets=secrets,
        flavor=flavor,
        timeout=timeout,
        labels={
            "project": "rlpaperdetector",
            "task": "benchmark",
            "base_model": sanitize_label(base_model_id),
        },
        namespace=namespace,
        token=token,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a benchmark job to Hugging Face Jobs.")
    parser.add_argument("--hf-pref-repo-id", required=True)
    parser.add_argument("--hf-model-repo-id", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--benchmark-script", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, default=Path("configs/retraction_rubric.json"))
    parser.add_argument("--anthropic-model", default="")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--anthropic-api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.token:
        raise SystemExit("Missing Hugging Face token. Pass --token or set HF_TOKEN.")
    if not args.benchmark_script.exists():
        raise SystemExit(f"Benchmark script does not exist: {args.benchmark_script}")
    if not args.rubric.exists():
        raise SystemExit(f"Rubric file does not exist: {args.rubric}")
    job = submit_benchmark_job(
        benchmark_script_path=args.benchmark_script,
        rubric_path=args.rubric,
        hf_pref_repo_id=args.hf_pref_repo_id,
        hf_model_repo_id=args.hf_model_repo_id,
        base_model_id=args.base_model_id,
        anthropic_model=args.anthropic_model,
        sample_size=args.sample_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        token=args.token,
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace or None,
        anthropic_api_key=args.anthropic_api_key,
    )
    print(f"HF benchmark job submitted: {job.id}")
    print(f"HF benchmark job URL: {job.url}")
    if args.wait:
        final_job = wait_for_job(job.id, token=args.token, namespace=args.namespace or None)
        stage = final_job.status.stage if final_job.status is not None else "UNKNOWN"
        print(f"HF benchmark job completed with stage: {stage}")
        return 0 if stage == "COMPLETED" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
