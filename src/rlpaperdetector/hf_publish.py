from __future__ import annotations

import argparse
import os
from pathlib import Path


def publish_folder(repo_id: str, folder_path: Path, token: str, private: bool = False) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for publishing. Install it with `pip install huggingface_hub`.") from exc

    if not folder_path.exists():
        raise SystemExit(f"Folder does not exist: {folder_path}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder_path),
        commit_message="Upload dataset artifacts from GitHub Actions",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish dataset artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, for example username/rlpaperdetector.")
    parser.add_argument("--folder", type=Path, required=True, help="Folder containing files to upload.")
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private if it does not exist.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="Hugging Face token. Defaults to HF_TOKEN.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.token:
        raise SystemExit("Missing Hugging Face token. Pass --token or set HF_TOKEN.")
    publish_folder(
        repo_id=args.repo_id,
        folder_path=args.folder,
        token=args.token,
        private=args.private,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
