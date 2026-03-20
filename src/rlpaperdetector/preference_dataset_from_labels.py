from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlpaperdetector.judge_data import (
    DEFAULT_SYSTEM_PROMPT,
    load_rows,
    sample_paper_rows,
    synthesize_candidate_rows,
    write_jsonl,
)
from rlpaperdetector.preference_pipeline import build_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Axolotl preference data from labeled paper rows.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to labeled dataset CSV or JSONL.")
    parser.add_argument("--output-dir", type=Path, default=Path("preferences"))
    parser.add_argument("--rubric", type=Path, default=Path("configs/retraction_rubric.json"))
    parser.add_argument("--max-papers", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-score-gap", type=int, default=1)
    parser.add_argument("--system-prompt-file", type=Path, help="Optional system prompt override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.dataset)
    sampled_rows = sample_paper_rows(rows, max_papers=args.max_papers, seed=args.seed)
    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8").strip()
        if args.system_prompt_file
        else DEFAULT_SYSTEM_PROMPT
    )
    candidates = synthesize_candidate_rows(sampled_rows, system_prompt=system_prompt)
    raw_dir = args.output_dir / "raw"
    write_jsonl(raw_dir / "candidate_responses.jsonl", candidates)
    summary = {
        "dataset": str(args.dataset),
        "sampled_papers": len(sampled_rows),
        "candidate_rows": len(candidates),
        "system_prompt": system_prompt,
    }
    (raw_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_outputs(
        raw_rows=candidates,
        output_dir=args.output_dir,
        rubric_path=args.rubric,
        min_score_gap=args.min_score_gap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
