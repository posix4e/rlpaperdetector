from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DECISION_MAP = {
    "0": "<KEEP>",
    "1": "<RETRACT>",
    "keep": "<KEEP>",
    "retract": "<RETRACT>",
    "<keep>": "<KEEP>",
    "<retract>": "<RETRACT>",
}

METHODOLOGY_TERMS = {
    "confound",
    "confounding",
    "selection",
    "bias",
    "causal",
    "observational",
    "randomized",
    "underpowered",
    "sample",
    "method",
    "design",
    "measurement",
    "robustness",
    "sensitivity",
    "replication",
    "reanalysis",
    "heterogeneity",
    "subgroup",
    "baseline",
}
CALIBRATION_TERMS = {
    "uncertain",
    "uncertainty",
    "review",
    "needs review",
    "further review",
    "possible",
    "suggests",
    "may",
    "might",
    "appears",
    "preliminary",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class RubricResult:
    score: int
    cot_rationale: str
    final_decision: str
    dimension_scores: dict[str, int]


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_rubric(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_gold_decision(row: dict[str, object]) -> str:
    for key in ("gold_decision", "decision", "label", "gold_label"):
        value = row.get(key)
        if value is None:
            continue
        normalized = DECISION_MAP.get(str(value).strip().lower())
        if normalized:
            return normalized
    raise ValueError(f"Missing recognizable gold decision in row: {row}")


def normalize_response_text(row: dict[str, object]) -> str:
    for key in ("response", "completion", "text"):
        value = row.get(key)
        if value:
            return clean_text(str(value))
    raise ValueError(f"Missing response text in row: {row}")


def extract_messages(row: dict[str, object]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, dict):
                normalized.append(
                    {
                        "role": clean_text(str(message.get("role", "user"))),
                        "content": clean_text(str(message.get("content", ""))),
                    }
                )
        if normalized:
            return normalized
    prompt = clean_text(str(row.get("prompt", "")))
    system = clean_text(str(row.get("system", "")))
    result: list[dict[str, str]] = []
    if system:
        result.append({"role": "system", "content": system})
    if prompt:
        result.append({"role": "user", "content": prompt})
    if result:
        return result
    raise ValueError(f"Missing messages/prompt in row: {row}")


def extract_final_decision(response: str, allowed_tokens: list[str]) -> str:
    stripped = response.strip()
    for token in allowed_tokens:
        if stripped.endswith(token):
            return token
    return ""


def token_overlap_score(title: str, abstract: str, response: str) -> int:
    paper_tokens = set(TOKEN_PATTERN.findall(f"{title} {abstract}".lower()))
    response_tokens = set(TOKEN_PATTERN.findall(response.lower()))
    overlap = paper_tokens & response_tokens
    return 1 if len(overlap) >= 3 else 0


def methodology_score(response: str) -> int:
    lowered = response.lower()
    return 1 if any(term in lowered for term in METHODOLOGY_TERMS) else 0


def calibration_score(response: str) -> int:
    lowered = response.lower()
    return 1 if any(term in lowered for term in CALIBRATION_TERMS) else 0


def format_score(response: str, final_decision: str) -> int:
    if not final_decision:
        return 0
    rationale = response[: -len(final_decision)].strip()
    return 1 if rationale else 0


def grade_response(row: dict[str, object], rubric: dict[str, object]) -> RubricResult:
    response = normalize_response_text(row)
    gold_decision = normalize_gold_decision(row)
    final_decision = extract_final_decision(response, list(rubric["decision_tokens"]))
    title = clean_text(str(row.get("paper_title") or row.get("title") or ""))
    abstract = clean_text(str(row.get("paper_abstract") or row.get("abstract") or ""))

    dimension_scores = {
        "decision_correctness": 1 if final_decision == gold_decision else 0,
        "paper_specificity": token_overlap_score(title, abstract, response),
        "methodological_substance": methodology_score(response),
        "uncertainty_calibration": calibration_score(response),
        "format_compliance": format_score(response, final_decision),
    }
    total = sum(dimension_scores.values())
    score = max(1, min(5, total))

    rationale_lines = [
        f"decision_correctness={dimension_scores['decision_correctness']}: expected {gold_decision}, got {final_decision or 'missing token'}",
        f"paper_specificity={dimension_scores['paper_specificity']}: {'paper-specific overlap found' if dimension_scores['paper_specificity'] else 'response is too generic'}",
        f"methodological_substance={dimension_scores['methodological_substance']}: {'method/evidence critique present' if dimension_scores['methodological_substance'] else 'little methodological substance'}",
        f"uncertainty_calibration={dimension_scores['uncertainty_calibration']}: {'review/calibration language present' if dimension_scores['uncertainty_calibration'] else 'overconfident or uncalibrated'}",
        f"format_compliance={dimension_scores['format_compliance']}: {'rationale plus final token' if dimension_scores['format_compliance'] else 'missing rationale or terminal token'}",
    ]
    return RubricResult(
        score=score,
        cot_rationale="\n".join(rationale_lines),
        final_decision=final_decision,
        dimension_scores=dimension_scores,
    )


def prompt_key(row: dict[str, object]) -> str:
    for key in ("prompt_id", "paper_id", "group_id", "pmid", "doi"):
        value = clean_text(str(row.get(key, "")))
        if value:
            return value
    return json.dumps(extract_messages(row), sort_keys=True)


def score_gap_to_weight(score_gap: int) -> float:
    return max(0.25, min(1.0, score_gap / 4.0))


def score_gap_to_repeats(score_gap: int) -> int:
    return max(1, min(4, score_gap))


def build_preference_exports(
    scored_rows: list[dict[str, object]],
    min_score_gap: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in scored_rows:
        grouped[str(row["prompt_id"])].append(row)

    sft_rows: list[dict[str, object]] = []
    dpo_rows: list[dict[str, object]] = []
    orpo_rows: list[dict[str, object]] = []

    for prompt_id, candidates in grouped.items():
        ordered = sorted(candidates, key=lambda row: (-int(row["rubric_score"]), str(row["response"])))
        best = ordered[0]
        if int(best["rubric_score"]) >= 4:
            sft_rows.append(
                {
                    "messages": best["messages"],
                    "completion": best["response"],
                    "prompt_id": prompt_id,
                    "rubric_score": best["rubric_score"],
                    "grader_cot": best["grader_cot"],
                }
            )

        for index, chosen in enumerate(ordered):
            for rejected in ordered[index + 1 :]:
                score_gap = int(chosen["rubric_score"]) - int(rejected["rubric_score"])
                if score_gap < min_score_gap:
                    continue
                pair_weight = score_gap_to_weight(score_gap)
                pair_repeats = score_gap_to_repeats(score_gap)
                metadata = {
                    "prompt_id": prompt_id,
                    "chosen_score": chosen["rubric_score"],
                    "rejected_score": rejected["rubric_score"],
                    "score_gap": score_gap,
                    "pair_weight": pair_weight,
                    "pair_repeats": pair_repeats,
                }
                dpo_rows.append(
                    {
                        "messages": chosen["messages"],
                        "chosen": {"role": "assistant", "content": chosen["response"]},
                        "rejected": {"role": "assistant", "content": rejected["response"]},
                        **metadata,
                    }
                )
                system_message = next((m["content"] for m in chosen["messages"] if m["role"] == "system"), "")
                non_system_messages = [m for m in chosen["messages"] if m["role"] != "system"]
                chosen_messages = non_system_messages + [{"role": "assistant", "content": chosen["response"]}]
                rejected_messages = non_system_messages + [{"role": "assistant", "content": rejected["response"]}]
                orpo_rows.append(
                    {
                        "system": system_message,
                        "prompt": non_system_messages[-1]["content"] if non_system_messages else "",
                        "chosen": chosen_messages,
                        "rejected": rejected_messages,
                        **metadata,
                    }
                )
    return sft_rows, dpo_rows, orpo_rows


def expand_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    expanded: list[dict[str, object]] = []
    for row in rows:
        repeats = int(row.get("pair_repeats", 1))
        for repeat_index in range(repeats):
            duplicated = dict(row)
            duplicated["pair_repeat_index"] = repeat_index
            expanded.append(duplicated)
    return expanded


def split_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    buckets = {"train": [], "validation": [], "test": []}
    for row in rows:
        split = clean_text(str(row.get("split", "train"))) or "train"
        if split not in buckets:
            split = "train"
        buckets[split].append(row)
    return buckets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade candidate responses and build Axolotl preference datasets.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL file with one candidate response per row.")
    parser.add_argument("--output-dir", type=Path, default=Path("preferences"))
    parser.add_argument("--rubric", type=Path, default=Path("configs/retraction_rubric.json"))
    parser.add_argument("--min-score-gap", type=int, default=1)
    return parser.parse_args()


def build_outputs(
    raw_rows: list[dict[str, object]],
    output_dir: Path,
    rubric_path: Path,
    min_score_gap: int,
) -> int:
    rubric = load_rubric(rubric_path)
    scored_rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        messages = extract_messages(raw_row)
        grade = grade_response(raw_row, rubric)
        scored_rows.append(
            {
                **raw_row,
                "prompt_id": prompt_key(raw_row),
                "messages": messages,
                "response": normalize_response_text(raw_row),
                "gold_decision": normalize_gold_decision(raw_row),
                "rubric_score": grade.score,
                "grader_cot": grade.cot_rationale,
                "graded_decision": grade.final_decision,
                "dimension_scores": grade.dimension_scores,
                "split": clean_text(str(raw_row.get("split", "train"))) or "train",
            }
        )

    split_scored = split_rows(scored_rows)
    scored_dir = output_dir / "scored"
    write_jsonl(scored_dir / "all.jsonl", scored_rows)
    for split, rows in split_scored.items():
        write_jsonl(scored_dir / f"{split}.jsonl", rows)

    sft_rows, dpo_rows, orpo_rows = build_preference_exports(scored_rows, min_score_gap)
    split_sft = split_rows(sft_rows)
    split_dpo = split_rows(dpo_rows)
    split_orpo = split_rows(orpo_rows)

    axolotl_dir = output_dir / "axolotl"
    write_jsonl(axolotl_dir / "sft_all.jsonl", sft_rows)
    write_jsonl(axolotl_dir / "dpo_all.jsonl", dpo_rows)
    write_jsonl(axolotl_dir / "orpo_all.jsonl", orpo_rows)

    for split, rows in split_sft.items():
        write_jsonl(axolotl_dir / f"sft_{split}.jsonl", rows)
    for split, rows in split_dpo.items():
        write_jsonl(axolotl_dir / f"dpo_{split}.jsonl", rows)
        write_jsonl(axolotl_dir / f"dpo_{split}_expanded.jsonl", expand_pairs(rows))
    for split, rows in split_orpo.items():
        write_jsonl(axolotl_dir / f"orpo_{split}.jsonl", rows)
        write_jsonl(axolotl_dir / f"orpo_{split}_expanded.jsonl", expand_pairs(rows))

    summary = {
        "input_rows": len(raw_rows),
        "scored_rows": len(scored_rows),
        "sft_rows": len(sft_rows),
        "dpo_pairs": len(dpo_rows),
        "orpo_pairs": len(orpo_rows),
        "dpo_pairs_expanded": len(expand_pairs(dpo_rows)),
        "orpo_pairs_expanded": len(expand_pairs(orpo_rows)),
        "rubric": str(rubric_path),
        "min_score_gap": min_score_gap,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()
    raw_rows = load_jsonl(args.input)
    return build_outputs(
        raw_rows=raw_rows,
        output_dir=args.output_dir,
        rubric_path=args.rubric,
        min_score_gap=args.min_score_gap,
    )


if __name__ == "__main__":
    raise SystemExit(main())
