from __future__ import annotations

import argparse
import json
import os
import random
import re
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
    return " ".join(str(value).split())


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


def extract_final_decision(response: str, allowed_tokens: list[str]) -> str:
    stripped = response.strip()
    for token in allowed_tokens:
        if stripped.endswith(token):
            return token
    return ""


def token_overlap_score(title: str, abstract: str, response: str) -> int:
    paper_tokens = set(TOKEN_PATTERN.findall(f"{title} {abstract}".lower()))
    response_tokens = set(TOKEN_PATTERN.findall(response.lower()))
    return 1 if len(paper_tokens & response_tokens) >= 3 else 0


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


def grade_response(row: dict[str, object], response: str, rubric: dict[str, object]) -> RubricResult:
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
    score = max(1, min(5, sum(dimension_scores.values())))
    rationale = [
        f"decision_correctness={dimension_scores['decision_correctness']}",
        f"paper_specificity={dimension_scores['paper_specificity']}",
        f"methodological_substance={dimension_scores['methodological_substance']}",
        f"uncertainty_calibration={dimension_scores['uncertainty_calibration']}",
        f"format_compliance={dimension_scores['format_compliance']}",
    ]
    return RubricResult(
        score=score,
        cot_rationale="\n".join(rationale),
        final_decision=final_decision,
        dimension_scores=dimension_scores,
    )


def collapse_prompt_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    prompts: list[dict[str, object]] = []
    for row in rows:
        prompt_id = clean_text(str(row.get("prompt_id") or row.get("pmid") or row.get("doi") or ""))
        if not prompt_id or prompt_id in seen:
            continue
        seen.add(prompt_id)
        prompts.append(row)
    return prompts


def sample_rows(rows: list[dict[str, object]], sample_size: int, seed: int) -> list[dict[str, object]]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return rows
    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)
    return copied[:sample_size]


def anthropic_messages_from_row(row: dict[str, object]) -> tuple[str, list[dict[str, str]]]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Missing messages in row: {row}")
    system_parts: list[str] = []
    chat_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = clean_text(message.get("role"))
        content = clean_text(message.get("content"))
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        else:
            chat_messages.append({"role": role, "content": content})
    return "\n\n".join(system_parts), chat_messages


def run_anthropic_model(
    rows: list[dict[str, object]],
    *,
    model_name: str,
    api_key: str,
    max_tokens: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit("anthropic is required for Claude benchmarking.") from exc

    client = anthropic.Anthropic(api_key=api_key)
    outputs: list[dict[str, object]] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for row in rows:
        system_text, messages = anthropic_messages_from_row(row)
        response = client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=0,
            system=system_text,
            messages=messages,
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()
        usage["input_tokens"] += int(getattr(response.usage, "input_tokens", 0))
        usage["output_tokens"] += int(getattr(response.usage, "output_tokens", 0))
        outputs.append({"prompt_id": row["prompt_id"], "response": text})
    return outputs, usage


def apply_chat_template(tokenizer, messages: list[dict[str, str]]):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def load_local_model(model_id: str, base_model_id: str | None = None):
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers and peft are required for local model benchmarking.") from exc

    auth_token = os.environ.get("HF_TOKEN", "")
    try:
        peft_config = PeftConfig.from_pretrained(model_id, token=auth_token or None)
        resolved_base = base_model_id or peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(resolved_base, trust_remote_code=True, token=auth_token or None)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            token=auth_token or None,
        )
        model = PeftModel.from_pretrained(model, model_id)
        return tokenizer, model
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=auth_token or None)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            token=auth_token or None,
        )
        return tokenizer, model


def run_local_model(
    rows: list[dict[str, object]],
    *,
    model_id: str,
    base_model_id: str | None,
    max_tokens: int,
) -> list[dict[str, object]]:
    tokenizer, model = load_local_model(model_id, base_model_id=base_model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs: list[dict[str, object]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Missing messages in row: {row}")
        input_ids = apply_chat_template(tokenizer, messages)
        input_ids = input_ids.to(model.device)
        generated = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completion_ids = generated[0][input_ids.shape[-1] :]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        outputs.append({"prompt_id": row["prompt_id"], "response": text})
    try:
        import gc
        import torch

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return outputs


def summarize_results(
    prompt_rows: list[dict[str, object]],
    model_name: str,
    responses: list[dict[str, object]],
    rubric: dict[str, object],
    extra_metrics: dict[str, int] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    response_by_prompt = {str(row["prompt_id"]): row["response"] for row in responses}
    graded_rows: list[dict[str, object]] = []
    metric_sums = {
        "rubric_score": 0,
        "decision_correctness": 0,
        "paper_specificity": 0,
        "methodological_substance": 0,
        "uncertainty_calibration": 0,
        "format_compliance": 0,
    }
    for row in prompt_rows:
        prompt_id = str(row["prompt_id"])
        response = clean_text(str(response_by_prompt.get(prompt_id, "")))
        grade = grade_response(row, response, rubric)
        graded = {
            "model_name": model_name,
            "prompt_id": prompt_id,
            "gold_decision": normalize_gold_decision(row),
            "response": response,
            "graded_decision": grade.final_decision,
            "rubric_score": grade.score,
            "grader_cot": grade.cot_rationale,
            "dimension_scores": grade.dimension_scores,
            "paper_title": clean_text(str(row.get("paper_title") or "")),
            "split": clean_text(str(row.get("split") or "test")) or "test",
        }
        graded_rows.append(graded)
        metric_sums["rubric_score"] += grade.score
        for key in grade.dimension_scores:
            metric_sums[key] += grade.dimension_scores[key]

    count = len(graded_rows) or 1
    summary = {
        "model_name": model_name,
        "prompt_count": len(graded_rows),
        "average_rubric_score": metric_sums["rubric_score"] / count,
        "decision_accuracy": metric_sums["decision_correctness"] / count,
        "paper_specificity_rate": metric_sums["paper_specificity"] / count,
        "methodological_substance_rate": metric_sums["methodological_substance"] / count,
        "uncertainty_calibration_rate": metric_sums["uncertainty_calibration"] / count,
        "format_compliance_rate": metric_sums["format_compliance"] / count,
    }
    if extra_metrics:
        summary.update(extra_metrics)
    return graded_rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a fine-tuned judge against base and API models.")
    parser.add_argument("--input", type=Path, required=True, help="Prompt rows JSONL. Deduped by prompt_id if needed.")
    parser.add_argument("--rubric", type=Path, required=True, help="Rubric JSON file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fine-tuned-model-id", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--anthropic-model", default="")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--anthropic-api-key", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rubric = load_rubric(args.rubric)
    prompt_rows = collapse_prompt_rows(load_jsonl(args.input))
    prompt_rows = sample_rows(prompt_rows, sample_size=args.sample_size, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, object]] = []

    base_responses = run_local_model(
        prompt_rows,
        model_id=args.base_model_id,
        base_model_id=None,
        max_tokens=args.max_new_tokens,
    )
    base_rows, base_summary = summarize_results(prompt_rows, "base_model", base_responses, rubric)
    write_jsonl(args.output_dir / "base_model_predictions.jsonl", base_rows)
    all_summaries.append(base_summary)

    tuned_responses = run_local_model(
        prompt_rows,
        model_id=args.fine_tuned_model_id,
        base_model_id=args.base_model_id,
        max_tokens=args.max_new_tokens,
    )
    tuned_rows, tuned_summary = summarize_results(prompt_rows, "fine_tuned_model", tuned_responses, rubric)
    write_jsonl(args.output_dir / "fine_tuned_model_predictions.jsonl", tuned_rows)
    all_summaries.append(tuned_summary)

    if args.anthropic_model and args.anthropic_api_key:
        anthropic_responses, usage = run_anthropic_model(
            prompt_rows,
            model_name=args.anthropic_model,
            api_key=args.anthropic_api_key,
            max_tokens=args.max_new_tokens,
        )
        anthropic_rows, anthropic_summary = summarize_results(
            prompt_rows,
            args.anthropic_model,
            anthropic_responses,
            rubric,
            extra_metrics=usage,
        )
        write_jsonl(args.output_dir / "anthropic_predictions.jsonl", anthropic_rows)
        all_summaries.append(anthropic_summary)

    (args.output_dir / "summary.json").write_text(
        json.dumps({"models": all_summaries, "prompt_count": len(prompt_rows)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
