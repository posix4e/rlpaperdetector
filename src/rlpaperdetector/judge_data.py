from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are a strict retraction judge. Analyze the paper only from the provided metadata. "
    "Write a concise chain-of-thought style rationale grounded in the paper details, then end with exactly one final token: "
    "<KEEP> or <RETRACT>."
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]+")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_decision(label: str | int) -> str:
    return "<RETRACT>" if str(label) == "1" else "<KEEP>"


def build_user_prompt(row: dict[str, str]) -> str:
    title = clean_text(row.get("title"))
    abstract = clean_text(row.get("abstract"))
    journal = clean_text(row.get("journal"))
    year = clean_text(row.get("publication_year"))
    source = clean_text(row.get("source"))
    return (
        "Paper metadata:\n"
        f"Title: {title}\n"
        f"Journal: {journal}\n"
        f"Year: {year}\n"
        f"Source bucket: {source}\n"
        f"Abstract: {abstract}\n\n"
        "Explain the methodological or evidentiary signal you see, then give the final decision token."
    )


def build_messages(row: dict[str, str], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(row)},
    ]


def extract_design_phrase(abstract: str) -> str:
    lowered = abstract.lower()
    if "randomized" in lowered:
        return "The abstract describes a randomized design"
    if "observational" in lowered:
        return "The abstract describes an observational design"
    if "cross-sectional" in lowered:
        return "The abstract is cross-sectional"
    if "retrospective" in lowered:
        return "The abstract is retrospective"
    if "meta-analysis" in lowered:
        return "The abstract is a meta-analysis"
    if "mendelian randomization" in lowered:
        return "The abstract relies on Mendelian randomization"
    if "cohort" in lowered:
        return "The abstract describes a cohort analysis"
    return "The abstract provides limited design detail"


def extract_paper_keywords(row: dict[str, str], limit: int = 5) -> list[str]:
    title = clean_text(row.get("title"))
    abstract = clean_text(row.get("abstract"))
    seen: list[str] = []
    for token in TOKEN_PATTERN.findall(f"{title} {abstract}"):
        lowered = token.lower()
        if len(lowered) < 6:
            continue
        if lowered.isdigit():
            continue
        if lowered not in seen:
            seen.append(lowered)
        if len(seen) >= limit:
            break
    return seen


def strong_response(row: dict[str, str], gold_decision: str) -> str:
    abstract = clean_text(row.get("abstract"))
    keywords = extract_paper_keywords(row)
    keyword_phrase = ", ".join(keywords[:3]) if keywords else "paper-specific details"
    design_phrase = extract_design_phrase(abstract)
    if gold_decision == "<RETRACT>":
        body = (
            f"{design_phrase}, and the paper language around {keyword_phrase} does not inspire confidence. "
            "The safest rubric-aligned decision is that the record belongs in the retract bucket, even if the abstract alone does not isolate the exact defect. "
            "This should be treated as a serious methodological or evidentiary failure rather than a minor fix."
        )
    else:
        body = (
            f"{design_phrase}, and the abstract grounds the claim in {keyword_phrase}. "
            "From the text alone I do not see enough evidence for retraction, so the calibrated decision is to keep it while acknowledging that abstract-only review is limited."
        )
    return f"{body}\n{gold_decision}"


def generic_correct_response(gold_decision: str) -> str:
    if gold_decision == "<RETRACT>":
        return (
            "The paper raises serious reliability concerns and should not remain in the keep bucket. "
            "The right decision is retraction after review.\n<RETRACT>"
        )
    return (
        "The abstract alone does not justify retraction. "
        "The safer decision is to keep it unless stronger contrary evidence appears.\n<KEEP>"
    )


def wrong_response(gold_decision: str) -> str:
    wrong = "<KEEP>" if gold_decision == "<RETRACT>" else "<RETRACT>"
    if wrong == "<RETRACT>":
        return "The result feels overstated, so I would retract it immediately.\n<RETRACT>"
    return "This looks broadly fine to me and should stay in the literature.\n<KEEP>"


def malformed_response(row: dict[str, str], gold_decision: str) -> str:
    title = clean_text(row.get("title"))
    if gold_decision == "<RETRACT>":
        return f"{title} looks suspicious but I am not going to follow the exact requested format."
    return f"{title} seems okay and probably does not need action."


def synthesize_candidate_rows(
    paper_rows: list[dict[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in paper_rows:
        gold_decision = normalize_decision(row["label"])
        base = {
            "prompt_id": clean_text(row.get("pmid")) or clean_text(row.get("doi")) or clean_text(row.get("title")),
            "pmid": clean_text(row.get("pmid")),
            "doi": clean_text(row.get("doi")),
            "split": clean_text(row.get("split")) or "train",
            "paper_title": clean_text(row.get("title")),
            "paper_abstract": clean_text(row.get("abstract")),
            "messages": build_messages(row, system_prompt=system_prompt),
            "gold_decision": gold_decision,
        }
        responses = [
            strong_response(row, gold_decision),
            generic_correct_response(gold_decision),
            wrong_response(gold_decision),
            malformed_response(row, gold_decision),
        ]
        for index, response in enumerate(responses):
            candidates.append(
                {
                    **base,
                    "candidate_id": f"{base['prompt_id']}::cand{index}",
                    "response": response,
                }
            )
    return candidates


def sample_paper_rows(rows: list[dict[str, str]], max_papers: int, seed: int) -> list[dict[str, str]]:
    usable = [row for row in rows if clean_text(row.get("title")) and clean_text(row.get("abstract"))]
    if len(usable) <= max_papers:
        return usable

    rng = random.Random(seed)
    by_split_and_label: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in usable:
        key = (clean_text(row.get("split")) or "train", str(row.get("label", "0")))
        by_split_and_label.setdefault(key, []).append(row)

    sampled: list[dict[str, str]] = []
    total = len(usable)
    for key, bucket in sorted(by_split_and_label.items()):
        rng.shuffle(bucket)
        take = max(1, round(max_papers * len(bucket) / total))
        sampled.extend(bucket[:take])

    if len(sampled) > max_papers:
        rng.shuffle(sampled)
        sampled = sampled[:max_papers]
    elif len(sampled) < max_papers:
        seen_ids = {id(row) for row in sampled}
        leftovers = [row for row in usable if id(row) not in seen_ids]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: max_papers - len(sampled)])
    return sampled
