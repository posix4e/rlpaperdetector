from __future__ import annotations

import json
from pathlib import Path


DEFAULT_EXCLUSIONS_PATH = Path("configs/excluded_papers.json")


def load_exclusions(path: Path | None = None) -> dict[str, set[str]]:
    source = path or DEFAULT_EXCLUSIONS_PATH
    if not source.exists():
        return {"pmids": set(), "dois": set(), "title_substrings": set()}

    payload = json.loads(source.read_text(encoding="utf-8"))
    return {
        "pmids": {str(value).strip() for value in payload.get("pmids", []) if str(value).strip()},
        "dois": {str(value).strip().lower() for value in payload.get("dois", []) if str(value).strip()},
        "title_substrings": {
            str(value).strip().lower() for value in payload.get("title_substrings", []) if str(value).strip()
        },
    }


def row_is_excluded(row: dict[str, object], exclusions: dict[str, set[str]]) -> bool:
    pmid = str(row.get("pmid", "")).strip()
    doi = str(row.get("doi", "")).strip().lower()
    title = str(row.get("title", "")).strip().lower()
    matched_positive_pmid = str(row.get("matched_positive_pmid", "")).strip()

    if pmid and pmid in exclusions["pmids"]:
        return True
    if matched_positive_pmid and matched_positive_pmid in exclusions["pmids"]:
        return True
    if doi and doi in exclusions["dois"]:
        return True
    return any(fragment in title for fragment in exclusions["title_substrings"])


def pubmed_record_is_excluded(
    pmid: str,
    doi: str,
    title: str,
    exclusions: dict[str, set[str]],
) -> bool:
    if pmid and pmid in exclusions["pmids"]:
        return True
    if doi and doi.lower() in exclusions["dois"]:
        return True
    lowered_title = title.lower()
    return any(fragment in lowered_title for fragment in exclusions["title_substrings"])
