from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlpaperdetector.dataset_builder import PubMedClient, fetch_pubmed_records
from rlpaperdetector.judge_data import DEFAULT_SYSTEM_PROMPT, build_messages, clean_text, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build held-out probe prompts for specific PMIDs.")
    parser.add_argument("--pmid", action="append", default=[], help="PMID to fetch. Can be repeated.")
    parser.add_argument("--pmid-file", type=Path, help="Text file with one PMID per line.")
    parser.add_argument("--output", type=Path, default=Path("preferences/probes/probe_papers.jsonl"))
    parser.add_argument("--crossref-email", default="test@example.org")
    parser.add_argument("--system-prompt-file", type=Path)
    return parser.parse_args()


def load_pmids(args: argparse.Namespace) -> list[str]:
    pmids = [clean_text(value) for value in args.pmid if clean_text(value)]
    if args.pmid_file and args.pmid_file.exists():
        for line in args.pmid_file.read_text(encoding="utf-8").splitlines():
            line = clean_text(line)
            if line and not line.startswith("#"):
                pmids.append(line)
    deduped: list[str] = []
    seen: set[str] = set()
    for pmid in pmids:
        if pmid not in seen:
            deduped.append(pmid)
            seen.add(pmid)
    if not deduped:
        raise SystemExit("Provide at least one --pmid or --pmid-file.")
    return deduped


def main() -> int:
    args = parse_args()
    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8").strip()
        if args.system_prompt_file
        else DEFAULT_SYSTEM_PROMPT
    )
    client = PubMedClient(args.crossref_email)
    records = fetch_pubmed_records(client, load_pmids(args))
    rows: list[dict[str, object]] = []
    for pmid, record in records.items():
        paper_row = {
            "pmid": record.pmid,
            "doi": record.doi,
            "title": record.title,
            "abstract": record.abstract,
            "journal": record.journal,
            "publication_year": str(record.publication_year or ""),
            "source": "probe_pubmed",
        }
        rows.append(
            {
                "prompt_id": pmid,
                "pmid": record.pmid,
                "doi": record.doi,
                "paper_title": record.title,
                "paper_abstract": record.abstract,
                "messages": build_messages(paper_row, system_prompt=system_prompt),
            }
        )
    write_jsonl(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
