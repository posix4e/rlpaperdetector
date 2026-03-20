from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from rlpaperdetector.baseline import load_model
from rlpaperdetector.dataset_builder import PubMedClient, PubMedRecord, fetch_pubmed_records


DATE_FIELD_TAGS = {
    "pdat": "PDAT",
    "edat": "EDAT",
    "crdt": "CRDT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor new subject-matched PubMed papers and score them.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model.json.")
    parser.add_argument("--query", action="append", default=[], help="PubMed query for a relevant subject. Can be repeated.")
    parser.add_argument("--query-file", type=Path, help="Text file with one PubMed query per line.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/monitor"))
    parser.add_argument("--state-file", type=Path, default=Path("state/seen_pmids.json"))
    parser.add_argument("--days-back", type=int, default=1, help="How many days of records to inspect, inclusive of end date.")
    parser.add_argument("--end-date", help="UTC end date in YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--date-field", choices=["pdat", "edat", "crdt"], default="edat")
    parser.add_argument("--max-results-per-query", type=int, default=200)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--crossref-email", default="test@example.org", help="Email passed to NCBI requests.")
    return parser.parse_args()


def parse_iso_date(raw_value: str | None) -> date:
    if raw_value:
        return date.fromisoformat(raw_value)
    return datetime.now(timezone.utc).date()


def format_pubmed_date(value: date) -> str:
    return value.strftime("%Y/%m/%d")


def load_queries(args: argparse.Namespace) -> list[str]:
    queries = [query.strip() for query in args.query if query.strip()]
    if args.query_file:
        for line in args.query_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if query not in seen:
            deduped.append(query)
            seen.add(query)
    if not deduped:
        raise SystemExit("Provide at least one --query or a --query-file.")
    return deduped


def build_date_filtered_query(query: str, start_date: date, end_date: date, date_field: str) -> str:
    tag = DATE_FIELD_TAGS[date_field]
    return f"({query}) AND ({format_pubmed_date(start_date)}:{format_pubmed_date(end_date)}[{tag}])"


def search_query_pmids(client: PubMedClient, query: str, max_results: int) -> list[str]:
    pmids: list[str] = []
    retstart = 0
    batch_size = min(100, max_results)
    while retstart < max_results:
        batch = client.esearch(term=query, retmax=min(batch_size, max_results - retstart), retstart=retstart)
        if not batch:
            break
        pmids.extend(batch)
        if len(batch) < batch_size:
            break
        retstart += len(batch)
    return pmids


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"seen_pmids": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def pubmed_record_to_row(record: PubMedRecord) -> dict[str, str]:
    return {
        "pmid": record.pmid,
        "doi": record.doi,
        "title": record.title,
        "abstract": record.abstract,
        "journal": record.journal,
        "publication_year": str(record.publication_year or ""),
    }


def write_outputs(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "scored_papers.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    csv_path = output_dir / "scored_papers.csv"
    fieldnames = [
        "pmid",
        "doi",
        "score",
        "predicted_label",
        "journal",
        "publication_year",
        "matched_queries",
        "title",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    model, metadata = load_model(args.model)
    threshold = float(metadata.get("threshold", 0.5))
    queries = load_queries(args)
    end_date = parse_iso_date(args.end_date)
    start_date = end_date - timedelta(days=max(args.days_back - 1, 0))

    client = PubMedClient(args.crossref_email)
    state = load_state(args.state_file)
    seen_pmids = set(str(value) for value in state.get("seen_pmids", []))

    query_matches: dict[str, set[str]] = {}
    for query in queries:
        filtered_query = build_date_filtered_query(query, start_date, end_date, args.date_field)
        for pmid in search_query_pmids(client, filtered_query, args.max_results_per_query):
            query_matches.setdefault(pmid, set()).add(query)

    unseen_pmids = [pmid for pmid in query_matches if pmid not in seen_pmids]
    records = fetch_pubmed_records(client, unseen_pmids)

    rows: list[dict[str, object]] = []
    for pmid in unseen_pmids:
        record = records.get(pmid)
        if not record or not record.has_usable_text:
            continue
        prediction = model.predict(pubmed_record_to_row(record), threshold=threshold)
        row = {
            "pmid": record.pmid,
            "doi": record.doi,
            "score": prediction["score"],
            "predicted_label": prediction["label"],
            "journal": record.journal,
            "publication_year": record.publication_year or "",
            "matched_queries": " | ".join(sorted(query_matches[pmid])),
            "title": record.title,
        }
        if float(row["score"]) >= args.min_score:
            rows.append(row)

    rows.sort(key=lambda row: (-float(row["score"]), str(row["pmid"])))
    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_field": args.date_field,
        "date_start": start_date.isoformat(),
        "date_end": end_date.isoformat(),
        "queries": queries,
        "matched_pmids": len(query_matches),
        "new_pmids": len(unseen_pmids),
        "scored_rows": len(rows),
        "threshold": threshold,
        "min_score": args.min_score,
    }
    write_outputs(args.output_dir, rows, summary)

    state["seen_pmids"] = sorted(seen_pmids.union(query_matches))
    state["last_run_utc"] = summary["run_at_utc"]
    state["last_queries"] = queries
    save_state(args.state_file, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
