from __future__ import annotations

import argparse
import csv
import http.client
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CROSSREF_RETRACTION_WATCH_URL = "https://api.labs.crossref.org/data/retractionwatch?{email}"
DEFAULT_CROSSREF_EMAIL = "test@example.org"
NCBI_TOOL_NAME = "rlpaperdetector"
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMID_PATTERN = re.compile(r"\b\d+\b")
SPLIT_NAMES = ("train", "validation", "test")


def log(message: str) -> None:
    print(message, file=sys.stderr)


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def normalize_pmid(raw_value: str | None) -> str | None:
    if not raw_value:
        return None
    match = PMID_PATTERN.search(raw_value)
    if not match:
        return None
    pmid = match.group(0)
    return None if pmid == "0" else pmid


def normalize_doi(raw_value: str | None) -> str:
    return clean_text(raw_value).lower()


def iter_chunks(items: Iterable[str], chunk_size: int) -> Iterable[list[str]]:
    chunk: list[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def extract_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return clean_text("".join(element.itertext()))


def append_unique(values: list[str], candidate: str) -> None:
    if candidate and candidate not in values:
        values.append(candidate)


@dataclass
class PubMedRecord:
    pmid: str
    doi: str
    title: str
    abstract: str
    journal: str
    publication_year: int | None
    publication_types: list[str]

    @property
    def is_retracted_publication(self) -> bool:
        lowered = {value.casefold() for value in self.publication_types}
        return "retracted publication" in lowered or "retraction of publication" in lowered

    @property
    def has_usable_text(self) -> bool:
        return bool(self.title and self.abstract)


class PubMedClient:
    def __init__(self, email: str, delay_seconds: float = 0.34) -> None:
        self.email = email
        self.delay_seconds = delay_seconds
        self._last_request_at = 0.0

    def _request(self, endpoint: str, params: dict[str, str]) -> bytes:
        params = {
            **params,
            "tool": NCBI_TOOL_NAME,
            "email": self.email,
        }
        query = urllib.parse.urlencode(params)
        url = f"{NCBI_BASE_URL}/{endpoint}?{query}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": f"{NCBI_TOOL_NAME}/0.1 ({self.email})"},
        )

        backoff_seconds = self.delay_seconds
        for attempt in range(7):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.delay_seconds:
                time.sleep(self.delay_seconds - elapsed)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                self._last_request_at = time.monotonic()
                return payload
            except urllib.error.HTTPError as error:
                self._last_request_at = time.monotonic()
                if error.code not in {429, 500, 502, 503, 504} or attempt == 6:
                    raise
                retry_after = error.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after and retry_after.isdigit() else backoff_seconds
                log(f"NCBI returned HTTP {error.code}; retrying in {sleep_seconds:.2f}s")
                time.sleep(sleep_seconds)
                backoff_seconds *= 2
            except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected) as error:
                self._last_request_at = time.monotonic()
                if attempt == 6:
                    raise
                log(f"NCBI request failed with {type(error).__name__}; retrying in {backoff_seconds:.2f}s")
                time.sleep(backoff_seconds)
                backoff_seconds *= 2
        raise RuntimeError(f"Failed to fetch {url}")

    def esearch(self, term: str, retmax: int, retstart: int = 0) -> list[str]:
        payload = self._request(
            "esearch.fcgi",
            {
                "db": "pubmed",
                "retmode": "json",
                "retmax": str(retmax),
                "retstart": str(retstart),
                "term": term,
            },
        )
        data = json.loads(payload)
        return data["esearchresult"].get("idlist", [])

    def efetch(self, pmids: list[str]) -> dict[str, PubMedRecord]:
        if not pmids:
            return {}
        payload = self._request(
            "efetch.fcgi",
            {
                "db": "pubmed",
                "retmode": "xml",
                "id": ",".join(pmids),
            },
        )
        root = ET.fromstring(payload)
        records: dict[str, PubMedRecord] = {}
        for article in root.findall(".//PubmedArticle"):
            record = parse_pubmed_article(article)
            if record:
                records[record.pmid] = record
        return records


def parse_pubmed_article(article: ET.Element) -> PubMedRecord | None:
    pmid = extract_text(article.find("./MedlineCitation/PMID"))
    if not pmid:
        return None

    article_node = article.find("./MedlineCitation/Article")
    title = extract_text(article_node.find("./ArticleTitle") if article_node is not None else None)

    abstract_parts: list[str] = []
    if article_node is not None:
        for section in article_node.findall("./Abstract/AbstractText"):
            label = clean_text(section.attrib.get("Label"))
            body = extract_text(section)
            if label and body:
                abstract_parts.append(f"{label}: {body}")
            elif body:
                abstract_parts.append(body)
    abstract = clean_text("\n".join(abstract_parts))

    journal_candidates: list[str] = []
    if article_node is not None:
        journal = article_node.find("./Journal")
        append_unique(journal_candidates, extract_text(journal.find("./ISOAbbreviation") if journal is not None else None))
        append_unique(journal_candidates, extract_text(journal.find("./Title") if journal is not None else None))
    append_unique(journal_candidates, extract_text(article.find("./MedlineCitation/MedlineJournalInfo/MedlineTA")))
    journal_name = journal_candidates[0] if journal_candidates else ""

    publication_types = [
        extract_text(node)
        for node in article.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")
        if extract_text(node)
    ]

    doi = ""
    for article_id in article.findall("./PubmedData/ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == "doi":
            doi = normalize_doi(extract_text(article_id))
            break

    publication_year = extract_publication_year(article)
    return PubMedRecord(
        pmid=pmid,
        doi=doi,
        title=title,
        abstract=abstract,
        journal=journal_name,
        publication_year=publication_year,
        publication_types=publication_types,
    )


def extract_publication_year(article: ET.Element) -> int | None:
    year_paths = [
        "./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year",
        "./MedlineCitation/Article/ArticleDate/Year",
        "./PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year",
    ]
    for path in year_paths:
        year_text = extract_text(article.find(path))
        if year_text.isdigit():
            return int(year_text)
    medline_date = extract_text(article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate"))
    match = re.search(r"\b(19|20)\d{2}\b", medline_date)
    if match:
        return int(match.group(0))
    return None


def download_retraction_watch_csv(email: str, destination: Path, force: bool = False) -> Path:
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = CROSSREF_RETRACTION_WATCH_URL.format(email=urllib.parse.quote(email, safe="@"))
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"{NCBI_TOOL_NAME}/0.1 ({email})",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        destination.write_bytes(response.read())
    return destination


def load_positive_seed_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    positives: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pmid = normalize_pmid(row.get("OriginalPaperPubMedID"))
            if not pmid:
                continue

            nature = clean_text(row.get("RetractionNature"))
            if nature and "retraction" not in nature.casefold():
                continue

            if pmid in positives:
                continue

            positives[pmid] = {
                "rw_record_id": clean_text(row.get("Record ID")),
                "rw_title": clean_text(row.get("Title")),
                "rw_journal": clean_text(row.get("Journal")),
                "rw_retraction_nature": nature,
                "rw_retraction_reason": clean_text(row.get("Reason")),
                "rw_retraction_date": clean_text(row.get("RetractionDate")),
                "rw_retraction_doi": normalize_doi(row.get("RetractionDOI")),
                "rw_retraction_pmid": normalize_pmid(row.get("RetractionPubMedID")) or "",
                "original_doi_from_rw": normalize_doi(row.get("OriginalPaperDOI")),
            }
    return positives


def fetch_pubmed_records(client: PubMedClient, pmids: Iterable[str], batch_size: int = 100) -> dict[str, PubMedRecord]:
    records: dict[str, PubMedRecord] = {}
    for batch in iter_chunks(list(pmids), batch_size):
        records.update(client.efetch(batch))
        log(f"Fetched PubMed metadata for {len(records)} records so far")
    return records


def build_group_key(record: PubMedRecord) -> tuple[str, int] | None:
    if not record.journal or record.publication_year is None:
        return None
    return (record.journal, record.publication_year)


def choose_negative_query(record: PubMedRecord) -> list[str]:
    exclusion = 'NOT ("Retracted Publication"[PT] OR "Retraction of Publication"[PT] OR "Published Erratum"[PT])'
    queries: list[str] = []
    if record.journal and record.publication_year is not None:
        queries.append(f'("{record.journal}"[TA] AND {record.publication_year}[PDAT]) {exclusion}')
    if record.publication_year is not None:
        queries.append(f"({record.publication_year}[PDAT]) {exclusion}")
    return queries


def sample_negative_records(
    client: PubMedClient,
    positives: dict[str, PubMedRecord],
    negatives_per_positive: int,
    seed: int,
    candidate_pool_size: int,
) -> dict[str, list[PubMedRecord]]:
    rng = random.Random(seed)
    positive_pmids = set(positives)
    grouped_positive_ids: dict[tuple[str, int], list[str]] = defaultdict(list)
    fallback_positive_ids: list[str] = []
    for pmid, record in positives.items():
        key = build_group_key(record)
        if key is None:
            fallback_positive_ids.append(pmid)
        else:
            grouped_positive_ids[key].append(pmid)

    matches: dict[str, list[PubMedRecord]] = {pmid: [] for pmid in positives}
    metadata_cache: dict[str, PubMedRecord] = {}

    def assign_group(positive_ids: list[str], exemplar: PubMedRecord) -> None:
        needed = len(positive_ids) * negatives_per_positive
        chosen: list[PubMedRecord] = []
        seen_negative_pmids: set[str] = set()
        for term in choose_negative_query(exemplar):
            candidate_pmids = client.esearch(term=term, retmax=candidate_pool_size)
            rng.shuffle(candidate_pmids)
            filtered_candidate_pmids = [
                pmid for pmid in candidate_pmids if pmid not in positive_pmids and pmid not in seen_negative_pmids
            ]
            for batch in iter_chunks(filtered_candidate_pmids, 100):
                missing = [pmid for pmid in batch if pmid not in metadata_cache]
                if missing:
                    metadata_cache.update(client.efetch(missing))
                for pmid in batch:
                    record = metadata_cache.get(pmid)
                    if not record or not record.has_usable_text or record.is_retracted_publication:
                        continue
                    chosen.append(record)
                    seen_negative_pmids.add(record.pmid)
                    if len(chosen) >= needed:
                        break
                if len(chosen) >= needed:
                    break
            if len(chosen) >= needed:
                break

        if len(chosen) < needed:
            log(
                f"Warning: found only {len(chosen)} negatives for group "
                f"{exemplar.journal!r}, {exemplar.publication_year}; needed {needed}"
            )

        index = 0
        for positive_id in positive_ids:
            for _ in range(negatives_per_positive):
                if index >= len(chosen):
                    return
                matches[positive_id].append(chosen[index])
                index += 1

    for positive_ids in grouped_positive_ids.values():
        exemplar = positives[positive_ids[0]]
        assign_group(positive_ids, exemplar)

    for positive_id in fallback_positive_ids:
        assign_group([positive_id], positives[positive_id])

    return matches


def build_dataset_rows(
    seeds: dict[str, dict[str, str]],
    positive_records: dict[str, PubMedRecord],
    negative_matches: dict[str, list[PubMedRecord]],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for pmid, record in positive_records.items():
        seed = seeds[pmid]
        rows.append(
            {
                "label": 1,
                "pmid": record.pmid,
                "doi": record.doi or seed["original_doi_from_rw"],
                "title": record.title,
                "abstract": record.abstract,
                "journal": record.journal,
                "publication_year": record.publication_year or "",
                    "source": "retraction_watch_pubmed",
                    "matched_positive_pmid": "",
                    "group_id": "",
                    "split": "",
                    "rw_record_id": seed["rw_record_id"],
                    "rw_retraction_nature": seed["rw_retraction_nature"],
                    "rw_retraction_reason": seed["rw_retraction_reason"],
                    "rw_retraction_date": seed["rw_retraction_date"],
                    "rw_retraction_doi": seed["rw_retraction_doi"],
                "rw_retraction_pmid": seed["rw_retraction_pmid"],
            }
        )
        for negative_record in negative_matches.get(pmid, []):
            rows.append(
                {
                    "label": 0,
                    "pmid": negative_record.pmid,
                    "doi": negative_record.doi,
                    "title": negative_record.title,
                    "abstract": negative_record.abstract,
                    "journal": negative_record.journal,
                    "publication_year": negative_record.publication_year or "",
                    "source": "pubmed_matched_negative",
                    "matched_positive_pmid": pmid,
                    "group_id": "",
                    "split": "",
                    "rw_record_id": "",
                    "rw_retraction_nature": "",
                    "rw_retraction_reason": "",
                    "rw_retraction_date": "",
                    "rw_retraction_doi": "",
                    "rw_retraction_pmid": "",
                }
            )
    return rows


def parse_optional_year(value: str | int | None) -> int | None:
    if isinstance(value, int):
        return value
    text = clean_text(str(value or ""))
    return int(text) if text.isdigit() else None


def assign_splits(rows: list[dict[str, str | int]]) -> dict[str, int]:
    grouped_rows: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    group_years: dict[str, int | None] = {}

    for row in rows:
        label = int(row["label"])
        group_id = str(row["pmid"] if label == 1 else row["matched_positive_pmid"])
        row["group_id"] = group_id
        grouped_rows[group_id].append(row)
        if group_id not in group_years or group_years[group_id] is None:
            group_years[group_id] = parse_optional_year(row.get("publication_year"))
        if label == 1:
            group_years[group_id] = parse_optional_year(row.get("publication_year"))

    ordered_groups = sorted(
        grouped_rows,
        key=lambda group_id: (
            group_years[group_id] is None,
            group_years[group_id] if group_years[group_id] is not None else 0,
            group_id,
        ),
    )

    total_groups = len(ordered_groups)
    if total_groups == 0:
        return {split: 0 for split in SPLIT_NAMES}

    test_groups = 1 if total_groups > 1 else 0
    validation_groups = max(1, total_groups // 10) if total_groups >= 10 else 0
    train_groups = max(0, total_groups - validation_groups - test_groups)

    split_by_group: dict[str, str] = {}
    for index, group_id in enumerate(ordered_groups):
        if index < train_groups:
            split = "train"
        elif index < train_groups + validation_groups:
            split = "validation"
        else:
            split = "test"
        split_by_group[group_id] = split

    split_counts = {split: 0 for split in SPLIT_NAMES}
    for row in rows:
        split = split_by_group[str(row["group_id"])]
        row["split"] = split
        split_counts[split] += 1
    return split_counts


def write_jsonl(path: Path, rows: list[dict[str, str | int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def render_dataset_card(summary: dict[str, object]) -> str:
    split_counts = summary.get("split_counts", {})
    split_lines = "\n".join(
        f"- `{split}`: {split_counts.get(split, 0)} rows"
        for split in SPLIT_NAMES
    )
    return f"""---
license: other
task_categories:
- text-classification
pretty_name: rlpaperdetector
---

# rlpaperdetector

Derived dataset for retraction prediction experiments built from Crossref Retraction Watch records and PubMed metadata.

## Contents

{split_lines}

## Schema

- `label`: `1` for original papers that were later retracted, `0` for matched PubMed negatives
- `pmid`, `doi`, `title`, `abstract`, `journal`, `publication_year`
- `source`: `retraction_watch_pubmed` or `pubmed_matched_negative`
- `matched_positive_pmid`: original positive PMID for matched negatives
- `group_id`: grouping key used to keep positives and their matched negatives in the same split
- `split`: deterministic temporal split assignment

## Caveats

- This is a derived dataset. Do not assume the raw Retraction Watch CSV is redistributable just because the derived artifacts are published.
- Label leakage is a real risk. Avoid using post-retraction metadata fields at training time.
- Negatives are matched heuristically by journal and publication year, so this is not a causal benchmark.

## Build Summary

- Positives requested: {summary.get("positives_requested", 0)}
- Positives written: {summary.get("positives_written", 0)}
- Negatives written: {summary.get("negatives_written", 0)}
- Rows written: {summary.get("rows_written", 0)}
"""


def write_hf_layout(
    rows: list[dict[str, str | int]],
    output_dir: Path,
    summary: dict[str, object],
    write_parquet: bool,
) -> None:
    hf_dir = output_dir / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in SPLIT_NAMES
    }
    for split, split_rows in rows_by_split.items():
        write_jsonl(hf_dir / f"{split}.jsonl", split_rows)

    if write_parquet:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("`--write-parquet` requires `pyarrow`. Install it with `pip install pyarrow`.") from exc

        for split, split_rows in rows_by_split.items():
            table = pa.Table.from_pylist(split_rows)
            pq.write_table(table, hf_dir / f"{split}.parquet")

    (hf_dir / "README.md").write_text(render_dataset_card(summary), encoding="utf-8")


def write_outputs(
    rows: list[dict[str, str | int]],
    output_dir: Path,
    summary: dict[str, object],
    write_parquet: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "pmid",
        "doi",
        "title",
        "abstract",
        "journal",
        "publication_year",
        "source",
        "matched_positive_pmid",
        "group_id",
        "split",
        "rw_record_id",
        "rw_retraction_nature",
        "rw_retraction_reason",
        "rw_retraction_date",
        "rw_retraction_doi",
        "rw_retraction_pmid",
    ]

    csv_path = output_dir / "dataset.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    jsonl_path = output_dir / "dataset.jsonl"
    write_jsonl(jsonl_path, rows)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_hf_layout(rows, output_dir, summary, write_parquet=write_parquet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a retraction prediction dataset.")
    parser.add_argument("--crossref-email", default=DEFAULT_CROSSREF_EMAIL, help="Email passed to Crossref and NCBI.")
    parser.add_argument("--rw-csv", type=Path, help="Existing Retraction Watch CSV. If omitted, download it.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory used for downloaded source files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where dataset files will be written.",
    )
    parser.add_argument("--max-positives", type=int, help="Limit the number of positive PMIDs for smoke tests.")
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--negative-candidate-pool-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--write-parquet", action="store_true", help="Write split Parquet files in the Hugging Face layout.")
    parser.add_argument("--force-download", action="store_true", help="Re-download the Retraction Watch CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.rw_csv
    if csv_path is None:
        csv_path = args.cache_dir / "retraction_watch.csv"
        if csv_path.exists() and not args.force_download:
            log(f"Using cached Retraction Watch CSV at {csv_path}")
        else:
            log(f"Downloading Retraction Watch CSV to {csv_path}")
        download_retraction_watch_csv(args.crossref_email, csv_path, force=args.force_download)

    log(f"Loading positives from {csv_path}")
    seeds = load_positive_seed_rows(csv_path)
    positive_pmids = list(seeds)
    if args.max_positives is not None:
        positive_pmids = positive_pmids[: args.max_positives]
        seeds = {pmid: seeds[pmid] for pmid in positive_pmids}

    log(f"Using {len(positive_pmids)} positive PMIDs from Retraction Watch")
    client = PubMedClient(email=args.crossref_email)
    positive_records = fetch_pubmed_records(client, positive_pmids)
    positive_records = {pmid: record for pmid, record in positive_records.items() if record.has_usable_text}
    dropped_pmids = sorted(set(positive_pmids) - set(positive_records))
    if dropped_pmids:
        log(f"Dropped {len(dropped_pmids)} positives with missing text or unusable PubMed metadata")
        seeds = {pmid: seed for pmid, seed in seeds.items() if pmid in positive_records}

    negative_matches = sample_negative_records(
        client=client,
        positives=positive_records,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        candidate_pool_size=args.negative_candidate_pool_size,
    )

    rows = build_dataset_rows(seeds, positive_records, negative_matches)
    split_counts = assign_splits(rows)
    total_negatives = sum(1 for row in rows if row["label"] == 0)
    summary = {
        "crossref_csv": str(csv_path),
        "positives_requested": len(positive_pmids),
        "positives_written": len(positive_records),
        "negatives_written": total_negatives,
        "rows_written": len(rows),
        "negatives_per_positive_target": args.negatives_per_positive,
        "split_counts": split_counts,
    }
    write_outputs(rows, args.output_dir, summary, write_parquet=args.write_parquet)
    log(f"Wrote dataset with {len(rows)} rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
