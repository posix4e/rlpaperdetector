# rlpaperdetector

Build a first-pass dataset for retraction prediction experiments from:

- Retraction Watch data served by Crossref
- PubMed abstracts and metadata fetched via NCBI E-utilities
- A pure-Python baseline retraction predictor
- A subject monitor for newly arriving PubMed papers

## Quick start

```bash
python3 build_dataset.py \
  --crossref-email you@example.org \
  --max-positives 100 \
  --negatives-per-positive 2 \
  --write-parquet
```

Use a real email address. Crossref includes it in the Retraction Watch download URL, and NCBI recommends it on E-utilities requests.

Outputs land in `data/processed/`:

- `dataset.csv`
- `dataset.jsonl`
- `summary.json`
- `hf/train.jsonl`
- `hf/validation.jsonl`
- `hf/test.jsonl`
- `hf/README.md`
- `hf/*.parquet` when `--write-parquet` is enabled

The dataset contains positives from original retracted papers and matched negatives sampled from PubMed by journal and publication year.

By default, dataset building and baseline training also apply [excluded_papers.json](/home/ubuntu/src/rlpaperdetector/configs/excluded_papers.json#L1) so specific PMIDs/DOIs/titles can be kept out of all generated datasets and training runs.

## Baseline Model

Train a dependency-free baseline model:

```bash
python3 train_baseline.py \
  --dataset data/processed/dataset.csv \
  --output-dir artifacts/baseline
```

Evaluate it:

```bash
python3 eval.py \
  --model artifacts/baseline/model.json \
  --dataset data/processed/dataset.csv \
  --split test
```

Score one paper:

```bash
python3 predict.py \
  --model artifacts/baseline/model.json \
  --title "Paper title" \
  --abstract "Paper abstract" \
  --journal "Journal name" \
  --publication-year 2025
```

Or score a batch file:

```bash
python3 predict.py \
  --model artifacts/baseline/model.json \
  --input-file data/processed/dataset.csv \
  --output predictions.jsonl
```

## Monitor New Papers

Watch subject-specific PubMed queries over an exact date window, score unseen papers, and write a ranked review queue:

```bash
python3 monitor_subjects.py \
  --model artifacts/baseline/model.json \
  --query-file configs/example_subject_queries.txt \
  --days-back 1 \
  --date-field edat \
  --output-dir reports/monitor
```

This writes:

- `reports/monitor/scored_papers.csv`
- `reports/monitor/scored_papers.jsonl`
- `reports/monitor/summary.json`

And updates a dedupe state file at `state/seen_pmids.json` so the same PMID is not rescored every run.

## GitHub Actions

This repo now includes:

- `.github/workflows/ci.yml` for deterministic local tests
- `.github/workflows/publish-huggingface.yml` for manual dataset publication to Hugging Face Hub

The publish workflow expects a GitHub Actions secret named `HF_TOKEN`. Trigger it manually and provide:

- `crossref_email`
- `hf_repo_id`
- `max_positives`
- `negatives_per_positive`
