# rlpaperdetector

Build a first-pass dataset for retraction prediction experiments from:

- Retraction Watch data served by Crossref
- PubMed abstracts and metadata fetched via NCBI E-utilities

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

## GitHub Actions

This repo now includes:

- `.github/workflows/ci.yml` for deterministic local tests
- `.github/workflows/publish-huggingface.yml` for manual dataset publication to Hugging Face Hub

The publish workflow expects a GitHub Actions secret named `HF_TOKEN`. Trigger it manually and provide:

- `crossref_email`
- `hf_repo_id`
- `max_positives`
- `negatives_per_positive`
