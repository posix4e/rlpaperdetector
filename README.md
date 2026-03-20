# rlpaperdetector

Build a dataset and Axolotl-ready preference corpora for retraction-judge experiments from:

- Retraction Watch data served by Crossref
- PubMed abstracts and metadata fetched via NCBI E-utilities
- Axolotl-ready preference-data generation for DPO / ORPO post-training

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

By default, dataset building and preference-data generation apply [excluded_papers.json](/home/ubuntu/src/rlpaperdetector/configs/excluded_papers.json#L1) so specific PMIDs/DOIs/titles can be kept out of generated labeled datasets.

## Preference Data For Axolotl

If you already have candidate judge responses for the same prompt, you can score them against a strict rubric, generate grader rationales, convert them into preference pairs, and export Axolotl-ready files:

```bash
python3 build_preference_data.py \
  --input candidate_responses.jsonl \
  --output-dir preferences \
  --rubric configs/retraction_rubric.json
```

Expected input rows are JSONL with:

- `prompt_id`
- `messages` or `prompt`
- `response`
- `gold_decision` or `label`
- optional `paper_title` / `paper_abstract`

The pipeline writes:

- `preferences/scored/*.jsonl` with grader CoT and 1-5 rubric scores
- `preferences/axolotl/sft_*.jsonl` for warm-start SFT on top responses
- `preferences/axolotl/dpo_*` in `chat_template.default` format
- `preferences/axolotl/orpo_*` in `chat_template.argilla` format

The pairwise exports also include `score_gap`, `pair_weight`, and `pair_repeats`. Stock Axolotl docs expose DPO / ORPO dataset formats and weighting knobs, but they do not document a per-example custom margin column, so this repo injects score strength by oversampling pairs according to score gap and preserving the raw gap metadata for later custom trainer work.

Axolotl configs are included at:

- `configs/axolotl/retraction_margin_dpo.yaml`
- `configs/axolotl/retraction_orpo.yaml`

The intended completion style is: rubric-aligned rationale first, then a single terminal decision token such as `<RETRACT>` or `<KEEP>`.

## Why Use A Custom-Trained Model

This repo uses a custom-trained judge instead of relying on a general-purpose model such as Grok because the goal is consistency and control, not open-ended conversation.

- The judge is trained against a fixed rubric, fixed label space, and fixed output format.
- The model can be evaluated reproducibly on the same distribution it was trained for.
- Preference data, failure cases, and checkpoints are all versioned and inspectable.
- The output format is constrained to a rationale followed by a terminal decision token such as `<RETRACT>` or `<KEEP>`.
- A general-purpose model may still be stronger in raw reasoning, but it is not tuned to this exact task and its behavior can drift over time.

## GitHub Actions Automation

This repo now includes a manual GitHub Actions workflow to do the whole data-prep path without local commands:

- `.github/workflows/build-preferences-huggingface.yml`

That workflow will:

1. build a fresh labeled dataset from Retraction Watch and PubMed
2. generate the preference corpus and Axolotl exports
3. build the held-out probe set
4. upload artifacts to GitHub Actions
5. publish the preference package to a Hugging Face dataset repo
6. optionally submit a Hugging Face GPU training job for Axolotl

It expects the `HF_TOKEN` repository secret.

There is also a second workflow for standalone training submission:

- `.github/workflows/train-axolotl-hf-jobs.yml`

That workflow submits an Axolotl training run to Hugging Face Jobs using the official Axolotl cloud image. No self-hosted runner is required. You still need a Hugging Face account with access to Jobs and GPU hardware.

Recommended flow:

1. run `.github/workflows/build-preferences-huggingface.yml`
2. set `launch_training_job=true`
3. provide both `hf_repo_id` and `hf_model_repo_id`
4. let GitHub Actions build the dataset, publish the preference package, and enqueue the remote training job on Hugging Face

To build a first synthetic preference corpus directly from Retraction Watch labels:

```bash
python3 build_rw_preferences.py \
  --dataset data/processed/dataset.csv \
  --output-dir preferences \
  --max-papers 2000
```

This samples up to `2000` labeled paper rows, creates `4` candidate judge responses per paper, grades them with the rubric, and emits Axolotl-ready SFT/DPO/ORPO files.

## Held-Out Probes

Some papers are useful as probes but should not be treated as ground-truth retraction labels. Build a held-out probe set for targeted scoring after training:

```bash
python3 build_probe_set.py \
  --pmid-file configs/probe_pmids.txt \
  --output preferences/probes/probe_papers.jsonl
```

The default probe file includes the black-babies original paper (`PMID 32817561`) and its re-analysis. These probes are not used as labeled preference-training examples unless you explicitly do so yourself.

## GitHub Actions

This repo now includes:

- `.github/workflows/ci.yml` for deterministic local tests
- `.github/workflows/publish-huggingface.yml` for manual dataset publication to Hugging Face Hub
- `.github/workflows/build-preferences-huggingface.yml` for dataset -> preference package -> optional HF Jobs training submission
- `.github/workflows/train-axolotl-hf-jobs.yml` for direct Hugging Face Jobs training submission

The publish workflow expects a GitHub Actions secret named `HF_TOKEN`. Trigger it manually and provide:

- `crossref_email`
- `hf_repo_id`
- `max_positives`
- `negatives_per_positive`
