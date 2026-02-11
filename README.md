# Phase 1 Dataset + Generation Pipeline (Master’s Thesis)

This repository contains the Phase 1 dataset and the scripts used to construct it for the Master’s thesis:

**Evaluating AI-Based Defences Against AI-Generated Social Engineering: A Technical and Human-Centered Analysis**. :contentReference[oaicite:1]{index=1}

Phase 1 focuses on technical robustness evaluation of phishing detection approaches against LLM-generated phishing messages and adversarial paraphrases.

---

## Contents

- `scripts/`
  - `pipeline_phase1.py` — end-to-end dataset construction pipeline (generation, paraphrasing, finalize/export)
  - (optional helper scripts you added during development)
- `prompts/`
  - system prompt and scenario/paraphrase prompt templates
- `raw/`
  - intermediate JSONL files produced by the pipeline (originals, paraphrases, legitimate)
- `final/`
  - **final dataset outputs**
    - `phase1_dataset.jsonl`
    - `phase1_dataset.csv`
    - `dataset_manifest.md` (counts, configuration, metadata)

> If you only need the dataset: use `final/phase1_dataset.jsonl` (source of truth).

---

## Dataset Overview

The final dataset is designed to be balanced and controlled for experimental benchmarking:

- **Phishing originals:** 120 (4 scenarios × 30 each)
- **Phishing paraphrases:** 600 (5 variants per original)
- **Legitimate emails:** 120
- **Total:** 840 rows

### Phishing scenarios
- `credential_theft`
- `impersonation`
- `invoice_fraud`
- `account_recovery`

### Paraphrase variants
Paraphrases are generated via multiple prompt strategies (e.g., tone shift, structural rewrite) while preserving the underlying scenario intent.

---

## Data Format (JSONL)

Each line in `final/phase1_dataset.jsonl` is one JSON record.

Common fields (typical):
- `msg_id` (string): unique identifier (e.g., `phish_invoice_fraud_023_p1`)
- `label` (string): `phish` or `legit`
- `scenario` (string or null): phishing scenario name; null/absent for legit
- `variant_type` (string): `original`, `paraphrase`, or `legit`
- `paraphrase_strategy` (string): e.g. `na` for originals; strategy name for paraphrases
- `base_msg_id` (string, optional): present for paraphrases; points to the corresponding original
- `text` (string): email content (subject + body)
- `length_words` (int)
- `text_hash` (string): SHA-256 of normalized text
- `generation_model`, `temperature`, `top_p`, `max_output_tokens` (metadata)
- `prompt_id` (string): prompt family + version used (e.g., `credential_theft_base_v2`)



---

## Reproducibility Notes

### Determinism
LLM generation is inherently stochastic. The pipeline records:
- model name
- sampling parameters (temperature/top_p)
- prompt identifiers

This supports auditability even when exact regeneration is not guaranteed.

### Completeness & Quality Gates
The pipeline enforces basic validity constraints such as:
- word-count bounds
- empty-output rejection
- duplicate-hash rejection
- retry/backoff logic

---

## Setup

### Requirements
- Python 3.14 recommended
- Google GenAI SDK (Gemini) + standard utilities

Install dependencies:
```bash
pip install google-genai pandas

