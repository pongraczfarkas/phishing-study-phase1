# Phase 1 Dataset Manifest

- Date generated: 2026-02-04

- Total messages (deduped): **840**

## Counts

- By label: {'phish': 720, 'legit': 120}

- Phishing by scenario: {'credential_theft': 180, 'impersonation': 180, 'invoice_fraud': 180, 'account_recovery': 180}

- Phishing by variant_type: {'paraphrase': 600, 'original': 120}

- Paraphrases by strategy: {'professional_polish': 120, 'friendly_tone': 120, 'reduced_urgency': 120, 'structural_rewrite': 120, 'remove_suspicious_keywords': 120}


## Length statistics (words)

- All: {'min': 145, 'p25': 166, 'median': 169, 'p75': 173, 'max': 187, 'mean': 169.32738095238096}

- Phish: {'min': 145, 'p25': 166, 'median': 170, 'p75': 173, 'max': 187, 'mean': 169.4597222222222}

- Legit: {'min': 157, 'p25': 166, 'median': 168, 'p75': 171, 'max': 177, 'mean': 168.53333333333333}


## Generation settings

- Model: gemini-3-flash-preview

- temperature: 0.85

- top_p: 0.95

- max_output_tokens: 900


## Quality gates

- Word bounds: 120–200

- Deduplication: SHA-256 of normalized text

- Rejections logged in logs/rejected.jsonl


## Files

- Raw: raw/phishing_original.jsonl, raw/phishing_paraphrased.jsonl, raw/legitimate.jsonl

- Final: final/phase1_dataset.jsonl, final/phase1_dataset.csv

- Logs: logs\run_log.jsonl, logs\rejected.jsonl
