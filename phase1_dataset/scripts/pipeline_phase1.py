#!/usr/bin/env python3
"""
Phase 1 dataset pipeline using Google Gen AI SDK (python-genai / google-genai).

Docs alignment:
- Client: from google import genai; client = genai.Client(api_key="...")  (or env var GEMINI_API_KEY)
- Generate: client.models.generate_content(model=..., contents=..., config=types.GenerateContentConfig(...))
- System instruction: config=types.GenerateContentConfig(system_instruction="...", ...)
- Errors: from google.genai import errors; except errors.APIError

References:
- https://googleapis.github.io/python-genai/  (client + generate_content + config)
"""

import os
import re
import json
import time
import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from google import genai
from google.genai import types, errors


# ----------------------------
# USER-TUNABLE SETTINGS
# ----------------------------

MODEL_NAME = "gemini-3-flash-preview"  # change if you want (e.g., gemini-2.0-flash-001)
TEMPERATURE = 0.85
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 900

SLEEP_BETWEEN_CALLS_SEC = 2.5
MAX_RETRIES = 8
BACKOFF_BASE_SEC = 3  # exponential backoff base for API errors

# Phase-1 scope gates
LENGTH_MIN_WORDS = 120
LENGTH_MAX_WORDS = 200
PARAPHRASES_PER_EMAIL = 5

PHISHING_TARGETS = {
    "credential_theft": 30,
    "impersonation": 30,
    "invoice_fraud": 30,
    "account_recovery": 30,
}
LEGIT_TARGET_TOTAL = 120

# Prompt files
PROMPT_FILES = {
    "system": "prompts/system_master_v2.txt",

    # Phishing originals
    "credential_theft": "prompts/credential_theft_base_v3.txt",
    "impersonation": "prompts/impersonation_base_v3.txt",
    "invoice_fraud": "prompts/invoice_fraud_base_v3.txt",
    "account_recovery": "prompts/account_recovery_base_v3.txt",

    # Legit
    "legit": "prompts/legit_base_v2.txt",

    # Paraphrases (must contain marker <<<PASTE ORIGINAL EMAIL HERE>>>)
    "p1": "prompts/paraphrase_p1_contextual_rephrasing_v2.txt",
    "p2": "prompts/paraphrase_p2_embedded_action_v2.txt",
    "p3": "prompts/paraphrase_p3_vocabulary_redistribution_v2.txt",
    "p4": "prompts/paraphrase_p4_reduced_urgency_v2.txt",
    "p5": "prompts/paraphrase_p5_operational_shift_v2.txt",
}

PARAPHRASE_STRATEGY_NAMES = {
    "p1": "professional_polish",
    "p2": "friendly_tone",
    "p3": "reduced_urgency",
    "p4": "structural_rewrite",
    "p5": "remove_suspicious_keywords",
}

# Output paths
RAW_ORIG_PATH = "raw/phishing_original.jsonl"
RAW_PARA_PATH = "raw/phishing_paraphrased.jsonl"
RAW_LEGIT_PATH = "raw/legitimate.jsonl"

FINAL_JSONL_PATH = "final/phase1_dataset.jsonl"
FINAL_CSV_PATH = "final/phase1_dataset.csv"
MANIFEST_PATH = "final/dataset_manifest.md"

LOGS_DIR = "logs"
RUN_LOG_PATH = os.path.join(LOGS_DIR, "run_log.jsonl")
REJECTED_LOG_PATH = os.path.join(LOGS_DIR, "rejected.jsonl")

# Optional subtle variation hints to prevent “samey” emails
# VARIATION_HINTS = [
#     "Reference an upcoming system update window.",
#     "Mention routine compliance with internal policy.",
#     "Mention maintaining uninterrupted access during maintenance.",
#     "Reference standard quarterly security procedures.",
#     "Mention a recent internal access review.",
#     "Mention scheduled maintenance of authentication services.",
#     "Use a calm, routine administrative tone.",
# ]


# ----------------------------
# FILE + TEXT HELPERS
# ----------------------------

def ensure_dirs() -> None:
    os.makedirs("prompts", exist_ok=True)
    os.makedirs("raw", exist_ok=True)
    os.makedirs("cleaned", exist_ok=True)
    os.makedirs("final", exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

def now_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def append_jsonl(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def validate_prompt_files() -> None:
    missing = [p for p in PROMPT_FILES.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing prompt files:\n" + "\n".join(missing) +
            "\n\nFix filenames or update PROMPT_FILES in the script."
        )

def log_run(event: dict) -> None:
    append_jsonl(RUN_LOG_PATH, event)

def log_rejected(event: dict) -> None:
    append_jsonl(REJECTED_LOG_PATH, event)


# ----------------------------
# QUALITY GATES
# ----------------------------

def passes_quality(text: str) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty"
    t = normalize_text(text)
    wc = word_count(t)
    if wc < LENGTH_MIN_WORDS:
        return False, f"too_short({wc})"
    if wc > LENGTH_MAX_WORDS:
        return False, f"too_long({wc})"
    low = t.lower()
    # reject “meta” outputs that sometimes slip in
    if "as an ai" in low or "i can't" in low and "email" in low:
        return False, "meta_or_refusal"
    if "this is a simulation" in low or "for research" in low:
        return False, "meta_simulation"
    return True, "ok"


# ----------------------------
# GENAI CLIENT WRAPPER
# ----------------------------

@dataclass
class GenAIEmailGenerator:
    client: genai.Client
    system_instruction: str

    def generate(self, user_prompt: str) -> tuple[str, str]:
        cfg = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=cfg,
        )
        text = (resp.text or "").strip()
        finish = ""
        if getattr(resp, "candidates", None):
            finish = str(getattr(resp.candidates[0], "finish_reason", "") or "")
        return text, finish

def make_client() -> GenAIEmailGenerator:
    validate_prompt_files()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")

    client = genai.Client(api_key=api_key)
    system_instruction = load_text(PROMPT_FILES["system"])
    return GenAIEmailGenerator(client=client, system_instruction=system_instruction)


# ----------------------------
# ID SCHEMES
# ----------------------------

def phish_id(scenario: str, idx: int, variant: str) -> str:
    return f"phish_{scenario}_{idx:03d}_{variant}"

def legit_id(idx: int) -> str:
    return f"legit_{idx:03d}"

def existing_hashes() -> set:
    seen = set()
    for path in (RAW_ORIG_PATH, RAW_PARA_PATH, RAW_LEGIT_PATH):
        for r in read_jsonl(path):
            h = r.get("text_hash")
            if h:
                seen.add(h)
            else:
                txt = r.get("text", "")
                if txt:
                    seen.add(sha256_text(normalize_text(txt)))
    return seen


# ----------------------------
# PIPELINE STAGES
# ----------------------------

def generate_phishing_originals(gen: GenAIEmailGenerator) -> None:
    ensure_dirs()
    seen = existing_hashes()
    existing = read_jsonl(RAW_ORIG_PATH)

    # count by scenario
    counts = {s: 0 for s in PHISHING_TARGETS}
    for r in existing:
        if r.get("variant_type") == "original" and r.get("scenario") in counts:
            counts[r["scenario"]] += 1

    for scenario, target_n in PHISHING_TARGETS.items():
        base_prompt = load_text(PROMPT_FILES[scenario])
        start = counts[scenario] + 1
        if start > target_n:
            print(f"[phish originals] {scenario}: complete ({counts[scenario]}/{target_n})")
            continue

        print(f"[phish originals] {scenario}: generating {start}..{target_n}")

        for i in range(start, target_n + 1):
            msgid = phish_id(scenario, i, "orig")
            # hint = VARIATION_HINTS[(i - 1) % len(VARIATION_HINTS)]

            user_prompt = (
                f"{base_prompt}\n\n"
                # f"Variation hint (apply subtly): {hint}\n"
            )

            last_reason = ""
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    text_raw, finish = gen.generate(user_prompt)
                    text = normalize_text(text_raw)
                    if finish.endswith("MAX_TOKENS"):
                        ok, reason = False, "truncated_max_tokens"
                    else:
                        ok, reason = passes_quality(text)

                    if not ok:
                        last_reason = reason
                        log_rejected({
                            "stage": "phish_original",
                            "scenario": scenario,
                            "msg_id": msgid,
                            "attempt": attempt,
                            "reason": reason,
                            "preview": text[:200],
                            "date": now_date(),
                        })
                        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                        continue

                    h = sha256_text(text)
                    if h in seen:
                        last_reason = "duplicate_hash"
                        log_rejected({
                            "stage": "phish_original",
                            "scenario": scenario,
                            "msg_id": msgid,
                            "attempt": attempt,
                            "reason": "duplicate_hash",
                            "date": now_date(),
                        })
                        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                        continue

                    rec = {
                        "msg_id": msgid,
                        "label": "phish",
                        "scenario": scenario,
                        "variant_type": "original",
                        "paraphrase_strategy": "na",
                        "source": "generated_llm",
                        "generation_model": MODEL_NAME,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "max_output_tokens": MAX_OUTPUT_TOKENS,
                        "prompt_id": f"{scenario}_base_v1",
                        "text": text,
                        "length_words": word_count(text),
                        "text_hash": h,
                        "date_generated": now_date(),
                    }
                    append_jsonl(RAW_ORIG_PATH, rec)
                    seen.add(h)

                    log_run({
                        "stage": "phish_original",
                        "scenario": scenario,
                        "msg_id": msgid,
                        "attempt": attempt,
                        "status": "ok",
                        "date": now_date(),
                    })

                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                    break

                except errors.APIError as e:
                    wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                    last_reason = f"api_error({getattr(e, 'code', 'na')})"
                    log_rejected({
                        "stage": "phish_original",
                        "scenario": scenario,
                        "msg_id": msgid,
                        "attempt": attempt,
                        "reason": last_reason,
                        "detail": str(e)[:300],
                        "date": now_date(),
                    })
                    time.sleep(wait)

                except Exception as e:
                    wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                    last_reason = f"exception({type(e).__name__})"
                    log_rejected({
                        "stage": "phish_original",
                        "scenario": scenario,
                        "msg_id": msgid,
                        "attempt": attempt,
                        "reason": last_reason,
                        "detail": str(e)[:300],
                        "date": now_date(),
                    })
                    time.sleep(wait)

            else:
                print(f"  [WARN] failed {msgid} after {MAX_RETRIES} attempts (last={last_reason})")


def generate_paraphrases(gen: GenAIEmailGenerator) -> None:
    ensure_dirs()
    seen = existing_hashes()

    originals = read_jsonl(RAW_ORIG_PATH)
    if not originals:
        print("[paraphrases] No originals found. Run --generate-phish first.")
        return

    existing_para = read_jsonl(RAW_PARA_PATH)
    existing_ids = {r.get("msg_id") for r in existing_para}

    para_templates = {k: load_text(PROMPT_FILES[k]) for k in ["p1", "p2", "p3", "p4", "p5"]}

    total_target = len(originals) * PARAPHRASES_PER_EMAIL
    print(f"[paraphrases] originals={len(originals)} target={total_target}")

    for orig in originals:
        scenario = orig["scenario"]
        # msg_id format: phish_<scenario>_<NNN>_orig
        base_idx = int(orig["msg_id"].rsplit("_", 2)[1])

        for pk in ["p1", "p2", "p3", "p4", "p5"]:
            msgid = phish_id(scenario, base_idx, pk)
            if msgid in existing_ids:
                continue

            template = para_templates[pk]
            strategy = PARAPHRASE_STRATEGY_NAMES[pk]
            user_prompt = template.replace("<<<PASTE ORIGINAL EMAIL HERE>>>", orig["text"])

            last_reason = ""
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    text_raw, finish = gen.generate(user_prompt)
                    text = normalize_text(text_raw)
                    if finish.endswith("MAX_TOKENS"):
                        ok, reason = False, "truncated_max_tokens"
                    else:
                        ok, reason = passes_quality(text)

                    if not ok:
                        last_reason = reason
                        log_rejected({
                            "stage": "paraphrase",
                            "scenario": scenario,
                            "base_msg_id": orig["msg_id"],
                            "msg_id": msgid,
                            "strategy": strategy,
                            "attempt": attempt,
                            "reason": reason,
                            "preview": text[:200],
                            "date": now_date(),
                        })
                        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                        continue

                    h = sha256_text(text)
                    if h in seen:
                        last_reason = "duplicate_hash"
                        log_rejected({
                            "stage": "paraphrase",
                            "scenario": scenario,
                            "base_msg_id": orig["msg_id"],
                            "msg_id": msgid,
                            "strategy": strategy,
                            "attempt": attempt,
                            "reason": "duplicate_hash",
                            "date": now_date(),
                        })
                        time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                        continue

                    rec = {
                        "msg_id": msgid,
                        "label": "phish",
                        "scenario": scenario,
                        "variant_type": "paraphrase",
                        "paraphrase_strategy": strategy,
                        "base_msg_id": orig["msg_id"],
                        "source": "generated_llm",
                        "generation_model": MODEL_NAME,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "max_output_tokens": MAX_OUTPUT_TOKENS,
                        "prompt_id": f"{scenario}_{pk}_v1",
                        "text": text,
                        "length_words": word_count(text),
                        "text_hash": h,
                        "date_generated": now_date(),
                    }
                    append_jsonl(RAW_PARA_PATH, rec)
                    existing_ids.add(msgid)
                    seen.add(h)

                    log_run({
                        "stage": "paraphrase",
                        "scenario": scenario,
                        "base_msg_id": orig["msg_id"],
                        "msg_id": msgid,
                        "strategy": strategy,
                        "attempt": attempt,
                        "status": "ok",
                        "date": now_date(),
                    })

                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                    break

                except errors.APIError as e:
                    wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                    last_reason = f"api_error({getattr(e, 'code', 'na')})"
                    log_rejected({
                        "stage": "paraphrase",
                        "scenario": scenario,
                        "base_msg_id": orig["msg_id"],
                        "msg_id": msgid,
                        "strategy": strategy,
                        "attempt": attempt,
                        "reason": last_reason,
                        "detail": str(e)[:300],
                        "date": now_date(),
                    })
                    time.sleep(wait)

                except Exception as e:
                    wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                    last_reason = f"exception({type(e).__name__})"
                    log_rejected({
                        "stage": "paraphrase",
                        "scenario": scenario,
                        "base_msg_id": orig["msg_id"],
                        "msg_id": msgid,
                        "strategy": strategy,
                        "attempt": attempt,
                        "reason": last_reason,
                        "detail": str(e)[:300],
                        "date": now_date(),
                    })
                    time.sleep(wait)

            else:
                print(f"  [WARN] failed paraphrase {msgid} after {MAX_RETRIES} attempts (last={last_reason})")


def generate_legitimate(gen: GenAIEmailGenerator) -> None:
    ensure_dirs()
    seen = existing_hashes()

    existing = read_jsonl(RAW_LEGIT_PATH)
    start = len(existing) + 1
    if start > LEGIT_TARGET_TOTAL:
        print(f"[legit] complete ({len(existing)}/{LEGIT_TARGET_TOTAL})")
        return

    base_prompt = load_text(PROMPT_FILES["legit"])
    print(f"[legit] generating {start}..{LEGIT_TARGET_TOTAL}")

    for i in range(start, LEGIT_TARGET_TOTAL + 1):
        msgid = legit_id(i)
        # hint = VARIATION_HINTS[(i - 1) % len(VARIATION_HINTS)]
        user_prompt = base_prompt

        last_reason = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                text_raw, finish = gen.generate(user_prompt)
                text = normalize_text(text_raw)
                if finish.endswith("MAX_TOKENS"):
                    ok, reason = False, "truncated_max_tokens"
                else:
                    ok, reason = passes_quality(text)

                if not ok:
                    last_reason = reason
                    log_rejected({
                        "stage": "legit",
                        "msg_id": msgid,
                        "attempt": attempt,
                        "reason": reason,
                        "preview": text[:200],
                        "date": now_date(),
                    })
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                    continue

                h = sha256_text(text)
                if h in seen:
                    last_reason = "duplicate_hash"
                    log_rejected({
                        "stage": "legit",
                        "msg_id": msgid,
                        "attempt": attempt,
                        "reason": "duplicate_hash",
                        "date": now_date(),
                    })
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                    continue

                rec = {
                    "msg_id": msgid,
                    "label": "legit",
                    "scenario": "legit_na",
                    "variant_type": "original",
                    "paraphrase_strategy": "na",
                    "source": "synthetic_legit",
                    "generation_model": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "prompt_id": "legit_base_v1",
                    "text": text,
                    "length_words": word_count(text),
                    "text_hash": h,
                    "date_generated": now_date(),
                }
                append_jsonl(RAW_LEGIT_PATH, rec)
                seen.add(h)

                log_run({
                    "stage": "legit",
                    "msg_id": msgid,
                    "attempt": attempt,
                    "status": "ok",
                    "date": now_date(),
                })

                time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                break

            except errors.APIError as e:
                wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                last_reason = f"api_error({getattr(e, 'code', 'na')})"
                log_rejected({
                    "stage": "legit",
                    "msg_id": msgid,
                    "attempt": attempt,
                    "reason": last_reason,
                    "detail": str(e)[:300],
                    "date": now_date(),
                })
                time.sleep(wait)

            except Exception as e:
                wait = BACKOFF_BASE_SEC * (2 ** (attempt - 1))
                last_reason = f"exception({type(e).__name__})"
                log_rejected({
                    "stage": "legit",
                    "msg_id": msgid,
                    "attempt": attempt,
                    "reason": last_reason,
                    "detail": str(e)[:300],
                    "date": now_date(),
                })
                time.sleep(wait)

        else:
            print(f"  [WARN] failed {msgid} after {MAX_RETRIES} attempts (last={last_reason})")


def finalize_dataset() -> None:
    ensure_dirs()

    orig = read_jsonl(RAW_ORIG_PATH)
    para = read_jsonl(RAW_PARA_PATH)
    legit = read_jsonl(RAW_LEGIT_PATH)

    all_rows = orig + para + legit
    if not all_rows:
        print("[finalize] No rows found in raw/.")
        return

    # Deduplicate by hash, keep first occurrence
    seen = set()
    deduped = []
    for r in all_rows:
        txt = normalize_text(r.get("text", ""))
        h = r.get("text_hash") or (sha256_text(txt) if txt else "")
        if not h or h in seen:
            continue
        seen.add(h)
        r["text"] = txt
        r["text_hash"] = h
        r["length_words"] = r.get("length_words") or word_count(txt)
        deduped.append(r)

    # Write final JSONL
    with open(FINAL_JSONL_PATH, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # CSV
    df = pd.DataFrame(deduped)
    df.to_csv(FINAL_CSV_PATH, index=False)

    # Manifest
    manifest = build_manifest(df)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        f.write(manifest)

    print(f"[finalize] wrote {FINAL_JSONL_PATH} ({len(deduped)} rows)")
    print(f"[finalize] wrote {FINAL_CSV_PATH}")
    print(f"[finalize] wrote {MANIFEST_PATH}")


def build_manifest(df: pd.DataFrame) -> str:
    def stats(series: pd.Series) -> Dict[str, float]:
        return {
            "min": int(series.min()),
            "p25": int(series.quantile(0.25)),
            "median": int(series.median()),
            "p75": int(series.quantile(0.75)),
            "max": int(series.max()),
            "mean": float(series.mean()),
        }

    total = len(df)
    by_label = df["label"].value_counts().to_dict()

    ph = df[df["label"] == "phish"]
    ph_by_scenario = ph["scenario"].value_counts().to_dict() if len(ph) else {}
    ph_by_variant = ph["variant_type"].value_counts().to_dict() if len(ph) else {}

    para = ph[ph["variant_type"] == "paraphrase"] if len(ph) else pd.DataFrame()
    para_by_strategy = para["paraphrase_strategy"].value_counts().to_dict() if len(para) else {}

    lines = []
    lines.append("# Phase 1 Dataset Manifest\n")
    lines.append(f"- Date generated: {now_date()}\n")
    lines.append(f"- Total messages (deduped): **{total}**\n")
    lines.append("## Counts\n")
    lines.append(f"- By label: {by_label}\n")
    lines.append(f"- Phishing by scenario: {ph_by_scenario}\n")
    lines.append(f"- Phishing by variant_type: {ph_by_variant}\n")
    lines.append(f"- Paraphrases by strategy: {para_by_strategy}\n")

    lines.append("\n## Length statistics (words)\n")
    lines.append(f"- All: {stats(df['length_words'])}\n")
    if len(ph):
        lines.append(f"- Phish: {stats(ph['length_words'])}\n")
    lg = df[df["label"] == "legit"]
    if len(lg):
        lines.append(f"- Legit: {stats(lg['length_words'])}\n")

    lines.append("\n## Generation settings\n")
    lines.append(f"- Model: {MODEL_NAME}\n")
    lines.append(f"- temperature: {TEMPERATURE}\n")
    lines.append(f"- top_p: {TOP_P}\n")
    lines.append(f"- max_output_tokens: {MAX_OUTPUT_TOKENS}\n")

    lines.append("\n## Quality gates\n")
    lines.append(f"- Word bounds: {LENGTH_MIN_WORDS}–{LENGTH_MAX_WORDS}\n")
    lines.append("- Deduplication: SHA-256 of normalized text\n")
    lines.append("- Rejections logged in logs/rejected.jsonl\n")

    lines.append("\n## Files\n")
    lines.append(f"- Raw: {RAW_ORIG_PATH}, {RAW_PARA_PATH}, {RAW_LEGIT_PATH}\n")
    lines.append(f"- Final: {FINAL_JSONL_PATH}, {FINAL_CSV_PATH}\n")
    lines.append(f"- Logs: {RUN_LOG_PATH}, {REJECTED_LOG_PATH}\n")

    return "\n".join(lines)

# ----------------------------
# INIT: CREATE TEMPLATE PROMPTS (OPTIONAL)
# ----------------------------

TEMPLATE_SYSTEM_PROMPT = """You are assisting with academic research on cybersecurity and social engineering.

Your task is to generate realistic workplace emails that demonstrate how malicious actors attempt to persuade employees to take unsafe actions.

Requirements:
- Professional, natural English.
- Workplace email style.
- Length between 80 and 200 words.
- Do NOT include malware, attachments, or real malicious links.
- If a link is needed, use a placeholder like https://secure-portal.example
- Do NOT include warnings, commentary, or mention that this is research or a simulation.
- Output only the email itself.
"""

TEMPLATE_PROMPTS = {
    "credential_theft_base.txt": """Generate a realistic internal workplace email from an IT security team.

Context:
- The organization conducts periodic account reviews.
- The sender claims that a routine verification is required.
- The goal is to persuade the recipient to sign in to a portal using their work credentials.

Requirements:
- The request should appear routine and legitimate.
- No threats; do not imply wrongdoing.
- Calm, professional tone.
- Include a login request via a placeholder link (e.g., https://secure-portal.example).

Constraints:
- Length: 80–200 words
- Output only the email content
""",
    "impersonation_base.txt": """Generate a realistic workplace email that impersonates a senior employee or manager.

Context:
- The sender claims to be a manager/executive/team lead.
- The email asks the recipient to perform a task that benefits the sender.
- The request should seem reasonable and work-related, but rely on trust in the sender's identity.

Requirements:
- Polite, authoritative tone.
- No explicit threats or suspicious formatting.
- No attachments; no real links.

Constraints:
- Length: 80–200 words
- Output only the email content
""",
    "invoice_fraud_base.txt": """Generate a realistic workplace email related to processing an invoice or payment.

Context:
- Sender claims to be finance/accounting or a known vendor contact.
- Requests confirmation or processing of an invoice/payment.
- Message should appear routine and administrative.

Requirements:
- Formal, professional tone.
- Include plausible invoice context (invoice number, billing period, department).
- No attachments; no real links.

Constraints:
- Length: 80–200 words
- Output only the email content
""",
    "account_recovery_base.txt": """Generate a realistic workplace email related to account recovery or access verification.

Context:
- Sender claims to be support/account services.
- States there was an access or verification issue.
- Asks the recipient to follow steps to maintain or restore access.

Requirements:
- Calm, supportive tone.
- Do not imply the recipient did something wrong.
- If a link is needed, use a placeholder like https://secure-portal.example

Constraints:
- Length: 80–200 words
- Output only the email content
""",
    "legit_base.txt": """Write a legitimate internal workplace email.

Context:
- Routine and non-malicious.
- Topics may include project updates, scheduling, policy reminders, internal announcements.

Requirements:
- Do NOT request sensitive information.
- Do NOT request login credential verification.
- Professional and neutral tone.

Constraints:
- Length: 80–200 words
- Output only the email content
""",
    "paraphrase_p1_professional.txt": """Rewrite the email below to sound more polished and professionally written.

Instructions:
- Preserve intent and requested action.
- Improve clarity and corporate tone.
- Do not add new information or change meaning.
- Keep 80–200 words.
- Output only the rewritten email.

Original email:
<<<PASTE ORIGINAL EMAIL HERE>>>
""",
    "paraphrase_p2_friendly.txt": """Rewrite the email below using a friendlier, more conversational workplace tone.

Instructions:
- Preserve intent and requested action.
- Use polite, approachable language.
- Do not add new facts or change meaning.
- Keep 80–200 words.
- Output only the rewritten email.

Original email:
<<<PASTE ORIGINAL EMAIL HERE>>>
""",
    "paraphrase_p3_reduced_urgency.txt": """Rewrite the email below to reduce any sense of urgency.

Instructions:
- Preserve intent and requested action.
- Remove time pressure or alarmist wording.
- Make it sound routine and non-urgent.
- Keep 80–200 words.
- Output only the rewritten email.

Original email:
<<<PASTE ORIGINAL EMAIL HERE>>>
""",
    "paraphrase_p4_structural.txt": """Rewrite the email below using different sentence structure and ordering.

Instructions:
- Preserve intent and requested action.
- Reorganize information flow and phrasing.
- Do not add new information or change meaning.
- Keep 80–200 words.
- Output only the rewritten email.

Original email:
<<<PASTE ORIGINAL EMAIL HERE>>>
""",
    "paraphrase_p5_remove_keywords.txt": """Rewrite the email below to avoid common phishing-related keywords or phrases.

Instructions:
- Preserve intent and requested action.
- Avoid words such as “urgent”, “verify immediately”, “suspended”, and similar.
- Maintain a neutral, professional tone.
- Keep 80–200 words.
- Output only the rewritten email.

Original email:
<<<PASTE ORIGINAL EMAIL HERE>>>
""",
}

def init_templates():
    ensure_dirs()

    # Write system prompt template if missing
    sys_path = PROMPT_FILES["system"]
    if not os.path.exists(sys_path):
        with open(sys_path, "w", encoding="utf-8") as f:
            f.write(TEMPLATE_SYSTEM_PROMPT)
        print(f"[init] wrote {sys_path}")

    # Write other prompt templates if missing
    for fname, content in TEMPLATE_PROMPTS.items():
        path = os.path.join("prompts", fname)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[init] wrote {path}")

    print("[init] done. Edit prompts/ as needed.")

# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Create template prompts if missing.")
    parser.add_argument("--generate-phish", action="store_true", help="Generate phishing originals.")
    parser.add_argument("--generate-paraphrases", action="store_true", help="Generate paraphrases from originals.")
    parser.add_argument("--generate-legit", action="store_true", help="Generate legitimate emails.")
    parser.add_argument("--finalize", action="store_true", help="Merge raw JSONL → final JSONL/CSV + manifest.")
    parser.add_argument("--run-all", action="store_true", help="Run full pipeline end-to-end.")
    args = parser.parse_args()

    ensure_dirs()
    validate_prompt_files()

    if args.init:
        init_templates()
        return

    if args.run_all:
        gen = make_client()
        generate_phishing_originals(gen)
        generate_paraphrases(gen)
        generate_legitimate(gen)
        finalize_dataset()
        return

    did_any = False

    if args.generate_phish or args.generate_paraphrases or args.generate_legit:
        gen = make_client()

        if args.generate_phish:
            generate_phishing_originals(gen)
            did_any = True
        if args.generate_paraphrases:
            generate_paraphrases(gen)
            did_any = True
        if args.generate_legit:
            generate_legitimate(gen)
            did_any = True

    if args.finalize:
        finalize_dataset()
        did_any = True

    if not did_any:
        parser.print_help()


if __name__ == "__main__":
    main()
