import os, json, re, time, random, hashlib
from datetime import datetime
from google import genai
from google.genai import types, errors

MODEL_NAME = "gemini-3-flash-preview"
TEMPERATURE = 0.95          # higher for paraphrases to avoid collisions
TOP_P = 0.98
MAX_OUTPUT_TOKENS = 900
SLEEP_SEC = 2.5
MAX_RETRIES = 12
BACKOFF_BASE = 2.5

LENGTH_MIN = 120
LENGTH_MAX = 200

RAW_ORIG = "raw/phishing_original.jsonl"
RAW_PARA = "raw/phishing_paraphrased.jsonl"
RAW_LEGIT = "raw/legitimate.jsonl"

SYSTEM_PROMPT = "prompts/system_master_v2.txt"
PARA_PROMPTS = {
  "p1": "prompts/paraphrase_p1_contextual_rephrasing_v2.txt",
  "p2": "prompts/paraphrase_p2_embedded_action_v2.txt",
  "p3": "prompts/paraphrase_p3_vocabulary_redistribution_v2.txt",
  "p4": "prompts/paraphrase_p4_reduced_urgency_v2.txt",
  "p5": "prompts/paraphrase_p5_operational_shift_v2.txt",
}

PARA_STRAT = {
  "p1": "professional_polish",
  "p2": "friendly_tone",
  "p3": "reduced_urgency",
  "p4": "structural_rewrite",
  "p5": "remove_suspicious_keywords",
}

TARGET_IDS = [
  "phish_impersonation_028_p4",
  "phish_invoice_fraud_007_p4",
  "phish_invoice_fraud_014_p4"
]

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_jsonl(path):
    if not os.path.exists(path): return []
    out=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out

def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def normalize(text):
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def wc(text):
    return len(re.findall(r"\S+", text.strip()))

def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def passes(text):
    if not text.strip(): return False, "empty"
    t = normalize(text)
    w = wc(t)
    if w < LENGTH_MIN: return False, f"too_short({w})"
    if w > LENGTH_MAX: return False, f"too_long({w})"
    low = t.lower()
    if "as an ai" in low: return False, "meta"
    return True, "ok"

def extract_text(resp):
    t = getattr(resp, "text", None)
    if t: return t.strip()
    chunks=[]
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            for p in parts:
                pt = getattr(p, "text", None)
                if pt: chunks.append(pt)
    return "\n".join(chunks).strip()

def polite_sleep(sec):
    time.sleep(sec + random.uniform(0.0, 1.0))

def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY)")
    client = genai.Client(api_key=api_key)

    system = load(SYSTEM_PROMPT)

    originals = {r["msg_id"]: r for r in read_jsonl(RAW_ORIG)}
    existing_para = {r["msg_id"] for r in read_jsonl(RAW_PARA)}

    # build hash set across all splits to avoid duplicates
    seen_hashes = set()
    for path in (RAW_ORIG, RAW_PARA, RAW_LEGIT):
        for r in read_jsonl(path):
            if r.get("text_hash"): seen_hashes.add(r["text_hash"])
            elif r.get("text"): seen_hashes.add(sha(normalize(r["text"])))

    today = datetime.now().strftime("%Y-%m-%d")

    for tid in TARGET_IDS:
        if tid in existing_para:
            print(f"{tid}: already exists, skipping")
            continue

        # parse: phish_<scenario>_<NNN>_<pk>
        parts = tid.rsplit("_", 2)
        prefix, idx_str, pk = parts[0], parts[1], parts[2]
        scenario = prefix.replace("phish_", "")
        base_msg_id = f"phish_{scenario}_{idx_str}_orig"

        orig = originals.get(base_msg_id)
        if not orig:
            print(f"{tid}: missing base original {base_msg_id}")
            continue

        template = load(PARA_PROMPTS[pk])
        strategy = PARA_STRAT[pk]

        # Add a nonce + stronger rewrite constraint to avoid duplicate collisions
        nonce = f"nonce-{random.randint(100000, 999999)}"

        base_prompt = template.replace("<<<PASTE ORIGINAL EMAIL HERE>>>", orig["text"])
        base_prompt += (
            f"\n\nDIVERSITY KEY: {nonce}\n"
            "Rewrite constraints (must follow):\n"
            "- Use different sentence order and different phrasing than the original.\n"
            "- Do not reuse full sentences verbatim.\n"
            "- Keep meaning, but change wording and structure.\n"
            "- 120–200 words.\n"
            "- Output only the rewritten email.\n"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                cfg = types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
                resp = client.models.generate_content(model=MODEL_NAME, contents=base_prompt, config=cfg)
                text = normalize(extract_text(resp))

                ok, reason = passes(text)
                if not ok:
                    wait = min(20, BACKOFF_BASE * (2 ** (attempt - 1)))
                    print(f"  {tid} attempt {attempt} -> {reason} (wait {wait}s)")
                    polite_sleep(wait)
                    continue

                h = sha(text)
                if h in seen_hashes:
                    wait = min(20, BACKOFF_BASE * (2 ** (attempt - 1)))
                    print(f"  {tid} attempt {attempt} -> duplicate_hash (wait {wait}s)")
                    polite_sleep(wait)
                    continue

                rec = {
                    "msg_id": tid,
                    "label": "phish",
                    "scenario": scenario,
                    "variant_type": "paraphrase",
                    "paraphrase_strategy": strategy,
                    "base_msg_id": base_msg_id,
                    "source": "generated_llm",
                    "generation_model": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "prompt_id": f"{scenario}_{pk}_v1_fill",
                    "text": text,
                    "length_words": wc(text),
                    "text_hash": h,
                    "date_generated": today,
                }
                append_jsonl(RAW_PARA, rec)
                seen_hashes.add(h)
                existing_para.add(tid)
                print(f"{tid} -> OK")
                polite_sleep(SLEEP_SEC)
                break

            except errors.APIError as e:
                wait = min(20, BACKOFF_BASE * (2 ** (attempt - 1)))
                print(f"  {tid} APIError attempt {attempt}: {str(e)[:120]} (wait {wait}s)")
                polite_sleep(wait)

        else:
            print(f"[WARN] {tid} failed after {MAX_RETRIES} attempts")

if __name__ == "__main__":
    main()
