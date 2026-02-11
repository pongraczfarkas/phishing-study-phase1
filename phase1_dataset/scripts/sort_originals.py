import json
from pathlib import Path

IN = Path("raw/phishing_original.jsonl")
OUT = Path("raw/phishing_original.jsonl")  # overwrite in place

def base_index(msg_id: str) -> int:
    # phish_<scenario>_<NNN>_orig
    return int(msg_id.rsplit("_", 2)[1])

rows = []
with IN.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

# Sort by scenario, then by numeric index
rows.sort(key=lambda r: (r["scenario"], base_index(r["msg_id"])))

with OUT.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Sorted and rewrote {OUT} ({len(rows)} records).")
