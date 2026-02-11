import argparse
import json
import random
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="final/phase1_dataset.jsonl", help="Path to final JSONL dataset")
    ap.add_argument("--outdir", default="data_splits", help="Output directory for splits")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio for splits")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    dataset_path = Path(args.dataset)
    outdir = Path(args.outdir)

    rows = read_jsonl(dataset_path)

    # --- Partition rows by type ---
    phish_orig = [r for r in rows if r.get("label") == "phish" and r.get("variant_type") == "original"]
    phish_para = [r for r in rows if r.get("label") == "phish" and r.get("variant_type") == "paraphrase"]
    legit = [r for r in rows if r.get("label") == "legit" or r.get("variant_type") == "legit"]

    # --- Build mapping: base_msg_id -> paraphrase rows ---
    para_by_base: dict[str, list[dict]] = {}
    for r in phish_para:
        base = r.get("base_msg_id")
        if not base:
            raise ValueError(f"Paraphrase missing base_msg_id: {r.get('msg_id')}")
        para_by_base.setdefault(base, []).append(r)

    # --- Split phishing originals by base message id ---
    phish_orig_ids = [r["msg_id"] for r in phish_orig]
    rng.shuffle(phish_orig_ids)

    n_test_phish = max(1, int(round(len(phish_orig_ids) * args.test_ratio)))
    test_phish_base_ids = set(phish_orig_ids[:n_test_phish])
    train_phish_base_ids = set(phish_orig_ids[n_test_phish:])

    # Safety check: disjoint
    assert test_phish_base_ids.isdisjoint(train_phish_base_ids)

    # --- Split legit normally ---
    legit_ids = [r.get("msg_id") for r in legit if r.get("msg_id") is not None]
    rng.shuffle(legit_ids)
    n_test_legit = max(1, int(round(len(legit_ids) * args.test_ratio)))
    test_legit_ids = set(legit_ids[:n_test_legit])
    train_legit_ids = set(legit_ids[n_test_legit:])

    # --- Assemble splits ---
    train_rows = []
    test_original_rows = []
    test_paraphrase_rows = []

    # Train: phishing originals (train base) + legit train
    train_rows.extend([r for r in phish_orig if r["msg_id"] in train_phish_base_ids])
    train_rows.extend([r for r in legit if r.get("msg_id") in train_legit_ids])

    # Test_original: phishing originals (test base) + legit test
    test_original_rows.extend([r for r in phish_orig if r["msg_id"] in test_phish_base_ids])
    test_original_rows.extend([r for r in legit if r.get("msg_id") in test_legit_ids])

    # Test_paraphrase: paraphrases of test phishing bases + same legit test
    for base_id in sorted(test_phish_base_ids):
        test_paraphrase_rows.extend(para_by_base.get(base_id, []))
    test_paraphrase_rows.extend([r for r in legit if r.get("msg_id") in test_legit_ids])

    # --- Leakage checks ---
    if any(r.get("variant_type") == "paraphrase" for r in train_rows):
        raise RuntimeError("Leakage: paraphrase found in training split")

    bad_para = []
    for r in test_paraphrase_rows:
        if r.get("label") == "phish" and r.get("variant_type") == "paraphrase":
            if r.get("base_msg_id") not in test_phish_base_ids:
                bad_para.append(r.get("msg_id"))
    if bad_para:
        raise RuntimeError(f"Leakage: found paraphrases whose base is not in test: {bad_para[:5]}...")

    assert test_legit_ids.isdisjoint(train_legit_ids)

    # --- Write outputs ---
    write_jsonl(outdir / "train.jsonl", train_rows)
    write_jsonl(outdir / "test_original.jsonl", test_original_rows)
    write_jsonl(outdir / "test_paraphrase.jsonl", test_paraphrase_rows)

    write_csv(outdir / "train.csv", train_rows)
    write_csv(outdir / "test_original.csv", test_original_rows)
    write_csv(outdir / "test_paraphrase.csv", test_paraphrase_rows)

    manifest = {
        "dataset": str(dataset_path),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "counts": {
            "phish_original_total": len(phish_orig),
            "phish_paraphrase_total": len(phish_para),
            "legit_total": len(legit),
            "train_total": len(train_rows),
            "test_original_total": len(test_original_rows),
            "test_paraphrase_total": len(test_paraphrase_rows),
        },
        "splits": {
            "train_phish_base_ids": sorted(train_phish_base_ids),
            "test_phish_base_ids": sorted(test_phish_base_ids),
            "train_legit_ids": sorted(train_legit_ids),
            "test_legit_ids": sorted(test_legit_ids),
        },
    }
    (outdir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Wrote splits to:", outdir.resolve())
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
