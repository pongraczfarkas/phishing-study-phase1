import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from experiments.common.plots import ensure_dir, plot_metric_bars, plot_deltas


# ----------------------------
# IO helpers
# ----------------------------
def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def label_to_int(s: pd.Series) -> np.ndarray:
    mapping = {"phish": 1, "legit": 0}
    bad = set(s.unique()) - set(mapping.keys())
    if bad:
        raise ValueError(f"Unexpected labels in dataset: {bad}")
    return s.map(mapping).to_numpy()


def compute_group_id(df: pd.DataFrame) -> pd.Series:
    """
    Ensures paraphrase families stay together:
      - phishing paraphrase -> base_msg_id
      - otherwise -> msg_id
    """
    is_para = (df["label"] == "phish") & (df["variant_type"] == "paraphrase")
    if is_para.any() and "base_msg_id" not in df.columns:
        raise ValueError("Found paraphrases but no base_msg_id column.")
    gid = df["msg_id"].astype(str)
    gid.loc[is_para] = df.loc[is_para, "base_msg_id"].astype(str)
    return gid


# ----------------------------
# Metrics
# ----------------------------
def eval_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)  # [[TN, FP],[FN, TP]]
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def summarize(vals: List[float]) -> dict:
    arr = np.array(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


# ----------------------------
# Feature engineering
# ----------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}")
MONEY_RE = re.compile(r"(\$|€|£)\s?\d+|\b\d+\s?(usd|eur|gbp)\b", re.IGNORECASE)

KEYWORDS = [
    "invoice", "payment", "billing", "vendor", "bank", "wire",
    "account", "password", "login", "verify", "verification", "credentials",
    "security", "compliance", "attestation", "portal", "recover", "recovery",
    "urgent", "immediately", "asap", "action required",
]


def basic_numeric_features(texts: np.ndarray) -> np.ndarray:
    """
    Returns dense feature matrix shape (n_samples, n_features)
    """
    feats = []
    for t in texts:
        s = str(t)
        s_strip = s.strip()
        n_chars = len(s_strip)
        words = re.findall(r"\S+", s_strip)
        n_words = len(words)

        n_upper = sum(1 for c in s_strip if c.isupper())
        n_lower = sum(1 for c in s_strip if c.islower())
        n_digits = sum(1 for c in s_strip if c.isdigit())
        n_punct = sum(1 for c in s_strip if c in ".,;:!?()[]{}<>\"'")

        n_lines = s_strip.count("\n") + 1 if s_strip else 0
        exclam = s_strip.count("!")
        question = s_strip.count("?")

        urls = URL_RE.findall(s_strip)
        emails = EMAIL_RE.findall(s_strip)
        phones = PHONE_RE.findall(s_strip)
        money = MONEY_RE.findall(s_strip)

        # ratios (avoid div by zero)
        char_den = max(n_chars, 1)
        alpha_den = max(n_upper + n_lower, 1)

        caps_ratio = n_upper / alpha_den
        digit_ratio = n_digits / char_den
        punct_ratio = n_punct / char_den

        avg_word_len = float(np.mean([len(w) for w in words])) if n_words else 0.0

        kw_hits = sum(1 for k in KEYWORDS if k in s_strip.lower())

        feats.append([
            n_chars,
            n_words,
            n_lines,
            avg_word_len,
            caps_ratio,
            digit_ratio,
            punct_ratio,
            exclam,
            question,
            len(urls),
            len(emails),
            len(phones),
            len(money),
            kw_hits,
            1.0 if len(urls) > 0 else 0.0,
            1.0 if "http" in s_strip.lower() else 0.0,
        ])

    return np.array(feats, dtype=float)


def build_text_vectorizers(seed: int):
    """
    Two TF-IDF views:
      - word ngrams (captures topical wording)
      - char ngrams (captures obfuscation / formatting patterns)
    """
    tfidf_word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
    )
    tfidf_char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
    )
    return tfidf_word, tfidf_char


def fit_transform_features(
    train_text: np.ndarray,
    test_text: np.ndarray,
    use_char_ngrams: bool,
    seed: int
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build RF features: [numeric_dense | tfidf_word | (optional) tfidf_char]
    """
    Xtr_num = basic_numeric_features(train_text)
    Xte_num = basic_numeric_features(test_text)

    tfidf_word, tfidf_char = build_text_vectorizers(seed=seed)

    Xtr_w = tfidf_word.fit_transform(train_text)
    Xte_w = tfidf_word.transform(test_text)

    blocks_tr = [sparse.csr_matrix(Xtr_num), Xtr_w]
    blocks_te = [sparse.csr_matrix(Xte_num), Xte_w]

    if use_char_ngrams:
        Xtr_c = tfidf_char.fit_transform(train_text)
        Xte_c = tfidf_char.transform(test_text)
        blocks_tr.append(Xtr_c)
        blocks_te.append(Xte_c)

    Xtr = sparse.hstack(blocks_tr).tocsr()
    Xte = sparse.hstack(blocks_te).tocsr()
    return Xtr, Xte


def build_rf(seed: int, n_estimators: int, max_depth: int | None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="final/phase1_dataset.jsonl")
    ap.add_argument("--out", default="experiments/rf/results/metrics_groupcv.json")
    ap.add_argument("--plots-dir", default="experiments/rf/results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)

    # RF config
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=20, help="Use 0 for None (unlimited)")
    ap.add_argument("--use-char-ngrams", action="store_true",
                    help="Add char_wb TF-IDF (3-5) to features (often improves robustness)")

    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.plots_dir)
    ensure_dir(plots_dir)

    df = read_jsonl(dataset_path)

    for col in ["msg_id", "label", "variant_type", "text"]:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    df["label_int"] = label_to_int(df["label"])
    df["group_id"] = compute_group_id(df)

    # Group table for stratified splitting at group level
    g = df.groupby("group_id", as_index=False)["label_int"].max()
    group_ids = g["group_id"].to_numpy()
    group_y = g["label_int"].to_numpy()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    folds_out = []

    # store per-fold metric lists for plotting
    f1_A_list, f1_B_list = [], []
    r_A_list, r_B_list = [], []
    dF1_list, dR_list = [], []

    max_depth = None if args.max_depth == 0 else args.max_depth

    for fold_i, (train_g_idx, test_g_idx) in enumerate(skf.split(group_ids, group_y), start=1):
        train_groups = set(group_ids[train_g_idx])
        test_groups = set(group_ids[test_g_idx])

        fold_train = df[df["group_id"].isin(train_groups)].copy()
        fold_test = df[df["group_id"].isin(test_groups)].copy()

        # Train: legit + phish originals only
        train_mask = (fold_train["label"] == "legit") | (
            (fold_train["label"] == "phish") & (fold_train["variant_type"] == "original")
        )
        train_df = fold_train[train_mask]

        # Condition A: test originals + legit
        testA_mask = (fold_test["label"] == "legit") | (
            (fold_test["label"] == "phish") & (fold_test["variant_type"] == "original")
        )
        testA_df = fold_test[testA_mask]

        # Condition B: test paraphrases + legit
        testB_mask = (fold_test["label"] == "legit") | (
            (fold_test["label"] == "phish") & (fold_test["variant_type"] == "paraphrase")
        )
        testB_df = fold_test[testB_mask]

        # Sanity check for paraphrase families
        if (testB_df["label"] == "phish").any():
            bad = testB_df[
                (testB_df["label"] == "phish") &
                (testB_df["variant_type"] == "paraphrase") &
                (~testB_df["base_msg_id"].astype(str).isin(test_groups))
            ]
            if len(bad) > 0:
                raise RuntimeError("Leakage: paraphrase base_msg_id not in same fold groups.")

        # Fit features on TRAIN, evaluate A and B with different TEST sets
        Xtr, XA = fit_transform_features(
            train_df["text"].astype(str).to_numpy(),
            testA_df["text"].astype(str).to_numpy(),
            use_char_ngrams=args.use_char_ngrams,
            seed=args.seed,
        )
        _, XB = fit_transform_features(
            train_df["text"].astype(str).to_numpy(),
            testB_df["text"].astype(str).to_numpy(),
            use_char_ngrams=args.use_char_ngrams,
            seed=args.seed,
        )
        ytr = train_df["label_int"].to_numpy()
        yA = testA_df["label_int"].to_numpy()
        yB = testB_df["label_int"].to_numpy()

        clf = build_rf(args.seed, n_estimators=args.n_estimators, max_depth=max_depth)
        clf.fit(Xtr, ytr)

        pA = clf.predict(XA)
        resA = eval_binary(yA, pA)

        pB = clf.predict(XB)
        resB = eval_binary(yB, pB)

        fold_rec = {
            "fold": fold_i,
            "n_groups_train": int(len(train_groups)),
            "n_groups_test": int(len(test_groups)),
            "n_train_rows": int(len(train_df)),
            "n_testA_rows": int(len(testA_df)),
            "n_testB_rows": int(len(testB_df)),
            "condition_A": resA,
            "condition_B": resB,
            "delta_f1": float(resA["f1"] - resB["f1"]),
            "delta_recall": float(resA["recall"] - resB["recall"]),
        }
        folds_out.append(fold_rec)

        f1_A_list.append(resA["f1"])
        f1_B_list.append(resB["f1"])
        r_A_list.append(resA["recall"])
        r_B_list.append(resB["recall"])
        dF1_list.append(fold_rec["delta_f1"])
        dR_list.append(fold_rec["delta_recall"])

        print(f"\n=== Fold {fold_i}/{args.folds} ===")
        print("A (originals+legit):", {k: resA[k] for k in ["accuracy", "precision", "recall", "f1"]})
        print("B (paraphrases+legit):", {k: resB[k] for k in ["accuracy", "precision", "recall", "f1"]})
        print("ΔF1:", fold_rec["delta_f1"], "ΔRecall:", fold_rec["delta_recall"])

    agg = {
        "A_f1": summarize([f["condition_A"]["f1"] for f in folds_out]),
        "B_f1": summarize([f["condition_B"]["f1"] for f in folds_out]),
        "A_recall": summarize([f["condition_A"]["recall"] for f in folds_out]),
        "B_recall": summarize([f["condition_B"]["recall"] for f in folds_out]),
        "delta_f1": summarize([f["delta_f1"] for f in folds_out]),
        "delta_recall": summarize([f["delta_recall"] for f in folds_out]),
    }

    out = {
        "model": "random_forest_engineered",
        "dataset": str(dataset_path),
        "seed": args.seed,
        "folds": args.folds,
        "grouping": {
            "rule": "phish paraphrase→base_msg_id else→msg_id",
            "n_groups_total": int(len(group_ids)),
        },
        "training_rule": "train on legit + phishing originals only",
        "conditions": {
            "A": "test on legit + phishing originals",
            "B": "test on legit + phishing paraphrases",
        },
        "rf": {
            "n_estimators": args.n_estimators,
            "max_depth": max_depth,
            "class_weight": "balanced_subsample",
            "use_char_ngrams": bool(args.use_char_ngrams),
        },
        "features": {
            "numeric": "length/case/digit/punct/lines + url/email/phone/money counts + keyword hit count",
            "tfidf_word": {"ngram_range": [1, 2], "min_df": 2, "max_df": 0.95},
            "tfidf_char": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 2, "max_df": 0.95}
            if args.use_char_ngrams else None,
        },
        "fold_results": folds_out,
        "aggregate": agg,
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Plots
    plot_metric_bars(
        f1_A_list, f1_B_list,
        plots_dir / "f1_A_vs_B.png",
        title="Random Forest F1 (mean ± std across folds)",
        ylabel="F1",
    )
    plot_metric_bars(
        r_A_list, r_B_list,
        plots_dir / "recall_A_vs_B.png",
        title="Random Forest Recall (mean ± std across folds)",
        ylabel="Recall",
    )
    plot_deltas(
        dF1_list,
        plots_dir / "delta_f1_per_fold.png",
        title="Random Forest ΔF1 per fold (A − B)",
        ylabel="ΔF1",
    )
    plot_deltas(
        dR_list,
        plots_dir / "delta_recall_per_fold.png",
        title="Random Forest ΔRecall per fold (A − B)",
        ylabel="ΔRecall",
    )

    print("\n=== Aggregate (mean ± std) ===")
    print("F1 A:", agg["A_f1"])
    print("F1 B:", agg["B_f1"])
    print("ΔF1 :", agg["delta_f1"])
    print("ΔRecall:", agg["delta_recall"])
    print("\nSaved:", out_path.resolve())
    print("Saved plots to:", plots_dir.resolve())


if __name__ == "__main__":
    main()