import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from plots import ensure_dir, plot_metric_bars, plot_deltas

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


def summarize(vals: list[float]) -> dict:
    arr = np.array(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def build_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                strip_accents="unicode",
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )),
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="final/phase1_dataset.jsonl",
                    help="Path to full dataset JSONL (recommended: final/phase1_dataset.jsonl)")
    ap.add_argument("--out", default="experiments/lr_tfidf/results/metrics_groupcv.json",
                    help="Output JSON metrics path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--folds", type=int, default=5, help="Number of folds")
    ap.add_argument("--plots-dir", default="experiments/lr_tfidf/results",
                    help="Directory to save plots")

    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(dataset_path)

    # Basic required columns
    for col in ["msg_id", "label", "variant_type", "text"]:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    df["label_int"] = label_to_int(df["label"])
    df["group_id"] = compute_group_id(df)

    # Build a GROUP table for stratified splitting at group level
    # Group label: phish if any phish in the group else legit
    g = df.groupby("group_id", as_index=False)["label_int"].max()
    group_ids = g["group_id"].to_numpy()
    group_y = g["label_int"].to_numpy()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    folds_out = []

    for fold_i, (train_g_idx, test_g_idx) in enumerate(skf.split(group_ids, group_y), start=1):
        train_groups = set(group_ids[train_g_idx])
        test_groups = set(group_ids[test_g_idx])

        fold_train = df[df["group_id"].isin(train_groups)].copy()
        fold_test = df[df["group_id"].isin(test_groups)].copy()

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

        # Sanity: ensure paraphrase families stayed in fold_test if any paraphrases exist
        if (testB_df["label"] == "phish").any():
            bad = testB_df[
                (testB_df["label"] == "phish") &
                (testB_df["variant_type"] == "paraphrase") &
                (~testB_df["base_msg_id"].astype(str).isin(test_groups))
            ]
            if len(bad) > 0:
                raise RuntimeError("Leakage: paraphrase base_msg_id not in same fold groups.")

        model = build_model(args.seed)
        model.fit(train_df["text"].astype(str).to_numpy(), train_df["label_int"].to_numpy())

        # Evaluate A
        yA = testA_df["label_int"].to_numpy()
        pA = model.predict(testA_df["text"].astype(str).to_numpy())
        resA = eval_binary(yA, pA)

        # Evaluate B
        yB = testB_df["label_int"].to_numpy()
        pB = model.predict(testB_df["text"].astype(str).to_numpy())
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

        print(f"\n=== Fold {fold_i}/{args.folds} ===")
        print("A (originals+legit):", {k: resA[k] for k in ["accuracy", "precision", "recall", "f1"]})
        print("B (paraphrases+legit):", {k: resB[k] for k in ["accuracy", "precision", "recall", "f1"]})
        print("ΔF1:", fold_rec["delta_f1"], "ΔRecall:", fold_rec["delta_recall"])

    # Aggregate
    agg = {
        "A_f1": summarize([f["condition_A"]["f1"] for f in folds_out]),
        "B_f1": summarize([f["condition_B"]["f1"] for f in folds_out]),
        "A_recall": summarize([f["condition_A"]["recall"] for f in folds_out]),
        "B_recall": summarize([f["condition_B"]["recall"] for f in folds_out]),
        "delta_f1": summarize([f["delta_f1"] for f in folds_out]),
        "delta_recall": summarize([f["delta_recall"] for f in folds_out]),
    }

    out = {
        "model": "tfidf_logreg",
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
        "fold_results": folds_out,
        "aggregate": agg,
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # --- Plots ---
    plots_dir = Path(args.plots_dir)
    ensure_dir(plots_dir)

    f1_A = [f["condition_A"]["f1"] for f in folds_out]
    f1_B = [f["condition_B"]["f1"] for f in folds_out]
    rec_A = [f["condition_A"]["recall"] for f in folds_out]
    rec_B = [f["condition_B"]["recall"] for f in folds_out]
    dF1 = [f["delta_f1"] for f in folds_out]
    dR = [f["delta_recall"] for f in folds_out]

    plot_metric_bars(
        f1_A, f1_B,
        plots_dir / "f1_A_vs_B.png",
        title="F1 (mean ± std across folds)",
        ylabel="F1",
    )
    plot_metric_bars(
        rec_A, rec_B,
        plots_dir / "recall_A_vs_B.png",
        title="Recall (mean ± std across folds)",
        ylabel="Recall",
    )
    plot_deltas(
        dF1,
        plots_dir / "delta_f1_per_fold.png",
        title="ΔF1 per fold (A − B)",
        ylabel="ΔF1",
    )
    plot_deltas(
        dR,
        plots_dir / "delta_recall_per_fold.png",
        title="ΔRecall per fold (A − B)",
        ylabel="ΔRecall",
    )

    print("Saved plots to:", plots_dir.resolve())

    print("\n=== Aggregate (mean ± std) ===")
    print("F1 A:", agg["A_f1"])
    print("F1 B:", agg["B_f1"])
    print("ΔF1 :", agg["delta_f1"])
    print("ΔRecall:", agg["delta_recall"])
    print("\nSaved:", out_path.resolve())


if __name__ == "__main__":
    main()
