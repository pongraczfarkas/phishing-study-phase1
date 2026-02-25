import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)

from experiments.common.plots import ensure_dir, plot_metric_bars, plot_deltas


# -----------------------------
# IO + preprocessing
# -----------------------------
def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def label_to_int(s: pd.Series) -> np.ndarray:
    mapping = {"legit": 0, "phish": 1}
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
    gid = df["msg_id"].astype(str)
    is_para = (df["label"] == "phish") & (df["variant_type"] == "paraphrase")
    if is_para.any() and "base_msg_id" not in df.columns:
        raise ValueError("Found paraphrases but no base_msg_id column.")
    gid.loc[is_para] = df.loc[is_para, "base_msg_id"].astype(str)
    return gid


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    # Trainer expects the label column to be named "labels"
    return Dataset.from_dict(
        {
            "text": df["text"].astype(str).tolist(),
            "labels": df["label_int"].astype(int).tolist(),
        }
    )


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


def eval_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def summarize(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


# -----------------------------
# Tokenization
# -----------------------------
def tokenize(ds: Dataset, tokenizer, max_len: int) -> Dataset:
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    # Remove raw text after tokenization; keep only tensors + labels
    return ds.map(_tok, batched=True, remove_columns=["text"])


# -----------------------------
# Version-robust TrainingArguments builder
# -----------------------------
def make_training_args_kwargs(**kwargs) -> dict:
    """
    transformers API changed:
      - older: evaluation_strategy
      - newer: eval_strategy
    Also: some keys may not exist in some versions.
    This filters/massages kwargs to match the installed transformers version.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    out = dict(kwargs)

    # Map eval key depending on what's supported
    if "eval_strategy" in out and "eval_strategy" not in params and "evaluation_strategy" in params:
        out["evaluation_strategy"] = out.pop("eval_strategy")
    if "evaluation_strategy" in out and "evaluation_strategy" not in params and "eval_strategy" in params:
        out["eval_strategy"] = out.pop("evaluation_strategy")

    # Drop unsupported keys defensively
    for k in list(out.keys()):
        if k not in params:
            out.pop(k)

    # Remove None values (older versions sometimes dislike explicit None)
    for k in list(out.keys()):
        if out[k] is None:
            out.pop(k)

    return out


# -----------------------------
# Optional: class-weighted loss
# -----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # NOTE: transformers Trainer may pass extra kwargs like num_items_in_batch
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="final/phase1_dataset.jsonl")
    ap.add_argument("--plots-dir", default="experiments/bert/results")
    ap.add_argument("--out", default="experiments/bert/results/metrics_groupcv.json")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)

    ap.add_argument("--checkpoint", default="bert-base-uncased")
    ap.add_argument("--max-len", type=int, default=256)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)

    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--eval-strategy", default="epoch", choices=["epoch", "steps"])
    ap.add_argument("--eval-steps", type=int, default=50)

    ap.add_argument("--early-stop", type=int, default=2, help="Patience in eval cycles")
    ap.add_argument("--use-class-weights", action="store_true")

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    ap.add_argument("--max-train-rows", type=int, default=0, help="0 = no cap")

    args = ap.parse_args()

    # Windows/tokenizers can over-parallelize
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    plots_dir = Path(args.plots_dir)
    out_path = Path(args.out)

    ensure_dir(plots_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_jsonl(dataset_path)
    for col in ["msg_id", "label", "variant_type", "text"]:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    df["label_int"] = label_to_int(df["label"])
    df["group_id"] = compute_group_id(df)

    # Group-level stratification
    g = df.groupby("group_id", as_index=False)["label_int"].max()
    group_ids = g["group_id"].to_numpy()
    group_y = g["label_int"].to_numpy()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    folds_out = []
    f1A, f1B, dF1 = [], [], []
    rA, rB, dR = [], [], []

    for fold_i, (train_g_idx, test_g_idx) in enumerate(skf.split(group_ids, group_y), start=1):
        print(f"\n=== Fold {fold_i}/{args.folds} ===")

        train_groups = set(group_ids[train_g_idx])
        test_groups = set(group_ids[test_g_idx])

        fold_train = df[df["group_id"].isin(train_groups)].copy()
        fold_test = df[df["group_id"].isin(test_groups)].copy()

        # Train: legit + phishing originals only
        train_mask = (fold_train["label"] == "legit") | (
            (fold_train["label"] == "phish") & (fold_train["variant_type"] == "original")
        )
        train_df = fold_train[train_mask].copy()

        # Optional cap for speed
        if args.max_train_rows and args.max_train_rows > 0:
            train_df = train_df.sample(n=min(args.max_train_rows, len(train_df)), random_state=args.seed)

        # Test A: legit + phishing originals
        testA_df = fold_test[
            (fold_test["label"] == "legit")
            | ((fold_test["label"] == "phish") & (fold_test["variant_type"] == "original"))
        ].copy()

        # Test B: legit + phishing paraphrases
        testB_df = fold_test[
            (fold_test["label"] == "legit")
            | ((fold_test["label"] == "phish") & (fold_test["variant_type"] == "paraphrase"))
        ].copy()

        # HF datasets
        train_ds_full = tokenize(to_hf_dataset(train_df), tokenizer, args.max_len)

        # Validation split from training partition for early stopping / best model
        train_val = train_ds_full.train_test_split(test_size=0.2, seed=args.seed)
        train_split = train_val["train"]
        val_split = train_val["test"]

        testA_ds = tokenize(to_hf_dataset(testA_df), tokenizer, args.max_len)
        testB_ds = tokenize(to_hf_dataset(testB_df), tokenizer, args.max_len)

        # Model per fold
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=2)

        # Class weights (per fold)
        class_weights = None
        if args.use_class_weights:
            counts = train_df["label_int"].value_counts().to_dict()
            # inverse frequency
            w0 = 1.0 / max(counts.get(0, 1), 1)
            w1 = 1.0 / max(counts.get(1, 1), 1)
            s = w0 + w1
            class_weights = torch.tensor([w0 / s, w1 / s], dtype=torch.float32)

        fold_outdir = plots_dir / f"fold_{fold_i}"
        fold_outdir.mkdir(parents=True, exist_ok=True)

        # Warmup steps (warmup_ratio is deprecated in newer versions)
        # We compute from an estimate of total steps.
        steps_per_epoch = int(np.ceil(len(train_split) / max(args.batch_size, 1)))
        total_steps = max(1, steps_per_epoch * max(args.epochs, 1))
        warmup_steps = int(total_steps * float(args.warmup_ratio))

        training_kwargs = make_training_args_kwargs(
            output_dir=str(fold_outdir / "ckpt"),
            seed=args.seed,

            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=warmup_steps,

            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,

            gradient_accumulation_steps=args.grad_accum,

            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,

            save_strategy=args.eval_strategy,
            save_steps=args.eval_steps if args.eval_strategy == "steps" else None,

            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,

            logging_steps=25,
            report_to=[],

            fp16=args.fp16,
            bf16=args.bf16,

            save_total_limit=1,
        )

        training_args = TrainingArguments(**training_kwargs)

        # Choose trainer class
        if args.use_class_weights:
            trainer = WeightedTrainer(
                model=model,
                args=training_args,
                train_dataset=train_split,
                eval_dataset=val_split,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop)],
                class_weights=class_weights,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_split,
                eval_dataset=val_split,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop)],
            )

        trainer.train()

        # Predict A
        predA = trainer.predict(testA_ds)
        pA = np.argmax(predA.predictions, axis=1)
        yA = testA_df["label_int"].to_numpy()
        resA = eval_binary(yA, pA)

        # Predict B
        predB = trainer.predict(testB_ds)
        pB = np.argmax(predB.predictions, axis=1)
        yB = testB_df["label_int"].to_numpy()
        resB = eval_binary(yB, pB)

        delta_f1 = float(resA["f1"] - resB["f1"])
        delta_rec = float(resA["recall"] - resB["recall"])

        print("A (originals+legit):", {k: resA[k] for k in ["accuracy", "precision", "recall", "f1"]})
        print("B (paraphrases+legit):", {k: resB[k] for k in ["accuracy", "precision", "recall", "f1"]})
        print("ΔF1:", delta_f1, "ΔRecall:", delta_rec)

        folds_out.append(
            {
                "fold": fold_i,
                "n_train_rows": int(len(train_df)),
                "n_testA_rows": int(len(testA_df)),
                "n_testB_rows": int(len(testB_df)),
                "warmup_steps": int(warmup_steps),
                "condition_A": resA,
                "condition_B": resB,
                "delta_f1": delta_f1,
                "delta_recall": delta_rec,
            }
        )

        f1A.append(resA["f1"])
        f1B.append(resB["f1"])
        dF1.append(delta_f1)
        rA.append(resA["recall"])
        rB.append(resB["recall"])
        dR.append(delta_rec)

    agg = {
        "A_f1": summarize(f1A),
        "B_f1": summarize(f1B),
        "A_recall": summarize(rA),
        "B_recall": summarize(rB),
        "delta_f1": summarize(dF1),
        "delta_recall": summarize(dR),
    }

    out = {
        "model": "bert_finetuned",
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "folds": args.folds,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "early_stop_patience": args.early_stop,
        "use_class_weights": bool(args.use_class_weights),
        "train_rule": "train on legit + phishing originals only",
        "conditions": {
            "A": "test on legit + phishing originals",
            "B": "test on legit + phishing paraphrases",
        },
        "aggregate": agg,
        "fold_results": folds_out,
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Plots (same API as your other models)
    plot_metric_bars(f1A, f1B, plots_dir / "f1_A_vs_B.png", title="BERT F1: Condition A vs B", ylabel="F1")
    plot_metric_bars(rA, rB, plots_dir / "recall_A_vs_B.png", title="BERT Recall: Condition A vs B", ylabel="Recall")
    plot_deltas(dF1, plots_dir / "delta_f1.png", title="BERT ΔF1 per fold (A − B)", ylabel="ΔF1")
    plot_deltas(dR, plots_dir / "delta_recall.png", title="BERT ΔRecall per fold (A − B)", ylabel="ΔRecall")

    print("\n=== Aggregate (mean ± std) ===")
    print("F1 A:", agg["A_f1"])
    print("F1 B:", agg["B_f1"])
    print("ΔF1 :", agg["delta_f1"])
    print("ΔRecall:", agg["delta_recall"])
    print("\nSaved:", out_path.resolve())
    print("Saved plots to:", plots_dir.resolve())


if __name__ == "__main__":
    main()