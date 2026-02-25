import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    cm = confusion_matrix(y_true, y_pred)
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
# Text preprocessing / vocab
# ----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\s]", re.UNICODE)


def normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: List[str], vocab_size: int, min_freq: int) -> Dict[str, int]:
    # 0: PAD, 1: UNK
    freq: Dict[str, int] = {}
    for t in texts:
        for tok in tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1

    items = [(w, c) for w, c in freq.items() if c >= min_freq]
    items.sort(key=lambda x: x[1], reverse=True)

    vocab = {"[PAD]": 0, "[UNK]": 1}
    for w, _ in items[: max(0, vocab_size - 2)]:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    ids = [vocab.get(tok, 1) for tok in tokenize(text)]
    length = min(len(ids), max_len)
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))
    return ids, length


# ----------------------------
# Torch Dataset
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, X_ids: np.ndarray, lengths: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X_ids, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.lengths[idx], self.y[idx]


# ----------------------------
# New LSTM Algorithm: Packed BiLSTM + MultiheadAttention pooling
# ----------------------------
class PackedAttnBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_layers: int,
                 attn_heads: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM supports dropout only when num_layers > 1
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

        # learned query token for attention pooling
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size * 2))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B,T), lengths: (B,)
        emb = self.embedding(x)  # (B,T,E)

        # Pack padded sequences so LSTM ignores PAD tokens
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, Tmax, 2H)
        out = self.norm(out)

        B, Tmax, _ = out.shape
        idxs = torch.arange(Tmax, device=out.device).unsqueeze(0).expand(B, Tmax)
        key_padding_mask = idxs >= lengths.unsqueeze(1)  # (B, Tmax)

        # Attention pooling: query attends to sequence
        q = self.query.expand(B, -1, -1)  # (B,1,2H)
        attn_out, _ = self.attn(q, out, out, key_padding_mask=key_padding_mask)
        pooled = attn_out.squeeze(1)  # (B,2H)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(1)
        return logits


# ----------------------------
# Train / eval
# ----------------------------
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_arrays(texts: List[str], vocab: Dict[str, int], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, L = [], []
    for t in texts:
        ids, length = encode(t, vocab, max_len)
        X.append(ids)
        L.append(length if length > 0 else 1)
    return np.array(X, dtype=np.int64), np.array(L, dtype=np.int64)


def build_sampler(y: np.ndarray) -> WeightedRandomSampler:
    # WeightedRandomSampler expects one weight per sample
    y = y.astype(int)
    counts = np.bincount(y, minlength=2)
    w0 = 1.0 / max(counts[0], 1)
    w1 = 1.0 / max(counts[1], 1)
    weights = np.array([w1 if yi == 1 else w0 for yi in y], dtype=np.float32)
    return WeightedRandomSampler(
        weights=torch.tensor(weights),
        num_samples=len(weights),
        replacement=True
    )


def train_one_fold(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    cfg: dict,
    seed: int,
    device: torch.device,
) -> Tuple[PackedAttnBiLSTM, Dict[str, float]]:
    vocab = build_vocab(train_texts, vocab_size=cfg["vocab_size"], min_freq=cfg["min_freq"])
    Xtr, Ltr = make_arrays(train_texts, vocab, cfg["max_len"])
    Xva, Lva = make_arrays(val_texts, vocab, cfg["max_len"])

    ds_tr = TextDataset(Xtr, Ltr, train_labels.astype(np.float32))
    ds_va = TextDataset(Xva, Lva, val_labels.astype(np.float32))

    sampler = build_sampler(train_labels)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], sampler=sampler)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False)

    model = PackedAttnBiLSTM(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        attn_heads=cfg["attn_heads"],
        dropout=cfg["dropout"],
    ).to(device)

    # Stable baseline loss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for xb, lb, yb in dl_tr:
            xb = xb.to(device)
            lb = lb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()

            # gradient clipping for stability on small data
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

            optimizer.step()

        # validation
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, lb, yb in dl_va:
                xb = xb.to(device)
                lb = lb.to(device)
                yb = yb.to(device)
                logits = model(xb, lb)
                loss = criterion(logits, yb)
                va_losses.append(float(loss.detach().cpu().item()))

        val_loss = float(np.mean(va_losses)) if va_losses else 0.0
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # attach vocab for inference
    model._vocab = vocab  # type: ignore[attr-defined]

    info = {
        "vocab_final": int(len(vocab)),
        "best_val_loss": float(best_val),
    }
    return model, info


def predict(model: PackedAttnBiLSTM, texts: List[str], max_len: int, device: torch.device) -> np.ndarray:
    vocab = model._vocab  # type: ignore[attr-defined]
    X, L = make_arrays(texts, vocab, max_len)
    ds = TextDataset(X, L, np.zeros(len(texts), dtype=np.float32))
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    probs = []
    model.eval()
    with torch.no_grad():
        for xb, lb, _ in dl:
            xb = xb.to(device)
            lb = lb.to(device)
            logits = model(xb, lb)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)

    prob = np.concatenate(probs, axis=0).reshape(-1)
    return (prob >= 0.5).astype(int)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="final/phase1_dataset.jsonl")
    ap.add_argument("--out", default="experiments/lstm/results/metrics_groupcv_v2.json")
    ap.add_argument("--plots-dir", default="experiments/lstm/results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)

    # model config
    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=96)
    ap.add_argument("--hidden-size", type=int, default=96)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.25)

    # training config
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    args = ap.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plots_dir = Path(args.plots_dir)
    ensure_dir(plots_dir)

    df = read_jsonl(dataset_path)
    for col in ["msg_id", "label", "variant_type", "text"]:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    df["text"] = df["text"].astype(str).map(normalize_text)
    df["label_int"] = label_to_int(df["label"])
    df["group_id"] = compute_group_id(df)

    # Group-level stratified CV
    g = df.groupby("group_id", as_index=False)["label_int"].max()
    group_ids = g["group_id"].to_numpy()
    group_y = g["label_int"].to_numpy()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    cfg = {
        "vocab_size": args.vocab_size,
        "min_freq": args.min_freq,
        "max_len": args.max_len,
        "embed_dim": args.embed_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "attn_heads": args.attn_heads,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "grad_clip": args.grad_clip,
    }

    folds_out = []
    f1_A_list, f1_B_list = [], []
    r_A_list, r_B_list = [], []
    dF1_list, dR_list = [], []

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

        # Leakage check
        if (testB_df["label"] == "phish").any():
            bad = testB_df[
                (testB_df["label"] == "phish") &
                (testB_df["variant_type"] == "paraphrase") &
                (~testB_df["base_msg_id"].astype(str).isin(test_groups))
            ]
            if len(bad) > 0:
                raise RuntimeError("Leakage: paraphrase base_msg_id not in same fold groups.")

        # Train/val split within training fold
        tr_texts = train_df["text"].tolist()
        tr_y = train_df["label_int"].to_numpy()

        X_tr, X_val, y_tr, y_val = train_test_split(
            tr_texts, tr_y,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=tr_y,
        )

        model, info = train_one_fold(
            train_texts=X_tr,
            train_labels=y_tr,
            val_texts=X_val,
            val_labels=y_val,
            cfg=cfg,
            seed=args.seed,
            device=device,
        )

        # Evaluate A
        yA = testA_df["label_int"].to_numpy()
        predA = predict(model, testA_df["text"].tolist(), max_len=args.max_len, device=device)
        resA = eval_binary(yA, predA)

        # Evaluate B
        yB = testB_df["label_int"].to_numpy()
        predB = predict(model, testB_df["text"].tolist(), max_len=args.max_len, device=device)
        resB = eval_binary(yB, predB)

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
            "model_cfg": cfg,
            "train_info": info,
            "device": str(device),
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
        "model": "packed_attn_bilstm_v2",
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
        "aggregate": agg,
        "fold_results": folds_out,
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    plot_metric_bars(
        f1_A_list, f1_B_list,
        plots_dir / "f1_A_vs_B_v2.png",
        title="Packed+Attention BiLSTM F1 (mean ± std across folds)",
        ylabel="F1",
    )
    plot_metric_bars(
        r_A_list, r_B_list,
        plots_dir / "recall_A_vs_B_v2.png",
        title="Packed+Attention BiLSTM Recall (mean ± std across folds)",
        ylabel="Recall",
    )
    plot_deltas(
        dF1_list,
        plots_dir / "delta_f1_per_fold_v2.png",
        title="Packed+Attention BiLSTM ΔF1 per fold (A − B)",
        ylabel="ΔF1",
    )
    plot_deltas(
        dR_list,
        plots_dir / "delta_recall_per_fold_v2.png",
        title="Packed+Attention BiLSTM ΔRecall per fold (A − B)",
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