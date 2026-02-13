from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from plots import ensure_dir, plot_seed_sweep


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="final/phase1_dataset.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--start", type=int, default=1, help="Start seed (inclusive)")
    ap.add_argument("--end", type=int, default=10, help="End seed (inclusive)")
    ap.add_argument("--outdir", default="experiments/lr_tfidf/seed_sweep")
    ap.add_argument("--python", default="py", help="Python launcher (Windows: py)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    train_script = Path("experiments/lr_tfidf/train_eval_lr.py")
    if not train_script.exists():
        raise FileNotFoundError(f"Could not find {train_script}. Run from repo root.")

    seeds = list(range(args.start, args.end + 1))
    delta_f1_means = []
    delta_recall_means = []
    results_table = []

    for s in seeds:
        metrics_path = outdir / f"metrics_seed_{s}.json"
        plots_dir = outdir / f"plots_seed_{s}"
        ensure_dir(plots_dir)

        cmd = [
            args.python,
            str(train_script),
            "--dataset", args.dataset,
            "--seed", str(s),
            "--folds", str(args.folds),
            "--out", str(metrics_path),
            "--plots-dir", str(plots_dir),
        ]

        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        m = read_metrics(metrics_path)
        d_f1 = m["aggregate"]["delta_f1"]["mean"]
        d_r = m["aggregate"]["delta_recall"]["mean"]

        delta_f1_means.append(float(d_f1))
        delta_recall_means.append(float(d_r))

        results_table.append({
            "seed": s,
            "delta_f1_mean": float(d_f1),
            "delta_recall_mean": float(d_r),
            "A_f1_mean": float(m["aggregate"]["A_f1"]["mean"]),
            "B_f1_mean": float(m["aggregate"]["B_f1"]["mean"]),
        })

        print(f"Seed {s}: ΔF1_mean={d_f1:.6f}, ΔRecall_mean={d_r:.6f}")

    # Save summary JSON
    summary_path = outdir / "seed_sweep_summary.json"
    summary = {
        "dataset": args.dataset,
        "folds": args.folds,
        "seeds": seeds,
        "rows": results_table,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved summary:", summary_path.resolve())

    # Plot across seeds
    plot_seed_sweep(
        seeds, delta_f1_means,
        outdir / "delta_f1_across_seeds.png",
        title="ΔF1 across seeds (A − B)",
        ylabel="ΔF1 mean",
    )
    plot_seed_sweep(
        seeds, delta_recall_means,
        outdir / "delta_recall_across_seeds.png",
        title="ΔRecall across seeds (A − B)",
        ylabel="ΔRecall mean",
    )
    print("Saved sweep plots to:", outdir.resolve())


if __name__ == "__main__":
    main()
