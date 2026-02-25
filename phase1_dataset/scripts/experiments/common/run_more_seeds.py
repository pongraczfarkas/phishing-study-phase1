from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.common.plots import ensure_dir, plot_seed_sweep


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()

    # What to run
    ap.add_argument("--script", required=True,
                    help="Path to a train/eval script, e.g. experiments/lr_tfidf/train_eval_lr.py")
    ap.add_argument("--dataset", default="final/phase1_dataset.jsonl")
    ap.add_argument("--folds", type=int, default=5)

    # Seeds
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10)

    # Where to save
    ap.add_argument("--outdir", default="experiments/seed_sweeps/lr_tfidf")
    ap.add_argument("--python", default="py", help="Python launcher (Windows: py)")

    # Where to find metrics in the output JSON
    ap.add_argument("--metric-path", default="aggregate.delta_f1.mean",
                    help="Dot-path into the metrics JSON, e.g. aggregate.delta_f1.mean")
    ap.add_argument("--metric2-path", default="aggregate.delta_recall.mean",
                    help="Second dot-path into JSON")
    ap.add_argument("--label", default="ΔF1 (A − B)")
    ap.add_argument("--label2", default="ΔRecall (A − B)")

    args = ap.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}. Run from scripts/ folder.")

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    seeds = list(range(args.start, args.end + 1))
    values1: list[float] = []
    values2: list[float] = []
    rows = []

    def get_by_dotpath(obj: dict, dot: str):
        cur = obj
        for part in dot.split("."):
            if part not in cur:
                raise KeyError(f"Missing key '{part}' while resolving '{dot}'")
            cur = cur[part]
        return cur

    for s in seeds:
        metrics_path = outdir / f"metrics_seed_{s}.json"
        plots_dir = outdir / f"plots_seed_{s}"
        ensure_dir(plots_dir)

        cmd = [
            args.python,
            str(script_path),
            "--dataset", args.dataset,
            "--seed", str(s),
            "--folds", str(args.folds),
            "--out", str(metrics_path),
            "--plots-dir", str(plots_dir),
        ]

        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        m = read_json(metrics_path)

        v1 = float(get_by_dotpath(m, args.metric_path))
        v2 = float(get_by_dotpath(m, args.metric2_path))
        values1.append(v1)
        values2.append(v2)

        rows.append({
            "seed": s,
            args.metric_path: v1,
            args.metric2_path: v2,
        })

        print(f"Seed {s}: {args.label}={v1:.6f}, {args.label2}={v2:.6f}")

    # Save summary
    summary = {
        "script": str(script_path),
        "dataset": args.dataset,
        "folds": args.folds,
        "seeds": seeds,
        "metric_path": args.metric_path,
        "metric2_path": args.metric2_path,
        "rows": rows,
    }
    (outdir / "seed_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved summary:", (outdir / "seed_sweep_summary.json").resolve())

    # Plots across seeds
    plot_seed_sweep(
        seeds, values1,
        outdir / "metric1_across_seeds.png",
        title=f"{args.label} across seeds",
        ylabel=args.label,
    )
    plot_seed_sweep(
        seeds, values2,
        outdir / "metric2_across_seeds.png",
        title=f"{args.label2} across seeds",
        ylabel=args.label2,
    )
    print("Saved sweep plots to:", outdir.resolve())


if __name__ == "__main__":
    main()