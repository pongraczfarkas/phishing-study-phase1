from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_metric_bars(
    f1_a: list[float],
    f1_b: list[float],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    a = np.array(f1_a, dtype=float)
    b = np.array(f1_b, dtype=float)

    means = [float(a.mean()), float(b.mean())]
    stds = [
        float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        float(b.std(ddof=1)) if len(b) > 1 else 0.0,
    ]

    fig = plt.figure()
    ax = plt.gca()
    ax.bar(["A (original)", "B (paraphrase)"], means, yerr=stds)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_deltas(
    deltas: list[float],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    d = np.array(deltas, dtype=float)
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(range(1, len(d) + 1), d, marker="o")
    ax.axhline(0.0)
    ax.set_title(title)
    ax.set_xlabel("Fold")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_seed_sweep(
    seeds: list[int],
    values: list[float],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(seeds, values, marker="o")
    ax.axhline(0.0)
    ax.set_title(title)
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
