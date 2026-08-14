#!/usr/bin/env python
"""Genus-level KNN accuracy vs. k, one line per softmax temperature T, for
the temperature ablation (best config, softmax voting only). Same visual
style as plot_kcurve.py, extended to an arbitrary number of lines instead of
the fixed uniform/softmax/random trio, colored on a gradient from low T
(sharp, near winner-take-all) to high T (soft, near-uniform).

Usage:
    python plot_temperature_ablation.py --results-csv bioscan5m_temperature_ablation.csv \
        --out Figures/bioscan5m_temperature_ablation.pdf --default-t 0.07
"""

import argparse
import csv

import matplotlib.pyplot as plt

K_VALUES = [1, 3, 5, 7, 10, 15, 20, 25, 50]


def load_data(csv_path):
    """CSV columns: T,k1,k3,k5,k7,k10,k15,k20,k25,k50. The row with T="uniform"
    (non-numeric) is the uniform-voting curve (the T -> infinity limit of
    softmax voting), kept separate from the numeric temperature values."""
    data = {}
    uniform = None
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() == "t":
                continue
            if row[0].strip().lower() == "uniform":
                uniform = [float(v) for v in row[1:10]]
                continue
            T = float(row[0])
            data[T] = [float(v) for v in row[1:10]]
    return data, uniform


def plot(data, uniform, out_path, default_t):
    fig, ax = plt.subplots(figsize=(7, 5))
    temps = sorted(data.keys())
    cmap = plt.get_cmap("viridis")
    x = range(len(K_VALUES))

    for i, T in enumerate(temps):
        color = cmap(i / max(len(temps) - 1, 1))
        is_default = abs(T - default_t) < 1e-9
        ax.plot(
            x, data[T],
            marker="o", markersize=5,
            linestyle="-" if is_default else "--",
            linewidth=3 if is_default else 1.6,
            color="#D62728" if is_default else color,
            label=f"T={T:g}" + (" (default)" if is_default else ""),
            zorder=10 if is_default else 5,
        )

    if uniform is not None:
        ax.plot(
            x, uniform,
            marker="x", markersize=6,
            linestyle=":", linewidth=2,
            color="black",
            label=r"Uniform ($T\to\infty$ limit)",
            zorder=9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("$k$ (number of neighbors)")
    ax.set_ylabel("Genus-level accuracy (%)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  Wrote {out_path}")


def get_parser():
    p = argparse.ArgumentParser(description="Plot softmax temperature ablation k-curves.")
    p.add_argument("--results-csv", dest="results_csv", required=True,
                    help="CSV with columns T,k1,k3,k5,k7,k10,k15,k20,k25,k50")
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--default-t", dest="default_t", type=float, default=0.07)
    return p


def cli():
    args = get_parser().parse_args()
    data, uniform = load_data(args.results_csv)
    plot(data, uniform, args.out, args.default_t)


if __name__ == "__main__":
    cli()