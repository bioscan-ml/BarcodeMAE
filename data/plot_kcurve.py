#!/usr/bin/env python
"""Genus-level KNN accuracy vs. k, uniform vs. softmax voting vs. random-init
baseline, for the best BIOSCAN-5M configuration (encoder-decoder, +CLS+CE)
and the best UNITE+INSD configuration (encoder-decoder, +CLS+Binary) on both
test sets. Recreates main_kcurve_uniform_vs_softmax.pdf / its_kcurve_uniform_
vs_softmax.pdf in the same visual style (colors/markers/legend), but without
the shaded fill between the uniform and softmax lines.

The softmax line is read from the per-temperature ablation CSVs (see
plot_temperature_ablation.py) at --softmax-t, defaulting to T=0.02 -- the
value adopted as the main-text default after the sweep in Appendix Section H
showed it beats DINOv2's T=0.07 default on both peak and mean accuracy, on
both barcode domains. Uniform and random-init lines still come from the
original bioscan5m/its5m_uniform_vs_softmax_genus.csv files (voting rule
without a temperature, so unaffected by which T is adopted).

Usage:
    python plot_kcurve.py --bioscan5m-csv ./bioscan5m_uniform_vs_softmax_genus.csv \
        --its5m-csv ./its5m_uniform_vs_softmax_genus.csv \
        --bioscan5m-temp-csv ./temperature_ablation_bioscan5m.csv \
        --its5m-yeast-temp-csv ./temperature_ablation_its5m_yeast.csv \
        --its5m-filamentous-temp-csv ./temperature_ablation_its5m_filamentous.csv \
        --softmax-t 0.02 --out-dir ./Figures
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt

K_VALUES = [1, 3, 5, 7, 10, 15, 20, 25, 50]
UNIFORM_COLOR = "#3B7EA8"
SOFTMAX_COLOR = "#7B3FA0"
RANDOM_COLOR = "#999999"


def load_row(csv_path, match_fn):
    with open(csv_path) as f:
        for row in csv.reader(f):
            if match_fn(row):
                return row
    raise ValueError(f"No matching row found in {csv_path}")


def load_temp_row(csv_path, T):
    """Read the softmax k1..k50 values for temperature T from a
    temperature_ablation_*.csv (columns: T,k1,k3,k5,k7,k10,k15,k20,k25,k50)."""
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row or row[0].strip().lower() in ("t", "uniform"):
                continue
            if abs(float(row[0]) - T) < 1e-9:
                return [float(v) for v in row[1:10]]
    raise ValueError(f"T={T} not found in {csv_path}")


def plot_panel(ax, uniform_vals, softmax_vals, random_vals, softmax_t, title=None, show_legend=False):
    x = range(len(K_VALUES))
    ax.plot(x, uniform_vals, marker="o", linestyle="--", color=UNIFORM_COLOR,
             label="Uniform (hard vote)", linewidth=2, markersize=7)
    ax.plot(x, softmax_vals, marker="s", linestyle="-", color=SOFTMAX_COLOR,
             label=f"Softmax (T={softmax_t:g})", linewidth=2.5, markersize=7)
    ax.plot(x, random_vals, marker="D", linestyle=":", color=RANDOM_COLOR,
             label="Random init (untrained, softmax)", linewidth=1.5, markersize=6)

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("$k$ (number of neighbors)")
    ax.set_ylabel("Genus-level accuracy (%)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontweight="bold")
    if show_legend:
        ax.legend(loc="lower left", frameon=False)


def run_bioscan5m(csv_path, temp_csv_path, softmax_t, out_dir):
    row = load_row(csv_path, lambda r: r[:3] == ["enc-dec", "+CLS+CE", "CLS"])
    uniform_vals = [float(v) for v in row[3:12]]
    softmax_vals = load_temp_row(temp_csv_path, softmax_t)
    random_row = load_row(csv_path, lambda r: r[:3] == ["RANDOM", "Random init", "CLS"])
    random_vals = [float(v) for v in random_row[12:21]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_panel(ax, uniform_vals, softmax_vals, random_vals, softmax_t, show_legend=True)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "main_kcurve_uniform_vs_softmax.pdf")
    fig.savefig(out_path)
    print(f"  Wrote {out_path}")


def run_its5m(csv_path, yeast_temp_csv, filamentous_temp_csv, softmax_t, out_dir):
    temp_csvs = {"Yeast": yeast_temp_csv, "Filamentous": filamentous_temp_csv}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, test_name, title in zip(axes, ["Yeast", "Filamentous"], ["Yeast", "Filamentous"]):
        row = load_row(csv_path, lambda r: r[:4] == ["enc-dec", "+CLS+Binary", "CLS", test_name])
        uniform_vals = [float(v) for v in row[4:13]]
        softmax_vals = load_temp_row(temp_csvs[test_name], softmax_t)
        random_row = load_row(csv_path, lambda r: r[:4] == ["RANDOM", "Random init", "CLS", test_name])
        random_vals = [float(v) for v in random_row[13:22]]
        plot_panel(ax, uniform_vals, softmax_vals, random_vals, softmax_t, title=title,
                   show_legend=(test_name == "Yeast"))
    fig.tight_layout()
    out_path = os.path.join(out_dir, "its_kcurve_uniform_vs_softmax.pdf")
    fig.savefig(out_path)
    print(f"  Wrote {out_path}")


def get_parser():
    p = argparse.ArgumentParser(description="Regenerate the k-curve uniform-vs-softmax figures.")
    p.add_argument("--bioscan5m-csv", dest="bioscan5m_csv", required=True)
    p.add_argument("--its5m-csv", dest="its5m_csv", required=True)
    p.add_argument("--bioscan5m-temp-csv", dest="bioscan5m_temp_csv", required=True,
                    help="temperature_ablation_bioscan5m.csv")
    p.add_argument("--its5m-yeast-temp-csv", dest="its5m_yeast_temp_csv", required=True,
                    help="temperature_ablation_its5m_yeast.csv")
    p.add_argument("--its5m-filamentous-temp-csv", dest="its5m_filamentous_temp_csv", required=True,
                    help="temperature_ablation_its5m_filamentous.csv")
    p.add_argument("--softmax-t", dest="softmax_t", type=float, default=0.02)
    p.add_argument("--out-dir", dest="out_dir", default=".")
    return p


def cli():
    args = get_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_bioscan5m(args.bioscan5m_csv, args.bioscan5m_temp_csv, args.softmax_t, args.out_dir)
    run_its5m(args.its5m_csv, args.its5m_yeast_temp_csv, args.its5m_filamentous_temp_csv,
              args.softmax_t, args.out_dir)


if __name__ == "__main__":
    cli()