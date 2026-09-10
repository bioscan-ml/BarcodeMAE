#!/usr/bin/env python
"""
Species-level label distribution analysis across ITS-5M splits.

Usage:
    python check_label_distribution.py --data-dir <path/to/ITS-5M>
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

UNLABELED = 9999999
COL = "species"

# Fixed phylum colour palette (matches reference figure)
PHYLUM_COLORS = {
    "Basidiomycota":     "#5472d3",
    "Ascomycota":        "#c0392b",
    "Mortierellomycota": "#27ae60",
    "Mucoromycota":      "#8e44ad",
    "Chytridiomycota":   "#f39c12",
    "Rozellomycota":     "#e91e8c",
    "Zoopagomycota":     "#8bc34a",
    "?":                 "#e67e22",
}
_FALLBACK_COLORS = [
    "#1abc9c", "#3498db", "#9b59b6", "#e74c3c", "#f1c40f",
    "#2ecc71", "#e67e22", "#95a5a6", "#34495e", "#d35400",
]


def load(path):
    df = pd.read_csv(path)
    before = len(df)
    df = df[df[COL] != UNLABELED].copy()
    print(f"  {os.path.basename(path):<35}  {len(df):>7} labeled  ({before - len(df):>7} unlabeled removed)")
    return df


def plot_train_vs_test(ref_counts, test_counts_dict, out_path):
    """
    One subplot per test split.

    Each point = one species shared between train+valid and that test split.
      x-axis : normalised frequency in train+valid  (log)
      y-axis : normalised frequency in that test split  (log)

    Diagonal (y = x) = perfectly matched representation.
    Below diagonal   = species over-represented in train, under-represented in test.
    Above diagonal   = species under-represented in train, over-represented in test.

    Colour encodes log2(train_freq / test_freq):
      blue  → more common in train than test
      red   → more common in test than train
      white → roughly equal

    Marginal bars on the right show species present ONLY in that test split (unseen).
    """
    n = len(test_counts_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    ref_prob = ref_counts / ref_counts.sum()

    cmap = plt.cm.RdBu              # blue = more in train, red = more in test
    norm = mcolors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)

    for ax, (name, tc) in zip(axes, test_counts_dict.items()):
        test_prob = tc / tc.sum()

        shared = sorted(set(ref_prob.index) & set(test_prob.index))
        unseen_in_train = sorted(set(test_prob.index) - set(ref_prob.index))

        if not shared:
            ax.set_title(f"{name}\n(no shared species)")
            continue

        x = np.array([ref_prob[s]  for s in shared])
        y = np.array([test_prob[s] for s in shared])
        log_ratio = np.log2(x / y)   # positive = more in train; negative = more in test

        sc = ax.scatter(x, y, c=log_ratio, cmap=cmap, norm=norm,
                        alpha=0.7, s=18, linewidths=0)

        # diagonal reference line
        lo = min(x.min(), y.min()) * 0.8
        hi = max(x.max(), y.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="equal representation")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Normalised frequency in train+valid (log)", fontsize=9)
        ax.set_ylabel(f"Normalised frequency in {name} (log)", fontsize=9)

        n_shared  = len(shared)
        n_unseen  = len(unseen_in_train)
        n_total   = len(test_prob)
        ax.set_title(
            f"{name}\n"
            f"{n_shared}/{n_total} species shared with train  |  {n_unseen} unseen",
            fontsize=9,
        )

        cb = plt.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label("log₂(train freq / test freq)\nblue = more in train", fontsize=7)
        cb.ax.tick_params(labelsize=7)

        ax.legend(fontsize=7)

        # Annotate quadrant counts in corners
        # n_below: log_ratio > 1 → train >> test → high x, low y → bottom-right
        # n_above: log_ratio < -1 → test >> train → low x, high y → top-left
        n_below = int((log_ratio > 1).sum())
        n_above = int((log_ratio < -1).sum())
        ax.text(0.97, 0.02, f"over-represented\nin train: {n_below}",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom", color="#2166ac")
        ax.text(0.02, 0.97, f"over-represented\nin test: {n_above}",
                transform=ax.transAxes, fontsize=7, ha="left", va="top", color="#d6604d")

    plt.suptitle("Train+Valid vs Test — species frequency comparison\n"
                 "(shared species only; points below diagonal = over-represented in train)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Scatter plot saved → {out_path}")


def plot_sorted_histogram(ref_counts, test_counts_dict, out_path):
    """
    One subplot per test split.
    Species sorted by training frequency (most → rarest on x-axis), log y-axis.
    Train line shown in black for reference in every panel.
    Gaps = species absent from that test set.
    """
    sorted_species = ref_counts.sort_values(ascending=False).index
    x = np.arange(len(sorted_species))
    train_y = ref_counts[sorted_species].values.astype(float)

    n = len(test_counts_dict)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, (name, tc), color in zip(axes, test_counts_dict.items(), colors):
        test_y = np.array([tc.get(s, 0) for s in sorted_species], dtype=float)
        test_y[test_y == 0] = np.nan  # gaps where the test set has no samples

        ax.plot(x, train_y, color="black", linewidth=1.2, label="train+valid", zorder=3)
        ax.plot(x, test_y,  color=color,   linewidth=1.0, alpha=0.85, label=name)

        ax.set_yscale("log")
        ax.set_xlabel("Species rank (most common → rarest in train)", fontsize=9)
        ax.set_ylabel("Sample count (log)", fontsize=9)
        ax.set_title(f"train+valid  vs  {name}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)

    plt.suptitle("Sorted species frequency — x-axis ordered by train rank\n"
                 "gaps = species absent from that test set", fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Sorted histogram saved → {out_path}")


def plot_frequency_ratio(ref_counts, test_counts_dict, out_path):
    """
    For each shared species, plot test_count / train_count (both normalised to frequencies).
    A flat horizontal line = perfectly proportional split.
    Deviations show which species are over/under-represented in each test set.
    """
    n = len(test_counts_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    ref_freq = ref_counts / ref_counts.sum()

    for ax, (name, tc) in zip(axes, test_counts_dict.items()):
        test_freq = tc / tc.sum()

        shared = sorted(set(ref_freq.index) & set(test_freq.index),
                        key=lambda s: -ref_freq[s])  # sort by train frequency

        ratios = np.array([test_freq[s] / ref_freq[s] for s in shared])
        x = np.arange(len(shared))

        # Colour: ratio > 1 = more in test (red), ratio < 1 = more in train (blue)
        colors = ["#d6604d" if r > 1 else "#4393c3" for r in ratios]
        ax.bar(x, ratios, color=colors, width=1.0, linewidth=0)

        # Ideal flat line
        ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="equal (ratio = 1)")

        ax.set_yscale("log")
        ax.set_xlabel("Species (sorted by train frequency, most common → rarest)", fontsize=8)
        ax.set_ylabel("test freq / train freq  (log)", fontsize=8)
        ax.set_title(f"{name}  —  {len(shared)} shared species\n"
                     f"red = more in test, blue = more in train", fontsize=9)
        ax.set_xticks([])
        ax.legend(fontsize=8)
        ax.axhline(2.0, color="#d6604d", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.axhline(0.5, color="#4393c3", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.text(len(shared) * 0.98, 2.1, "2×", fontsize=7, color="#d6604d", ha="right")
        ax.text(len(shared) * 0.98, 0.55, "0.5×", fontsize=7, color="#4393c3", ha="right")

    plt.suptitle("Frequency ratio: test / train  (flat line = perfect match)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Ratio plot saved       → {out_path}")


def _build_color_map(dfs, phylum_col):
    """Build a consistent phylum → colour mapping across all splits."""
    all_phyla = set()
    for df in dfs.values():
        if phylum_col in df.columns:
            all_phyla.update(df[phylum_col].astype(str).unique())
    color_map = {}
    fb_idx = 0
    for p in sorted(all_phyla):
        if p in PHYLUM_COLORS:
            color_map[p] = PHYLUM_COLORS[p]
        else:
            color_map[p] = _FALLBACK_COLORS[fb_idx % len(_FALLBACK_COLORS)]
            fb_idx += 1
    color_map["(missing)"] = "#aaaaaa"
    return color_map


def _sunburst_trace_data(df, levels, color_map):
    """
    Recursively build (ids, labels, parents, values, colors) for go.Sunburst.
    Inner-most level gets phylum colors; all descendants inherit their phylum's color.
    """
    ids, labels, parents, vals, colors = [], [], [], [], []

    def recurse(sub, lvl_idx, parent_id, phylum_color):
        if lvl_idx >= len(levels):
            return
        lev = levels[lvl_idx]
        if lev not in sub.columns:
            return
        counts = sub.groupby(lev, sort=False).size().sort_values(ascending=False)
        for val_raw, cnt in counts.items():
            val = str(val_raw)
            node_id = f"{parent_id}/{val}" if parent_id else val
            c = color_map.get(val, "#888888") if lvl_idx == 0 else phylum_color
            ids.append(node_id)
            labels.append(val)
            parents.append(parent_id)
            vals.append(int(cnt))
            colors.append(c)
            recurse(sub[sub[lev] == val_raw], lvl_idx + 1, node_id,
                    c if lvl_idx == 0 else phylum_color)

    recurse(df, 0, "", "")
    return ids, labels, parents, vals, colors


def plot_sunburst(dfs, levels, out_path):
    """
    Multi-panel sunburst (one per split) using Plotly, styled after the reference figure.

    Rings from inside out follow the taxonomy in *levels* (e.g. phylum → class → order →
    family → genus).  The innermost ring is coloured by phylum with a fixed palette;
    all descendant rings inherit their phylum's colour.

    Saves an interactive HTML file always, and a PNG if kaleido is installed:
        pip install kaleido
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError:
        print("Plotly not installed — run: pip install plotly")
        return

    splits = list(dfs.keys())
    n = len(splits)
    if n == 0:
        return

    phylum_col = levels[0]
    color_map = _build_color_map(dfs, phylum_col)

    # Grid: at most 3 columns
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    # Subplot titles (padded to fill the grid)
    titles = [f"{nm.replace('_', ' ').title()}  ({len(dfs[nm]):,} samples)" for nm in splits]
    titles += [""] * (nrows * ncols - n)

    specs = [[{"type": "domain"} for _ in range(ncols)] for _ in range(nrows)]
    fig = make_subplots(rows=nrows, cols=ncols, specs=specs, subplot_titles=titles)

    for idx, (name, df) in enumerate(dfs.items()):
        row = idx // ncols + 1
        col = idx % ncols + 1

        # Normalise: convert UNLABELED sentinel → "?" string
        sub = df.copy()
        for lev in levels:
            if lev in sub.columns:
                sub[lev] = sub[lev].astype(str)
                sub.loc[sub[lev] == str(UNLABELED), lev] = "?"

        ids, lbls, par, vals, clrs = _sunburst_trace_data(sub, levels, color_map)
        if not ids:
            continue

        fig.add_trace(
            go.Sunburst(
                ids=ids,
                labels=lbls,
                parents=par,
                values=vals,
                marker=dict(colors=clrs, line=dict(color="white", width=0.5)),
                branchvalues="total",
                textinfo="label",
                insidetextorientation="radial",
                maxdepth=len(levels),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value:,}<br>"
                    "%{percentRoot:.1%} of total<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

    # Shared phylum legend (invisible scatter markers)
    shown_phyla = set()
    for df in dfs.values():
        if phylum_col in df.columns:
            shown_phyla.update(df[phylum_col].astype(str).unique())
    for phylum in sorted(shown_phyla):
        if phylum == "(missing)":
            continue
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color_map.get(phylum, "#888888"), symbol="square"),
                name=phylum,
                showlegend=True,
            )
        )

    cell_px = 560
    fig.update_layout(
        title=dict(
            text=(
                "Taxonomic Distribution  ·  "
                + "  →  ".join(levels)
            ),
            x=0.5,
            font=dict(size=16),
        ),
        height=cell_px * nrows + 80,
        width=cell_px * ncols + 220,
        legend=dict(
            title=dict(text=phylum_col.title(), font=dict(size=13)),
            x=1.01, y=0.5,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            itemsizing="constant",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    html_path = out_path.replace(".png", ".html")
    fig.write_html(html_path)
    print(f"Sunburst (interactive)  → {html_path}")

    try:
        pio.write_image(fig, out_path, scale=2)
        print(f"Sunburst (PNG)          → {out_path}")
    except Exception as exc:
        print(f"PNG export skipped ({exc}).")
        print("  Install kaleido for PNG:  pip install kaleido")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Path to ITS-5M data folder")
    parser.add_argument("--sunburst-levels", default="phylum,class,order,family,genus",
                        help="Comma-separated taxonomic levels for sunburst rings (inside → out). "
                             "First level sets the colour palette. "
                             "(default: phylum,class,order,family,genus)")
    args = parser.parse_args()

    d = args.data_dir
    files = {
        "train": os.path.join(d, "trainset_labels.csv"),
        "valid": os.path.join(d, "trainset_valid_labels.csv"),
        "test1": os.path.join(d, "test1_labels.csv"),
        "test2": os.path.join(d, "test2_labels.csv"),
        "test3": os.path.join(d, "test3_labels.csv"),
    }

    print("Loading files (removing unlabeled rows where species == 9999999):\n")
    dfs = {name: load(path) for name, path in files.items() if os.path.exists(path)}

    # Combined train + valid counts as reference
    ref_counts = dfs["train"][COL].value_counts() if "train" in dfs else pd.Series(dtype=int)
    if "valid" in dfs:
        ref_counts = ref_counts.add(dfs["valid"][COL].value_counts(), fill_value=0).astype(int)

    test_counts = {
        name: dfs[name][COL].value_counts()
        for name in dfs
        if name not in ("train", "valid")
    }

    if not test_counts:
        print("No test files found.")
        return

    # Print a quick summary table
    print(f"\nTrain+Valid: {len(ref_counts)} unique species, {ref_counts.sum()} samples")
    print(f"\n{'Split':<10} {'Total sp.':>10} {'Shared w/ train':>17} {'Unseen':>10} {'% shared':>10}")
    print("-" * 60)
    for name, tc in test_counts.items():
        shared = len(set(tc.index) & set(ref_counts.index))
        unseen = len(set(tc.index) - set(ref_counts.index))
        print(f"{name:<10} {len(tc):>10} {shared:>17} {unseen:>10} {100*shared/len(tc):>9.1f}%")
    print("-" * 60)

    plot_train_vs_test(ref_counts, test_counts,
                       os.path.join(d, "species_train_vs_test.png"))
    plot_sorted_histogram(ref_counts, test_counts,
                          os.path.join(d, "species_sorted_histogram.png"))
    plot_frequency_ratio(ref_counts, test_counts,
                         os.path.join(d, "species_frequency_ratio.png"))

    levels = [s.strip() for s in args.sunburst_levels.split(",")]
    sunburst_stem = "_".join(levels)
    plot_sunburst(dfs, levels, os.path.join(d, f"{sunburst_stem}_sunburst.png"))


if __name__ == "__main__":
    main()