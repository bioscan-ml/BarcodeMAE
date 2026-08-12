#!/usr/bin/env python
"""Hierarchical taxonomic sunburst plots (phylum -> class -> order -> family
-> genus) for the KNN reference set and query (test) set of BIOSCAN-5M or
UNITE+INSD, for Appendix Section A ("Dataset taxonomic distribution").

BIOSCAN-5M reads directly from the supervised-split CSVs (reference = Seen
partition's training subset, query = Unseen partition). UNITE+INSD reads
fasta files via mycoai's UNITE-header parser (reference = training set,
query = a leakage-free task CSV produced by analyze_its_overlap.py /
its_overlap_clean.py, or a raw test fasta if no task CSV is given).

Usage:
    # BIOSCAN-5M (reads local CSVs directly)
    python plot_taxonomic_sunburst.py --dataset bioscan5m \
        --data-dir ./BarcodeMAE/data/BIOSCAN-5M --out-dir ./Figures

    # UNITE+INSD (reads fasta via mycoai; run where mycoai + the fasta files are)
    python plot_taxonomic_sunburst.py --dataset its5m \
        --data-dir ./BarcodeMAE/data/ITS-5M --out-dir ./Figures \
        --its-query-fasta test1.fasta --its-query-name yeast
"""

import argparse
import os

import pandas as pd
import plotly.express as px

LEVELS = ["phylum", "class", "order", "family", "genus"]


def make_sunburst(df, levels, title, out_path):
    present_levels = [lvl for lvl in levels if lvl in df.columns and df[lvl].notna().any()]
    counted = df.groupby(present_levels, dropna=False).size().reset_index(name="count")
    fig = px.sunburst(
        counted,
        path=present_levels,
        values="count",
        title=title,
    )
    fig.update_layout(margin=dict(t=60, l=0, r=0, b=0))
    fig.write_image(out_path, width=1000, height=1000, scale=2)
    print(f"  Wrote {out_path}  ({len(df)} specimens, {len(present_levels)} ranks: {present_levels})")


# --- BIOSCAN-5M --------------------------------------------------------------

def load_bioscan5m(csv_path):
    df = pd.read_csv(csv_path)
    return df.rename(columns={
        "order_name": "order",
        "family_name": "family",
        "genus_name": "genus",
    })[["phylum", "class", "order", "family", "genus"]]


def run_bioscan5m(data_dir, out_dir):
    ref_df = load_bioscan5m(os.path.join(data_dir, "supervised_train.csv"))
    query_df = load_bioscan5m(os.path.join(data_dir, "unseen.csv"))
    make_sunburst(ref_df, LEVELS, "BIOSCAN-5M: KNN reference set (Seen/train)",
                  os.path.join(out_dir, "taxdist_bioscan5m_reference.pdf"))
    make_sunburst(query_df, LEVELS, "BIOSCAN-5M: query set (Unseen)",
                  os.path.join(out_dir, "taxdist_bioscan5m_query.pdf"))


# --- UNITE+INSD ----------------------------------------------------------------

def load_its5m_fasta(fasta_path):
    from mycoai.data import Data
    allow_duplicates = "train" in os.path.basename(fasta_path)
    data = Data(fasta_path, allow_duplicates=allow_duplicates).data
    return data.rename(columns={"phylum": "phylum", "class": "class"})[
        [c for c in ["phylum", "class", "order", "family", "genus"] if c in data.columns]
    ]


def run_its5m(data_dir, out_dir, query_fasta, query_name):
    ref_df = load_its5m_fasta(os.path.join(data_dir, "trainset.fasta"))
    make_sunburst(ref_df, LEVELS, "UNITE+INSD: KNN reference set (train)",
                  os.path.join(out_dir, "taxdist_its5m_reference.pdf"))
    query_df = load_its5m_fasta(os.path.join(data_dir, query_fasta))
    make_sunburst(query_df, LEVELS, f"UNITE+INSD: query set ({query_name})",
                  os.path.join(out_dir, f"taxdist_its5m_query_{query_name}.pdf"))


def get_parser():
    p = argparse.ArgumentParser(description="Taxonomic sunburst plots for Appendix Section A.")
    p.add_argument("--dataset", choices=["bioscan5m", "its5m"], required=True)
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True)
    p.add_argument("--out-dir", "--out_dir", dest="out_dir", default=".")
    p.add_argument("--its-query-fasta", dest="its_query_fasta", default="test1.fasta",
                    help="[its5m only] query fasta filename, relative to --data-dir.")
    p.add_argument("--its-query-name", dest="its_query_name", default="yeast",
                    help="[its5m only] short name used in the output filename (e.g. yeast, filamentous).")
    return p


def cli():
    args = get_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset == "bioscan5m":
        run_bioscan5m(args.data_dir, args.out_dir)
    else:
        run_its5m(args.data_dir, args.out_dir, args.its_query_fasta, args.its_query_name)


if __name__ == "__main__":
    cli()