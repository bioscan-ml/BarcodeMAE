#!/usr/bin/env python
"""Hierarchical taxonomic sunburst plots (phylum -> class -> order -> family
-> genus) for the KNN reference set and query (test) set of BIOSCAN-5M or
UNITE+INSD, for Appendix Section A ("Dataset taxonomic distribution").

BIOSCAN-5M reads directly from the supervised-split CSVs (reference = Seen
partition's training subset, query = Unseen partition, both already the
real evaluation sets -- no further filtering needed).

UNITE+INSD reads fasta files via mycoai's UNITE-header parser. The reference
set is the full training set. The query set is NOT the raw released test
fasta -- it is filtered through the same leakage pipeline as
its_overlap_clean.py (exact-sequence duplicates removed, then restricted to
novel-species-but-genus-in-train specimens), so the plotted query population
matches the 526 (Yeast) / 3,136 (Filamentous) genus-level query set actually
used for evaluation, not the noisier original release.

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


def make_sunburst(df, levels, out_path):
    present_levels = [lvl for lvl in levels if lvl in df.columns and df[lvl].notna().any()]
    counted = df.groupby(present_levels, dropna=False).size().reset_index(name="count")
    fig = px.sunburst(
        counted,
        path=present_levels,
        values="count",
    )
    fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
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
    make_sunburst(ref_df, LEVELS, os.path.join(out_dir, "taxdist_bioscan5m_reference.pdf"))
    make_sunburst(query_df, LEVELS, os.path.join(out_dir, "taxdist_bioscan5m_query.pdf"))


# --- UNITE+INSD ----------------------------------------------------------------

def load_its5m_full(fasta_path):
    """Full mycoai-parsed dataframe (id, taxonomy levels, species, sequence),
    needed for leakage filtering -- not yet reduced to just the plot columns."""
    from mycoai.data import Data
    allow_duplicates = "train" in os.path.basename(fasta_path)
    return Data(fasta_path, allow_duplicates=allow_duplicates).data


def taxonomy_columns(df):
    return df[[c for c in LEVELS if c in df.columns]]


def filter_genus_level_queries(train_df, test_df):
    """Mirrors its_overlap_clean.py's leakage pipeline: remove exact-sequence
    duplicates of the training set, then keep only specimens whose species is
    NOT in training at all but whose genus IS -- the well-posed genus-level
    query population (526 Yeast / 3,136 Filamentous), not the raw release."""
    from mycoai import utils as mycoai_utils

    train_seqs_all = set(train_df["sequence"])
    train_known = train_df[train_df["species"] != mycoai_utils.UNKNOWN_STR]
    train_species = set(zip(train_known["genus"], train_known["species"]))
    train_genera = set(train_df[train_df["genus"] != mycoai_utils.UNKNOWN_STR]["genus"])

    is_exact = test_df["sequence"].isin(train_seqs_all)
    clean_df = test_df[~is_exact]

    is_known_species = clean_df["species"] != mycoai_utils.UNKNOWN_STR
    clean_known = clean_df[is_known_species]
    species_keys = list(zip(clean_known["genus"], clean_known["species"]))
    is_novel_species = pd.Series([k not in train_species for k in species_keys],
                                  index=clean_known.index)
    is_known_genus = clean_known["genus"] != mycoai_utils.UNKNOWN_STR
    is_genus_in_train = clean_known["genus"].isin(train_genera)

    return clean_known[is_novel_species & is_known_genus & is_genus_in_train]


def run_its5m(data_dir, out_dir, query_fasta, query_name):
    train_df = load_its5m_full(os.path.join(data_dir, "trainset.fasta"))
    make_sunburst(taxonomy_columns(train_df), LEVELS,
                  os.path.join(out_dir, "taxdist_its5m_reference.pdf"))

    raw_query_df = load_its5m_full(os.path.join(data_dir, query_fasta))
    query_df = filter_genus_level_queries(train_df, raw_query_df)
    print(f"  {query_name}: {len(raw_query_df)} raw specimens -> {len(query_df)} leakage-free "
          f"genus-level query specimens")
    make_sunburst(taxonomy_columns(query_df), LEVELS,
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