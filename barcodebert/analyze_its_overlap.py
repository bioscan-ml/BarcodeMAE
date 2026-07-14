#!/usr/bin/env python
"""Verify the ITS-5M train/test overlap numbers (species overlap and identical
barcode overlap) and break down WHY species overlap can be much higher than
identical-barcode overlap.

Motivation: Table 6 in the paper reports, for each of the 3 fungi test sets,
how many of its species/barcodes are also present in the training set. Test
Set 2 (Filamentous Fungi) has much higher species overlap (47.86%) than
identical-barcode overlap (6.48%) — i.e. many test specimens belong to a
species the model also saw in training, but as a DIFFERENT individual
specimen with a different exact ITS sequence (normal intraspecific sequence
variation). Test Sets 1 (Yeast) and 3 (MycoAI) instead have high overlap on
BOTH axes (86.73% / 100.00% identical barcodes), which looks like train/test
leakage of literal duplicate specimens rather than natural species overlap.

This script uses mycoai.data.Data for fasta parsing — the SAME loader
DNADataset uses for ITS-5M (barcodebert/datasets.py) — rather than a naive
line-by-line fasta parser, because mycoai's Data(..., allow_duplicates=False)
(used for "test" files) drops rows that are duplicates on the (sequence,
species) pair, parsed from the fasta headers. A plain re-parse of the raw
fasta file does NOT reproduce this and will over-count (raises a mismatch
error against *_labels.csv, which was generated against the deduplicated
count). Using mycoai directly guarantees this script sees exactly what the
training/eval pipeline sees, with no risk of reimplementing the header
parsing/dedup logic slightly wrong.

Usage:
    python analyze_its_overlap.py --data-dir ./BarcodeMAE/data/ITS-5M
"""

import argparse
import os

import pandas as pd
from mycoai.data import Data

TEST_SETS = [
    ("Test1 (Yeast)", "test1.fasta"),
    ("Test2 (Filamentous)", "test2.fasta"),
    ("Test3 (MycoAI)", "test3.fasta"),
]
UNKNOWN_LABEL = 9999999


def load_split(fasta_path, taxonomic_level):
    labels_path = fasta_path.replace(".fasta", "_labels.csv")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"Fasta file not found: {fasta_path}")

    # Mirrors datasets.py's DNADataset ITS-5M loading exactly.
    allow_duplicates = "train" in os.path.basename(fasta_path)
    fungi_data = Data(fasta_path, allow_duplicates=allow_duplicates)
    sequences = fungi_data.data["sequence"].tolist()

    labels_df = pd.read_csv(labels_path)

    if len(sequences) != len(labels_df):
        raise ValueError(
            f"Mismatch between mycoai-loaded fasta records ({len(sequences)}, "
            f"allow_duplicates={allow_duplicates}) and label rows ({len(labels_df)}) "
            f"for {fasta_path} — is this the right labels file?"
        )
    if taxonomic_level not in labels_df.columns:
        raise KeyError(
            f"Column '{taxonomic_level}' not found in {labels_path}. "
            f"Available columns: {list(labels_df.columns)}"
        )

    df = labels_df.copy()
    df["sequence"] = sequences
    return df


def compute_overlap(train_df, test_df, taxonomic_level):
    train_species = set(train_df.loc[train_df[taxonomic_level] != UNKNOWN_LABEL, taxonomic_level])
    train_seqs = set(train_df["sequence"])

    test_known = test_df[test_df[taxonomic_level] != UNKNOWN_LABEL]
    unique_test_species = set(test_known[taxonomic_level].unique())
    species_overlap = unique_test_species & train_species

    is_barcode_dup = test_df["sequence"].isin(train_seqs)
    n_barcode_overlap = int(is_barcode_dup.sum())

    # Same-species-different-barcode breakdown ---------------------------------
    is_shared_species = test_df[taxonomic_level].isin(species_overlap)
    n_shared_species_specimens = int(is_shared_species.sum())
    n_shared_species_and_dup_barcode = int((is_shared_species & is_barcode_dup).sum())
    n_shared_species_novel_barcode = n_shared_species_specimens - n_shared_species_and_dup_barcode

    return {
        "species_total": len(unique_test_species),
        "species_overlap_n": len(species_overlap),
        "species_overlap_pct": 100.0 * len(species_overlap) / len(unique_test_species) if unique_test_species else float("nan"),
        "barcode_total": len(test_df),
        "barcode_overlap_n": n_barcode_overlap,
        "barcode_overlap_pct": 100.0 * n_barcode_overlap / len(test_df) if len(test_df) else float("nan"),
        "shared_species_specimens": n_shared_species_specimens,
        "shared_species_dup_barcode": n_shared_species_and_dup_barcode,
        "shared_species_novel_barcode": n_shared_species_novel_barcode,
    }


def run(data_dir, taxonomic_level="species"):
    train_path = os.path.join(data_dir, "trainset.fasta")
    print(f"Loading train set: {train_path}")
    train_df = load_split(train_path, taxonomic_level)
    print(f"  {len(train_df)} train specimens, "
          f"{train_df.loc[train_df[taxonomic_level] != UNKNOWN_LABEL, taxonomic_level].nunique()} known species\n")

    rows = []
    for name, fname in TEST_SETS:
        fpath = os.path.join(data_dir, fname)
        print(f"Loading {name}: {fpath}")
        test_df = load_split(fpath, taxonomic_level)
        stats = compute_overlap(train_df, test_df, taxonomic_level)
        rows.append((name, stats))

        print(f"  Species overlap:  {stats['species_overlap_n']:>6d} / {stats['species_total']:<6d} "
              f"({stats['species_overlap_pct']:6.2f}%)")
        print(f"  Barcode overlap:  {stats['barcode_overlap_n']:>6d} / {stats['barcode_total']:<6d} "
              f"({stats['barcode_overlap_pct']:6.2f}%)")
        print(f"  Of the {stats['shared_species_specimens']} test specimens whose species IS in training:")
        print(f"    - {stats['shared_species_dup_barcode']} are EXACT duplicate barcodes (leakage)")
        print(f"    - {stats['shared_species_novel_barcode']} are a DIFFERENT individual of the same species "
              f"(same species, different barcode)")
        print()

    print("=" * 100)
    print(f"{'Test Set':<22}{'SpeciesOverlap':>16}{'SpeciesPct':>12}{'BarcodeOverlap':>16}{'BarcodePct':>12}")
    for name, stats in rows:
        print(f"{name:<22}"
              f"{stats['species_overlap_n']:>10d}/{stats['species_total']:<5d}"
              f"{stats['species_overlap_pct']:>11.2f}%"
              f"{stats['barcode_overlap_n']:>10d}/{stats['barcode_total']:<5d}"
              f"{stats['barcode_overlap_pct']:>11.2f}%")
    print("=" * 100)
    print("\nCompare the 'SpeciesPct'/'BarcodePct' columns above against Table 6:")
    print("  Test1 (Yeast):       species 53.24% | barcode 86.73%")
    print("  Test2 (Filamentous): species 47.86% | barcode  6.48%")
    print("  Test3 (MycoAI):      species 100.00%| barcode 100.00%")


def get_parser():
    p = argparse.ArgumentParser(description="Verify ITS-5M train/test overlap numbers (Table 6).")
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="Path to ITS-5M data directory (containing trainset.fasta, test1-3.fasta + *_labels.csv).")
    p.add_argument("--taxonomic-level", "--taxonomic_level", dest="taxonomic_level", default="species",
                    help="Label column to use for species-level overlap. Default: %(default)s")
    return p


def cli():
    args = get_parser().parse_args()
    run(args.data_dir, args.taxonomic_level)


if __name__ == "__main__":
    cli()