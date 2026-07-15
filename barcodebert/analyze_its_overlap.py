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

IMPORTANT — this does NOT use the *_labels.csv files. Their 'species' column
is a pre-built CLASSIFICATION label: a contiguous integer index factorized
against a fixed vocabulary (built from the training species), where any test
species NOT in that vocabulary collapses into a single '9999999' unknown
bucket, indistinguishable from every other novel species. Using it for an
overlap AUDIT is circular (every "known" index is trivially "in training" by
construction) and undercounts true species diversity. Instead this script
uses mycoai.data.Data's own per-file UNITE-header parsing (the same parser
DNADataset uses, tax_parser='unite' by default) to get the raw (genus,
species) taxonomy string pair fresh from each fasta file's headers — no
cross-file vocabulary, no collapsing of novel species. Species identity is
keyed on (genus, species) together, not species-epithet alone, since UNITE
species epithets can collide across unrelated genera.

Usage:
    python analyze_its_overlap.py --data-dir ./BarcodeMAE/data/ITS-5M
"""

import argparse
import os

import pandas as pd
from mycoai import utils as mycoai_utils
from mycoai.data import Data

TEST_SETS = [
    ("Test1 (Yeast)", "test1.fasta"),
    ("Test2 (Filamentous)", "test2.fasta"),
    ("Test3 (MycoAI)", "test3.fasta"),
]


def load_split(fasta_path):
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"Fasta file not found: {fasta_path}")
    # Mirrors datasets.py's DNADataset ITS-5M loading: allow_duplicates=True
    # for "train" files, False for "test" files. tax_parser defaults to
    # 'unite', so fungi_data.data already has genus/species parsed straight
    # from each header — no external labels file needed.
    allow_duplicates = "train" in os.path.basename(fasta_path)
    fungi_data = Data(fasta_path, allow_duplicates=allow_duplicates)
    return fungi_data.data


def species_key(df):
    """(genus, species) tuples, excluding rows where species is unresolved."""
    known = df[df["species"] != mycoai_utils.UNKNOWN_STR]
    return known, list(zip(known["genus"], known["species"]))


def compute_overlap(train_df, test_df):
    _, train_keys = species_key(train_df)
    train_species = set(train_keys)
    train_seqs = set(train_df["sequence"])

    _, test_keys = species_key(test_df)
    unique_test_species = set(test_keys)
    species_overlap = unique_test_species & train_species

    is_barcode_dup = test_df["sequence"].isin(train_seqs)
    n_barcode_overlap = int(is_barcode_dup.sum())

    # Same-species-different-barcode breakdown ---------------------------------
    test_species_col = list(zip(test_df["genus"], test_df["species"]))
    is_shared_species = pd.Series(
        [k in species_overlap for k in test_species_col], index=test_df.index
    )
    n_shared_species_specimens = int(is_shared_species.sum())
    n_shared_species_and_dup_barcode = int((is_shared_species & is_barcode_dup).sum())
    n_shared_species_novel_barcode = n_shared_species_specimens - n_shared_species_and_dup_barcode

    # Clean subset: specimens whose exact barcode is NOT in training (removes
    # anything literally seen during pretraining/KNN-gallery construction)
    # AND whose species label is resolved (drops '?'/unknown-species rows —
    # they can't be classified as "same species" or "novel species" at all,
    # so they shouldn't pad either the numerator or denominator here).
    # Species overlap recomputed on just this subset.
    clean_df = test_df[~is_barcode_dup]
    clean_known_df, clean_keys = species_key(clean_df)
    clean_unique_species = set(clean_keys)
    clean_species_overlap = clean_unique_species & train_species

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
        "clean_total": len(clean_known_df),
        "clean_species_total": len(clean_unique_species),
        "clean_species_overlap_n": len(clean_species_overlap),
        "clean_species_overlap_pct": (100.0 * len(clean_species_overlap) / len(clean_unique_species)
                                       if clean_unique_species else float("nan")),
    }


def sample_novel_barcode_examples(train_df, test_df, n=10, seed=0):
    """Sample up to n (species, test_seq, one_train_seq_of_same_species) examples
    from the "same species, different barcode" category, for eyeballing whether
    the sequences really are meaningfully different (not near-identical)."""
    _, train_keys = species_key(train_df)
    train_species = set(train_keys)
    train_seqs = set(train_df["sequence"])

    train_known_df, _ = species_key(train_df)
    # One representative train sequence per (genus, species) — cheap to build
    # (~14.7K groups), avoids storing all 5.2M sequences grouped.
    train_repr = train_known_df.groupby(["genus", "species"])["sequence"].first()

    test_species_col = list(zip(test_df["genus"], test_df["species"]))
    is_shared_species = pd.Series(
        [k in train_species for k in test_species_col], index=test_df.index
    )
    is_barcode_dup = test_df["sequence"].isin(train_seqs)
    candidates = test_df[is_shared_species & ~is_barcode_dup]

    sample = candidates.sample(n=min(n, len(candidates)), random_state=seed)
    examples = []
    for _, row in sample.iterrows():
        key = (row["genus"], row["species"])
        train_seq = train_repr.loc[key]
        test_seq = row["sequence"]
        same_len = len(train_seq) == len(test_seq)
        pct_identity = (
            100.0 * sum(a == b for a, b in zip(train_seq, test_seq)) / len(test_seq)
            if same_len and len(test_seq) > 0 else None
        )
        examples.append({
            "genus": row["genus"], "species": row["species"],
            "test_id": row["id"], "test_seq": test_seq, "test_len": len(test_seq),
            "train_seq": train_seq, "train_len": len(train_seq),
            "pct_identity": pct_identity,
        })
    return examples


def print_examples(examples, name):
    print(f"--- {name}: {len(examples)} sampled 'same species, different barcode' examples ---")
    for ex in examples:
        print(f"  {ex['genus']} {ex['species']}  (test id={ex['test_id']})")
        print(f"    test  ({ex['test_len']:>4d} bp): {ex['test_seq']}")
        print(f"    train ({ex['train_len']:>4d} bp): {ex['train_seq']}")
        if ex["pct_identity"] is not None:
            print(f"    same length — {ex['pct_identity']:.1f}% identical position-by-position")
        else:
            print(f"    different lengths ({ex['test_len']} vs {ex['train_len']} bp) — not directly comparable position-by-position")
        print()


def run(data_dir, show_examples=0):
    train_path = os.path.join(data_dir, "trainset.fasta")
    print(f"Loading train set: {train_path}")
    train_df = load_split(train_path)
    _, train_keys = species_key(train_df)
    print(f"  {len(train_df)} train specimens, {len(set(train_keys))} known (genus, species) taxa\n")

    rows = []
    for name, fname in TEST_SETS:
        fpath = os.path.join(data_dir, fname)
        print(f"Loading {name}: {fpath}")
        test_df = load_split(fpath)
        stats = compute_overlap(train_df, test_df)
        rows.append((name, stats))

        print(f"  Species overlap:  {stats['species_overlap_n']:>6d} / {stats['species_total']:<6d} "
              f"({stats['species_overlap_pct']:6.2f}%)")
        print(f"  Barcode overlap:  {stats['barcode_overlap_n']:>6d} / {stats['barcode_total']:<6d} "
              f"({stats['barcode_overlap_pct']:6.2f}%)")
        print(f"  Of the {stats['shared_species_specimens']} test specimens whose species IS in training:")
        print(f"    - {stats['shared_species_dup_barcode']} are EXACT duplicate barcodes (leakage)")
        print(f"    - {stats['shared_species_novel_barcode']} are a DIFFERENT individual of the same species "
              f"(same species, different barcode)")
        print(f"  CLEAN subset (barcode-duplicates AND unknown-species specimens removed — "
              f"{stats['clean_total']} / {stats['barcode_total']} specimens remain):")
        print(f"    Species overlap:  {stats['clean_species_overlap_n']:>6d} / {stats['clean_species_total']:<6d} "
              f"({stats['clean_species_overlap_pct']:6.2f}%)")
        print()

        if show_examples > 0:
            examples = sample_novel_barcode_examples(train_df, test_df, n=show_examples)
            print_examples(examples, name)

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
                    help="Path to ITS-5M data directory (containing trainset.fasta, test1-3.fasta).")
    p.add_argument("--show-examples", "--show_examples", dest="show_examples", type=int, default=0,
                    help="For each test set, print this many sampled 'same species, different barcode' "
                         "examples (test sequence + one same-species train sequence) so you can eyeball "
                         "whether they're really different. 0 = don't show (default).")
    return p


def cli():
    args = get_parser().parse_args()
    run(args.data_dir, args.show_examples)


if __name__ == "__main__":
    cli()