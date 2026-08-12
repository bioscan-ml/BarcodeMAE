#!/usr/bin/env python
"""Clean train/test overlap audit for UNITE+INSD (ITS-5M).

For each test set, in order:
  1. Exact sequence matches against the training set (literal duplicate reads).
  2. Substring matches among the remaining (non-exact) sequences: a test
     sequence that is a substring of some training sequence, or vice versa
     (same physical read, trimmed to a different length).
  3. Novel-species count: of what's left after removing (1) and (2), how many
     specimens belong to a species that does not appear in the training set
     at all (the genuinely out-of-distribution population).
  4. Unique-species reduction: training set's unique (genus, species) count
     vs. the final novel-species set's unique species count, and the same
     comparison for the ORIGINAL (unfiltered) test set's unique species count
     vs. the final set's, so both possible "reductions" are visible.
  5. Same two comparisons (train->final, original-test->final) for genus.

Uses mycoai.data.Data's own UNITE-header parser (tax_parser='unite') to get
(genus, species) straight from each fasta header, exactly as datasets.py does
for training/eval, so results match what the model was actually trained/
evaluated on.

Usage:
    python its_overlap_clean.py --data-dir ./BarcodeMAE/data/ITS-5M
    python its_overlap_clean.py --data-dir ./BarcodeMAE/data/ITS-5M --test-sets test1.fasta test2.fasta
"""

import argparse
import os

from mycoai import utils as mycoai_utils
from mycoai.data import Data

DEFAULT_TEST_SETS = [
    ("Test1 (Yeast)", "test1.fasta"),
    ("Test2 (Filamentous)", "test2.fasta"),
]


def load_split(fasta_path):
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"Fasta file not found: {fasta_path}")
    allow_duplicates = "train" in os.path.basename(fasta_path)
    return Data(fasta_path, allow_duplicates=allow_duplicates).data


def species_set(df):
    known = df[df["species"] != mycoai_utils.UNKNOWN_STR]
    return set(zip(known["genus"], known["species"]))


def genus_set(df):
    known = df[df["genus"] != mycoai_utils.UNKNOWN_STR]
    return set(known["genus"])


def find_substring_matches(remaining_df, train_seqs_by_species, train_seqs_all):
    """For each remaining (non-exact-match) test specimen, check whether its
    sequence is a substring of / contains as a substring any train sequence.
    Checked first against same-species train sequences (fast, catches the
    overwhelming majority — trimmed reads of the same specimen/species), then
    falls back to a full scan against all train sequences for specimens with
    an unresolved or train-absent species, so nothing is skipped by construction.
    """
    flagged = set()
    fallback_needed = []
    for idx, row in remaining_df.iterrows():
        key = (row["genus"], row["species"])
        test_seq = row["sequence"]
        same_species_train = train_seqs_by_species.get(key)
        if same_species_train:
            if any(test_seq in t or t in test_seq for t in same_species_train):
                flagged.add(idx)
                continue
        fallback_needed.append((idx, test_seq))

    for idx, test_seq in fallback_needed:
        if any(test_seq in t or t in test_seq for t in train_seqs_all):
            flagged.add(idx)

    return flagged


def run(data_dir, test_sets):
    train_path = os.path.join(data_dir, "trainset.fasta")
    print(f"Loading train set: {train_path}")
    train_df = load_split(train_path)
    train_seqs_all = set(train_df["sequence"])
    train_species = species_set(train_df)
    train_genera = genus_set(train_df)
    print(f"  {len(train_df)} train specimens | {len(train_species)} unique (genus, species) taxa | "
          f"{len(train_genera)} unique genera\n")

    train_known = train_df[train_df["species"] != mycoai_utils.UNKNOWN_STR].copy()
    train_known["_key"] = list(zip(train_known["genus"], train_known["species"]))
    train_seqs_by_species = train_known.groupby("_key")["sequence"].apply(lambda s: list(set(s))).to_dict()

    for name, fname in test_sets:
        fpath = os.path.join(data_dir, fname)
        print("=" * 100)
        print(f"{name}  ({fpath})")
        print("=" * 100)
        test_df = load_split(fpath)
        n_total = len(test_df)
        orig_unique_species = species_set(test_df)
        orig_unique_genera = genus_set(test_df)
        print(f"Original: {n_total} specimens | {len(orig_unique_species)} unique species | "
              f"{len(orig_unique_genera)} unique genera")

        # 1. Exact matches -----------------------------------------------------
        is_exact = test_df["sequence"].isin(train_seqs_all)
        n_exact = int(is_exact.sum())
        print(f"\n1. Exact sequence matches vs.\\ training set: {n_exact} / {n_total}")

        remaining = test_df[~is_exact]

        # 2. Substring matches (among the non-exact remainder) -----------------
        substring_idx = find_substring_matches(remaining, train_seqs_by_species, train_seqs_all)
        n_substring = len(substring_idx)
        print(f"2. Substring matches vs.\\ training set (of the {len(remaining)} non-exact remainder): "
              f"{n_substring}")

        is_leakage = is_exact | test_df.index.to_series().isin(substring_idx)
        clean_df = test_df[~is_leakage]
        print(f"   -> {len(clean_df)} specimens left after removing exact + substring matches "
              f"({n_exact + n_substring} / {n_total} total removed as leakage)")

        # 3. Novel-species count (species not in training at all) --------------
        is_known_species = clean_df["species"] != mycoai_utils.UNKNOWN_STR
        clean_known = clean_df[is_known_species]
        clean_species_keys = list(zip(clean_known["genus"], clean_known["species"]))
        is_novel_species = [k not in train_species for k in clean_species_keys]
        novel_df = clean_known[is_novel_species]
        n_novel = len(novel_df)
        print(f"\n3. Of the {len(clean_known)} leakage-free specimens with a resolved species, "
              f"{n_novel} belong to a species NOT present in training at all (novel species)")

        # 4. Unique-species reduction -------------------------------------------
        final_unique_species = species_set(novel_df)
        print(f"\n4. Unique species:")
        print(f"   Training set:                {len(train_species)}")
        print(f"   Original {name} test set:    {len(orig_unique_species)}")
        print(f"   Final novel-species set:     {len(final_unique_species)}")
        print(f"   Reduction, training -> final:        {len(train_species)} -> {len(final_unique_species)} "
              f"(-{len(train_species) - len(final_unique_species)})")
        print(f"   Reduction, original test -> final:   {len(orig_unique_species)} -> {len(final_unique_species)} "
              f"(-{len(orig_unique_species) - len(final_unique_species)})")

        # 5. Same for genus -------------------------------------------------------
        final_unique_genera = genus_set(novel_df)
        print(f"\n5. Unique genera:")
        print(f"   Training set:                {len(train_genera)}")
        print(f"   Original {name} test set:    {len(orig_unique_genera)}")
        print(f"   Final novel-species set:     {len(final_unique_genera)}")
        print(f"   Reduction, training -> final:        {len(train_genera)} -> {len(final_unique_genera)} "
              f"(-{len(train_genera) - len(final_unique_genera)})")
        print(f"   Reduction, original test -> final:   {len(orig_unique_genera)} -> {len(final_unique_genera)} "
              f"(-{len(orig_unique_genera) - len(final_unique_genera)})")
        print()


def get_parser():
    p = argparse.ArgumentParser(description="Clean ITS-5M (UNITE+INSD) train/test overlap audit.")
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="Path to ITS-5M data directory (containing trainset.fasta, test1.fasta, test2.fasta).")
    p.add_argument("--test-sets", "--test_sets", dest="test_sets", nargs="+", default=None,
                    help="Fasta filenames (relative to --data-dir) to audit. Default: test1.fasta test2.fasta "
                         "(Yeast, Filamentous — the two test sets used in the paper).")
    return p


def cli():
    args = get_parser().parse_args()
    if args.test_sets:
        test_sets = [(fname, fname) for fname in args.test_sets]
    else:
        test_sets = DEFAULT_TEST_SETS
    run(args.data_dir, test_sets)


if __name__ == "__main__":
    cli()