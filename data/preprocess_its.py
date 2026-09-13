"""
Preprocessing script for fungal ITS data (trainset.fasta + trainset_labels.csv).

Applies four filtering steps from BarcodeMamba+ (Gao et al., 2025):
  1. Remove duplicate sequence-label pairs
  2. Exclude sequences outside mean ± 4*std length
  3. Remove sequences with >5% ambiguous bases (non-ACGT)
  4. Eliminate samples whose finest known taxonomic label appears < 3 times

Usage:
    python preprocess_its.py \
        --fasta trainset.fasta \
        --labels trainset_labels.csv \
        --out_fasta trainset_filtered.fasta \
        --out_labels trainset_filtered_labels.csv
"""

import argparse

import numpy as np
import pandas as pd

UNKNOWN = 9999999
AMBIGUOUS = set("NRYWSKMBDHV")  # all IUPAC non-ACGT bases
TAX_LEVELS = ["phylum", "class", "order", "family", "genus", "species"]


# ── I/O helpers ──────────────────────────────────────────────────────────────


def read_fasta(path):
    """Return list of (header, sequence) tuples."""
    records = []
    header, seq_parts = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def write_fasta(records, path):
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")


# ── Filtering steps ───────────────────────────────────────────────────────────


def step1_remove_duplicates(sequences, labels_df):
    """Remove exact duplicate (sequence, full-label-tuple) pairs."""
    label_tuples = [tuple(row) for row in labels_df[TAX_LEVELS].itertuples(index=False)]
    seen = set()
    keep = []
    for i, (seq, lab) in enumerate(zip(sequences, label_tuples)):
        key = (seq, lab)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return keep


def step2_filter_length(sequences, n_std=4):
    """Keep sequences within mean ± n_std * std of sequence length."""
    lengths = np.array([len(s) for s in sequences])
    mean_len = lengths.mean()
    std_len = lengths.std()
    lower = mean_len - n_std * std_len
    upper = mean_len + n_std * std_len
    keep = [i for i, l in enumerate(lengths) if lower <= l <= upper]
    return keep, mean_len, std_len, lower, upper


def step3_filter_ambiguous(sequences, max_ambig_frac=0.05):
    """Remove sequences where fraction of ambiguous (non-ACGT) bases > threshold."""
    keep = []
    for i, seq in enumerate(sequences):
        n_ambig = sum(1 for b in seq if b in AMBIGUOUS)
        if len(seq) == 0 or n_ambig / len(seq) <= max_ambig_frac:
            keep.append(i)
    return keep


def step4_filter_rare_classes(labels_df, min_samples=3):
    """
    Remove samples whose finest known taxonomic label (lowest non-UNKNOWN level)
    appears fewer than min_samples times across the dataset.

    For each sample we find the most specific known level and apply the count
    threshold at that level.  Samples that are entirely UNKNOWN at all levels
    are retained (they carry no label signal to filter on).
    """
    df = labels_df[TAX_LEVELS].copy()

    # Build a per-level count map (ignoring UNKNOWN entries)
    level_counts = {}
    for level in TAX_LEVELS:
        valid = df[level][df[level] != UNKNOWN]
        level_counts[level] = valid.value_counts().to_dict()

    keep = []
    for i, row in df.iterrows():
        # Find the finest level that has a known label
        finest_level = None
        finest_label = None
        for level in reversed(TAX_LEVELS):  # species → phylum
            if row[level] != UNKNOWN:
                finest_level = level
                finest_label = row[level]
                break

        if finest_level is None:
            # Entirely unlabelled — keep
            keep.append(i)
            continue

        count = level_counts[finest_level].get(finest_label, 0)
        if count >= min_samples:
            keep.append(i)

    return keep


# ── Statistics printer ────────────────────────────────────────────────────────


def print_stats(tag, sequences, labels_df):
    lengths = [len(s) for s in sequences]
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    print(f"  Sequences        : {len(sequences):>10,}")
    if lengths:
        print(f"  Length  mean±std : {np.mean(lengths):.1f} ± {np.std(lengths):.1f} bp")
        print(f"  Length  min/max  : {min(lengths)} / {max(lengths)} bp")
    for level in TAX_LEVELS:
        col = labels_df[level]
        n_known = (col != UNKNOWN).sum()
        n_unique = col[col != UNKNOWN].nunique()
        pct = 100 * n_known / len(col) if len(col) > 0 else 0
        print(f"  {level:<10} known: {n_known:>8,}  ({pct:5.1f}%)  unique: {n_unique:>6,}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Preprocess ITS fungal dataset.")
    parser.add_argument("--fasta", default="trainset.fasta", help="Input FASTA file")
    parser.add_argument("--labels", default="trainset_labels.csv", help="Input labels CSV")
    parser.add_argument("--out_fasta", default="trainset_filtered.fasta", help="Output FASTA")
    parser.add_argument("--out_labels", default="trainset_filtered_labels.csv", help="Output labels CSV")
    parser.add_argument("--n_std", type=float, default=4.0, help="Std-dev multiplier for length filter")
    parser.add_argument("--max_ambig", type=float, default=0.05, help="Max fraction of ambiguous bases")
    parser.add_argument("--min_samples", type=int, default=3, help="Min samples per taxonomic class")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading FASTA  : {args.fasta}")
    records = read_fasta(args.fasta)
    headers = [h for h, _ in records]
    sequences = [s for _, s in records]

    print(f"Loading labels : {args.labels}")
    labels_df = pd.read_csv(args.labels)

    if len(sequences) != len(labels_df):
        raise ValueError(f"FASTA has {len(sequences)} sequences but labels CSV has {len(labels_df)} rows.")

    print_stats("BEFORE FILTERING", sequences, labels_df)

    # ── Step 1: Remove duplicates ─────────────────────────────────────────────
    keep = step1_remove_duplicates(sequences, labels_df)
    n_removed = len(sequences) - len(keep)
    sequences = [sequences[i] for i in keep]
    headers = [headers[i] for i in keep]
    labels_df = labels_df.iloc[keep].reset_index(drop=True)
    print(f"\n[Step 1] Removed {n_removed:,} duplicate sequence-label pairs  →  {len(sequences):,} remain")

    # ── Step 2: Length filter ─────────────────────────────────────────────────
    keep, mean_l, std_l, lo, hi = step2_filter_length(sequences, args.n_std)
    n_removed = len(sequences) - len(keep)
    sequences = [sequences[i] for i in keep]
    headers = [headers[i] for i in keep]
    labels_df = labels_df.iloc[keep].reset_index(drop=True)
    print(f"[Step 2] Length filter  mean={mean_l:.1f}  std={std_l:.1f}  window=[{lo:.0f}, {hi:.0f}] bp")
    print(f"         Removed {n_removed:,} sequences  →  {len(sequences):,} remain")

    # ── Step 3: Ambiguous base filter ─────────────────────────────────────────
    keep = step3_filter_ambiguous(sequences, args.max_ambig)
    n_removed = len(sequences) - len(keep)
    sequences = [sequences[i] for i in keep]
    headers = [headers[i] for i in keep]
    labels_df = labels_df.iloc[keep].reset_index(drop=True)
    print(
        f"[Step 3] Removed {n_removed:,} sequences with >{args.max_ambig*100:.0f}% ambiguous bases  →  {len(sequences):,} remain"
    )

    # ── Step 4: Rare-class filter ─────────────────────────────────────────────
    keep = step4_filter_rare_classes(labels_df, args.min_samples)
    n_removed = len(sequences) - len(keep)
    sequences = [sequences[i] for i in keep]
    headers = [headers[i] for i in keep]
    labels_df = labels_df.iloc[keep].reset_index(drop=True)
    print(
        f"[Step 4] Removed {n_removed:,} samples from classes with <{args.min_samples} representatives  →  {len(sequences):,} remain"
    )

    # ── Final stats ───────────────────────────────────────────────────────────
    print_stats("AFTER FILTERING", sequences, labels_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    write_fasta(list(zip(headers, sequences)), args.out_fasta)
    labels_df.to_csv(args.out_labels, index=False)
    print(f"\nSaved filtered FASTA   → {args.out_fasta}")
    print(f"Saved filtered labels  → {args.out_labels}")


if __name__ == "__main__":
    main()
