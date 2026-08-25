#!/usr/bin/env python
"""Check whether trainset_valid.fasta (the ITS-5M pretraining validation split)
overlaps with trainset.fasta (the training split / KNN gallery).

analyze_its_overlap.py only ever audits trainset.fasta against the three
external test pools (test1/test2/test3.fasta) -- it never touches
trainset_valid.fasta. Before using trainset_valid.fasta as a leakage-free
query set for aux-weight-tuning KNN eval (gallery = trainset.fasta, same as
the real KNN eval), we need the same species/genus/exact-barcode/substring-
duplicate audit run for it that test1/test2/test3 already got.

Reuses analyze_its_overlap.py's own load_split/compute_overlap/export_task_csv
so this is the exact same leakage definition, just pointed at a different
candidate file.

Usage:
    python barcodebert/analyze_its_valtrain_overlap.py --data-dir ./BarcodeMAE/data/ITS-5M
    python barcodebert/analyze_its_valtrain_overlap.py --data-dir ./BarcodeMAE/data/ITS-5M --export-dir ./BarcodeMAE/data/ITS-5M/tasks
"""

import argparse
import os

from analyze_its_overlap import (
    compute_overlap,
    export_task_csv,
    genus_key,
    load_split,
    print_examples,
    sample_novel_barcode_examples,
    species_key,
)


def run(data_dir, show_examples=0, export_dir=None):
    train_path = os.path.join(data_dir, "trainset.fasta")
    val_path = os.path.join(data_dir, "trainset_valid.fasta")

    print(f"Loading train set: {train_path}")
    train_df = load_split(train_path)
    train_known_df, train_keys = species_key(train_df)
    train_species = set(train_keys)
    _, train_genus_keys = genus_key(train_df)
    train_genera = set(train_genus_keys)
    train_seqs = set(train_df["sequence"])
    print(f"  {len(train_df)} train specimens, {len(train_species)} known (genus, species) taxa, "
          f"{len(train_genera)} known genera\n")

    print(f"Loading validation set: {val_path}")
    val_df = load_split(val_path)
    stats = compute_overlap(train_df, train_known_df, train_species, train_genera, train_seqs, val_df)

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        export_task_csv(val_df, stats, os.path.join(export_dir, "trainset_valid_tasks.csv"))

    name = "trainset_valid (vs train)"
    print(f"  Species overlap:  {stats['species_overlap_n']:>6d} / {stats['species_total']:<6d} "
          f"({stats['species_overlap_pct']:6.2f}%)")
    print(f"  Genus overlap:    {stats['genus_overlap_n']:>6d} / {stats['genus_total']:<6d} "
          f"({stats['genus_overlap_pct']:6.2f}%)")
    print(f"  Barcode overlap:  {stats['barcode_overlap_n']:>6d} / {stats['barcode_total']:<6d} "
          f"({stats['barcode_overlap_pct']:6.2f}%)  <- exact duplicate specimens (train == val, hard leakage)")
    print(f"  Of the {stats['shared_species_specimens']} val specimens whose species IS in training:")
    print(f"    - {stats['shared_species_dup_barcode']} are EXACT duplicate barcodes (leakage)")
    print(f"    - {stats['shared_species_novel_barcode']} are a DIFFERENT individual of the same species "
          f"(same species, different barcode -- expected/fine for a proper val split)")
    print(f"      - across {stats['shared_species_novel_barcode_unique_species']} unique species "
          f"(avg {stats['shared_species_novel_barcode_avg_per_species']:.2f} barcodes/species)")
    print(f"      - of which {stats['substring_dup_n']} are SUBSTRING duplicates (same read, different "
          f"trim -- soft leakage, not a real different individual)")
    print(f"    -> after also removing substring duplicates: {stats['task_species_level_n']} left, "
          f"across {stats['task_species_level_unique_species']} unique species "
          f"(avg {stats['task_species_level_avg_per_species']:.2f} barcodes/species)")
    print()

    if show_examples > 0:
        examples = sample_novel_barcode_examples(train_df, val_df, n=show_examples)
        print_examples(examples, name)

    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    if stats["barcode_overlap_n"] == 0 and stats["substring_dup_n"] == 0:
        print("Clean: trainset_valid.fasta has ZERO exact-duplicate or substring-duplicate specimens vs "
              "trainset.fasta. Safe to use as a leakage-free query set against the trainset.fasta gallery.")
    else:
        print(f"LEAKAGE FOUND: {stats['barcode_overlap_n']} exact-duplicate + {stats['substring_dup_n']} "
              f"substring-duplicate specimens in trainset_valid.fasta are also (near-)present in "
              f"trainset.fasta's gallery. Use the exported trainset_valid_tasks.csv (task in "
              f"{{species_level, genus_level}}) to filter them out before using this as a query set, "
              f"exactly like test1_tasks.csv/test2_tasks.csv already do for the real test pools.")
    print("=" * 100)


def get_parser():
    p = argparse.ArgumentParser(
        description="Check trainset_valid.fasta overlap with trainset.fasta (train/val leakage audit)."
    )
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="Path to ITS-5M data directory (containing trainset.fasta, trainset_valid.fasta).")
    p.add_argument("--show-examples", "--show_examples", dest="show_examples", type=int, default=0,
                    help="Print this many sampled 'same species, different barcode' examples. 0 = don't show.")
    p.add_argument("--export-dir", "--export_dir", dest="export_dir", default=None,
                    help="If set, write trainset_valid_tasks.csv here: id, genus, species, task. Filter to "
                         "task in {species_level, genus_level} for the leakage-free validation query set.")
    return p


def cli():
    args = get_parser().parse_args()
    run(args.data_dir, args.show_examples, args.export_dir)


if __name__ == "__main__":
    cli()