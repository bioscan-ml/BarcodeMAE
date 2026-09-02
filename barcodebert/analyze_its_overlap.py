#!/usr/bin/env python
"""Verify the ITS-5M train/test overlap numbers (species overlap and identical
barcode overlap), break down WHY species overlap can be much higher than
identical-barcode overlap, and build a "clean" table after removing all
leakage (exact duplicates AND same-species substring/trimming duplicates).

Motivation: Table 6 in the paper reports, for each of the 3 fungi test sets,
how many of its species/barcodes are also present in the training set. Test
Set 2 (Filamentous Fungi) has much higher species overlap (47.86%) than
identical-barcode overlap (6.48%) — i.e. many test specimens belong to a
species the model also saw in training, but as a DIFFERENT individual
specimen with a different exact ITS sequence (normal intraspecific sequence
variation). Test Sets 1 (Yeast) and 3 (MycoAI) instead have high overlap on
BOTH axes (86.73% / 100.00% identical barcodes), which looks like train/test
leakage of literal duplicate specimens rather than natural species overlap.

On top of exact-string duplicates, some "same species, different barcode"
pairs turn out to be the SAME physical read trimmed to different lengths
(one sequence is an exact substring of the other) rather than a genuinely
different individual — this script also detects and excludes those as a
second, softer leakage category.

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
    python analyze_its_overlap.py --data-dir ./BarcodeMAE/data/ITS-5M --show-examples 10
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


def genus_key(df):
    """genus values, excluding rows where genus is unresolved."""
    known = df[df["genus"] != mycoai_utils.UNKNOWN_STR]
    return known, known["genus"].tolist()


def find_substring_duplicates(train_known_df, candidates):
    """For each candidate test specimen (same species as some train specimen,
    but not an exact-sequence duplicate), check whether its sequence is a
    substring of — or contains as a substring — any TRAIN sequence of the
    SAME species. That indicates the same physical read trimmed differently,
    not a genuinely different individual. Only searches within the relevant
    species (not the full 5.2M-row train set) to stay fast.

    Returns a set of candidate row-index values flagged as substring-dupes.
    """
    if len(candidates) == 0:
        return set()

    needed_species = set(zip(candidates["genus"], candidates["species"]))
    train_known_df = train_known_df.copy()
    train_known_df["_key"] = list(zip(train_known_df["genus"], train_known_df["species"]))
    relevant_train = train_known_df[train_known_df["_key"].isin(needed_species)]
    species_to_train_seqs = relevant_train.groupby("_key")["sequence"].apply(lambda s: list(set(s)))

    flagged = set()
    for idx, row in candidates.iterrows():
        key = (row["genus"], row["species"])
        test_seq = row["sequence"]
        for train_seq in species_to_train_seqs.get(key, []):
            if test_seq in train_seq or train_seq in test_seq:
                flagged.add(idx)
                break
    return flagged


def compute_overlap(train_df, train_known_df, train_species, train_genera, train_seqs, test_df, include_leaked=False):
    _, test_keys = species_key(test_df)
    unique_test_species = set(test_keys)
    species_overlap = unique_test_species & train_species

    # Genus-level overlap (same idea as species, one taxonomic rank up) --------
    _, test_genus_keys = genus_key(test_df)
    unique_test_genera = set(test_genus_keys)
    genus_overlap = unique_test_genera & train_genera

    is_barcode_dup = test_df["sequence"].isin(train_seqs)
    n_barcode_overlap = int(is_barcode_dup.sum())

    test_species_col = list(zip(test_df["genus"], test_df["species"]))
    is_shared_species = pd.Series(
        [k in species_overlap for k in test_species_col], index=test_df.index
    )
    is_known_species = test_df["species"] != mycoai_utils.UNKNOWN_STR

    n_shared_species_specimens = int(is_shared_species.sum())
    n_shared_species_and_dup_barcode = int((is_shared_species & is_barcode_dup).sum())
    n_shared_species_novel_barcode = n_shared_species_specimens - n_shared_species_and_dup_barcode

    # Substring/trimming duplicates: same species, not an exact match, but one
    # sequence is a substring of the other (same physical read, different trim).
    candidates = test_df[is_shared_species & ~is_barcode_dup]
    substring_dup_idx = find_substring_duplicates(train_known_df, candidates)
    is_substring_dup = test_df.index.to_series().isin(substring_dup_idx)
    n_substring_dup = len(substring_dup_idx)

    # Among the "species overlap remaining after removing exact-duplicate
    # barcodes" specimens (candidates, above): how many distinct species do
    # they belong to, and on average how many of these specimens does each
    # of those species have? (Not yet substring-deduped -- this is the raw
    # "different barcode, same species" population.)
    n_shared_species_novel_barcode_unique_species = len(set(zip(candidates["genus"], candidates["species"])))
    shared_species_novel_barcode_avg_per_species = (
        len(candidates) / n_shared_species_novel_barcode_unique_species
        if n_shared_species_novel_barcode_unique_species else float("nan")
    )

    # Fully clean: known species, not an exact duplicate, not a substring
    # duplicate. Species overlap recomputed on just this subset.
    # include_leaked=True skips the duplicate exclusion entirely -- exact
    # and substring duplicates flow into species_level/genus_level below
    # exactly like any other same-species specimen, producing the
    # leakage-INCLUDED counterpart of the same task definitions (task_series
    # assignment order below overwrites "exact_duplicate"/"substring_duplicate"
    # with "species_level"/"genus_level" for these rows once is_clean covers
    # them, no other code path needs to change).
    is_clean = is_known_species if include_leaked else (is_known_species & ~is_barcode_dup & ~is_substring_dup)
    clean_df = test_df[is_clean]
    _, clean_keys = species_key(clean_df)
    clean_unique_species = set(clean_keys)
    clean_species_overlap = clean_unique_species & train_species

    n_unknown_species = int((~is_known_species).sum())

    # Task-specific specimen counts -------------------------------------------
    # Task A (species-level KNN): clean specimens whose SPECIES is in training
    # — same species, genuinely different individual/barcode. Well-posed
    # species-level classification (the answer key exists in the gallery).
    is_clean_species_seen = is_clean & is_shared_species
    task_species_level_n = int(is_clean_species_seen.sum())

    # Same as shared_species_novel_barcode_unique_species/avg_per_species
    # above, but on the FINAL clean population (substring duplicates also
    # removed) -- i.e. what's actually left for the species-level task.
    clean_seen_df = test_df[is_clean_species_seen]
    task_species_level_unique_species = len(set(zip(clean_seen_df["genus"], clean_seen_df["species"])))
    task_species_level_avg_per_species = (
        task_species_level_n / task_species_level_unique_species
        if task_species_level_unique_species else float("nan")
    )

    # Task B (genus-level KNN on unseen species): clean specimens whose SPECIES
    # is novel (not in training at all) but whose GENUS is in training. Species
    # can't be predicted (no gallery entry), but genus can.
    is_known_genus = test_df["genus"] != mycoai_utils.UNKNOWN_STR
    is_genus_in_train = test_df["genus"].isin(train_genera)
    is_clean_species_novel = is_clean & ~is_shared_species
    is_clean_species_novel_genus_seen = is_clean_species_novel & is_known_genus & is_genus_in_train
    task_genus_level_n = int(is_clean_species_novel_genus_seen.sum())

    # Specimens where even genus is novel — unusable at species OR genus level.
    is_clean_unusable = is_clean_species_novel & ~(is_known_genus & is_genus_in_train)
    task_unusable_n = int(is_clean_unusable.sum())

    # Full per-specimen task label, for CSV export. The known-species branches
    # below are collectively exhaustive over (barcode_dup, substring_dup,
    # shared_species, genus_seen), so every known-species row gets overwritten;
    # "_uncategorized_bug" is a canary that should never survive to output.
    task_series = pd.Series("unknown_species", index=test_df.index)
    task_series[is_known_species] = "_uncategorized_bug"
    task_series[is_known_species & is_barcode_dup] = "exact_duplicate"
    task_series[is_known_species & ~is_barcode_dup & is_substring_dup] = "substring_duplicate"
    task_series[is_clean_species_seen] = "species_level"          # Task A
    task_series[is_clean_species_novel_genus_seen] = "genus_level"  # Task B
    task_series[is_clean_unusable] = "unusable"

    return {
        "species_total": len(unique_test_species),
        "species_overlap_n": len(species_overlap),
        "species_overlap_pct": 100.0 * len(species_overlap) / len(unique_test_species) if unique_test_species else float("nan"),
        "genus_total": len(unique_test_genera),
        "genus_overlap_n": len(genus_overlap),
        "genus_overlap_pct": 100.0 * len(genus_overlap) / len(unique_test_genera) if unique_test_genera else float("nan"),
        "barcode_total": len(test_df),
        "barcode_overlap_n": n_barcode_overlap,
        "barcode_overlap_pct": 100.0 * n_barcode_overlap / len(test_df) if len(test_df) else float("nan"),
        "shared_species_specimens": n_shared_species_specimens,
        "shared_species_dup_barcode": n_shared_species_and_dup_barcode,
        "shared_species_novel_barcode": n_shared_species_novel_barcode,
        "shared_species_novel_barcode_unique_species": n_shared_species_novel_barcode_unique_species,
        "shared_species_novel_barcode_avg_per_species": shared_species_novel_barcode_avg_per_species,
        "substring_dup_n": n_substring_dup,
        "unknown_species_n": n_unknown_species,
        "clean_total": len(clean_df),
        "clean_species_total": len(clean_unique_species),
        "clean_species_overlap_n": len(clean_species_overlap),
        "clean_species_overlap_pct": (100.0 * len(clean_species_overlap) / len(clean_unique_species)
                                       if clean_unique_species else float("nan")),
        "task_species_level_n": task_species_level_n,
        "task_species_level_unique_species": task_species_level_unique_species,
        "task_species_level_avg_per_species": task_species_level_avg_per_species,
        "task_genus_level_n": task_genus_level_n,
        "task_unusable_n": task_unusable_n,
        "task_series": task_series,
    }


def sample_novel_barcode_examples(train_df, test_df, n=10, seed=0):
    """Sample up to n (species, test_seq, one_train_seq_of_same_species) examples
    from the "same species, different barcode" category, for eyeballing whether
    the sequences really are meaningfully different (not near-identical)."""
    _, train_keys = species_key(train_df)
    train_species = set(train_keys)
    train_seqs = set(train_df["sequence"])

    train_known_df, _ = species_key(train_df)
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


def export_task_csv(test_df, stats, out_path):
    """Write id, genus, species, task for every specimen in this test set.
    Downstream eval scripts filter on task in {"species_level", "genus_level"}
    to get the exact leakage-free query sets for Task A / Task B."""
    out_df = pd.DataFrame({
        "id": test_df["id"],
        "genus": test_df["genus"],
        "species": test_df["species"],
        "task": stats["task_series"],
    })
    out_df.to_csv(out_path, index=False)
    print(f"  Wrote {len(out_df)} rows -> {out_path}")
    print(f"    task counts: {out_df['task'].value_counts().to_dict()}")


def run(data_dir, show_examples=0, export_dir=None, include_leaked=False):
    train_path = os.path.join(data_dir, "trainset.fasta")
    print(f"Loading train set: {train_path}")
    train_df = load_split(train_path)
    train_known_df, train_keys = species_key(train_df)
    train_species = set(train_keys)
    _, train_genus_keys = genus_key(train_df)
    train_genera = set(train_genus_keys)
    train_seqs = set(train_df["sequence"])
    print(f"  {len(train_df)} train specimens, {len(train_species)} known (genus, species) taxa, "
          f"{len(train_genera)} known genera\n")

    rows = []
    for name, fname in TEST_SETS:
        fpath = os.path.join(data_dir, fname)
        print(f"Loading {name}: {fpath}")
        test_df = load_split(fpath)
        stats = compute_overlap(train_df, train_known_df, train_species, train_genera, train_seqs, test_df,
                                 include_leaked=include_leaked)
        rows.append((name, stats))

        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            tag = fname.replace(".fasta", "")
            export_task_csv(test_df, stats, os.path.join(export_dir, f"{tag}_tasks.csv"))

        print(f"  Species overlap:  {stats['species_overlap_n']:>6d} / {stats['species_total']:<6d} "
              f"({stats['species_overlap_pct']:6.2f}%)")
        print(f"  Genus overlap:    {stats['genus_overlap_n']:>6d} / {stats['genus_total']:<6d} "
              f"({stats['genus_overlap_pct']:6.2f}%)")
        print(f"  Barcode overlap:  {stats['barcode_overlap_n']:>6d} / {stats['barcode_total']:<6d} "
              f"({stats['barcode_overlap_pct']:6.2f}%)")
        print(f"  Of the {stats['shared_species_specimens']} test specimens whose species IS in training:")
        print(f"    - {stats['shared_species_dup_barcode']} are EXACT duplicate barcodes (leakage)")
        print(f"    - {stats['shared_species_novel_barcode']} are a DIFFERENT individual of the same species "
              f"(same species, different barcode)")
        print(f"      - across {stats['shared_species_novel_barcode_unique_species']} unique species "
              f"(avg {stats['shared_species_novel_barcode_avg_per_species']:.2f} barcodes/species)")
        print(f"      - of which {stats['substring_dup_n']} are SUBSTRING duplicates (same read, different "
              f"trim — soft leakage, not a real different individual)")
        print(f"    -> after also removing substring duplicates: {stats['task_species_level_n']} left, "
              f"across {stats['task_species_level_unique_species']} unique species "
              f"(avg {stats['task_species_level_avg_per_species']:.2f} barcodes/species)")
        print()

        if show_examples > 0:
            examples = sample_novel_barcode_examples(train_df, test_df, n=show_examples)
            print_examples(examples, name)

    print("=" * 130)
    print("CLEAN COMPLETE TABLE — exact duplicates, substring duplicates, and unknown-species specimens all removed")
    print("=" * 130)
    header = (f"{'Test Set':<22}{'Total':>8}{'ExactDup':>10}{'SubstrDup':>11}{'UnkSpecies':>12}"
              f"{'Clean':>8}{'Clean SpOverlap':>18}")
    print(header)
    for name, stats in rows:
        print(f"{name:<22}{stats['barcode_total']:>8d}{stats['barcode_overlap_n']:>10d}"
              f"{stats['substring_dup_n']:>11d}{stats['unknown_species_n']:>12d}"
              f"{stats['clean_total']:>8d}"
              f"{stats['clean_species_overlap_n']:>10d}/{stats['clean_species_total']:<6d}")
    print("=" * 130)
    print("Clean = barcode not seen in training AND species is resolved AND not a substring/trim duplicate.")
    print("'Clean SpOverlap' = of the clean specimens' unique species, how many are still in the training vocabulary.\n")

    print("=" * 130)
    print("GENUS-LEVEL OVERLAP (train vs each full test set, mirrors the species-overlap table)")
    print("=" * 130)
    print(f"{'Test Set':<22}{'GenusOverlap':>16}{'GenusPct':>12}")
    for name, stats in rows:
        print(f"{name:<22}{stats['genus_overlap_n']:>10d}/{stats['genus_total']:<5d}{stats['genus_overlap_pct']:>11.2f}%")
    print("=" * 130 + "\n")

    print("=" * 130)
    print("EVALUABLE TASK COUNTS — a KNN gallery built from train can only ever predict a label that exists in")
    print("train, so these are the only two well-posed evaluation tasks on the clean (leakage-free) specimens:")
    print("=" * 130)
    print(f"{'Test Set':<22}{'A: species-level':>18}{'B: genus-level':>16}{'Unusable':>10}")
    for name, stats in rows:
        print(f"{name:<22}{stats['task_species_level_n']:>18d}{stats['task_genus_level_n']:>16d}{stats['task_unusable_n']:>10d}")
    print("(A = species-in-train specimens; B = species-novel-but-genus-in-train specimens; "
          "Unusable = both novel)")
    print("=" * 130)
    print("A: same species, genuinely different barcode — tests whether two individuals of the same species land")
    print("   close together in embedding space (realistic 'ID a new specimen of a known species' scenario).")
    print("B: species never seen in training at all, but its genus was — species-level prediction is impossible")
    print("   by construction (no gallery entry for that species), so evaluate genus-level prediction instead.")
    print("Unusable: neither the species nor the genus exists in training — not evaluable at either rank with")
    print("          a closed-set KNN/classification gallery built from this train set.")


def get_parser():
    p = argparse.ArgumentParser(description="Verify ITS-5M train/test overlap numbers (Table 6).")
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="Path to ITS-5M data directory (containing trainset.fasta, test1-3.fasta).")
    p.add_argument("--show-examples", "--show_examples", dest="show_examples", type=int, default=0,
                    help="For each test set, print this many sampled 'same species, different barcode' "
                         "examples (test sequence + one same-species train sequence) so you can eyeball "
                         "whether they're really different. 0 = don't show (default).")
    p.add_argument("--export-dir", "--export_dir", dest="export_dir", default=None,
                    help="If set, write <test>_tasks.csv per test set here: id, genus, species, task "
                         "(task in {species_level, genus_level, unusable, exact_duplicate, "
                         "substring_duplicate, unknown_species}). Feed species_level/genus_level rows "
                         "into knn_its_clean.py for the leakage-free evaluation.")
    p.add_argument("--include-leaked", "--include_leaked", dest="include_leaked", action="store_true",
                    help="Do not exclude exact-duplicate/substring-duplicate specimens from the "
                         "species_level/genus_level task pools -- produces the leakage-INCLUDED counterpart "
                         "of the same task definitions, using the exact same downstream eval "
                         "(knn_its_clean.py) via --tasks-dir pointed at a separate --export-dir.")
    return p


def cli():
    args = get_parser().parse_args()
    run(args.data_dir, args.show_examples, args.export_dir, args.include_leaked)


if __name__ == "__main__":
    cli()