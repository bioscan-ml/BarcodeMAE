#!/usr/bin/env python
"""One-off distribution check for ITS-5M's trainset.fasta, mirroring the
BIOSCAN-5M supervised_train.csv analysis (genus/species specimen density,
sequence length variability) -- to compare against BIOSCAN-5M's gallery
distribution as a candidate explanation for why softmax-KNN gives a bigger
accuracy boost on ITS-5M than on BIOSCAN-5M.

Usage:
    python analyze_its_distribution.py --data-dir /path/to/ITS-5M
"""

import argparse

from mycoai import utils as mycoai_utils
from mycoai.data import Data


def run(data_dir):
    fasta_path = f"{data_dir}/trainset.fasta"
    print(f"Loading {fasta_path} ...")
    fungi_data = Data(fasta_path, allow_duplicates=True)
    df = fungi_data.data

    print(f"\n=== ITS-5M trainset.fasta (gallery) ===")
    print(f"Total specimens: {len(df)}")

    known_genus = df[df["genus"] != mycoai_utils.UNKNOWN_STR]
    genus_counts = known_genus["genus"].value_counts()
    print(f"Unique genera (known): {genus_counts.shape[0]}")
    print(f"Specimens per genus: mean={genus_counts.mean():.1f}, median={genus_counts.median():.0f}, "
          f"min={genus_counts.min()}, max={genus_counts.max()}")
    print(f"  Genera with only 1 specimen: {(genus_counts==1).sum()} ({100*(genus_counts==1).sum()/len(genus_counts):.1f}%)")
    print(f"  Genera with <=5 specimens: {(genus_counts<=5).sum()} ({100*(genus_counts<=5).sum()/len(genus_counts):.1f}%)")
    print(f"  Genera with >=20 specimens: {(genus_counts>=20).sum()} ({100*(genus_counts>=20).sum()/len(genus_counts):.1f}%)")

    known_species = df[df["species"] != mycoai_utils.UNKNOWN_STR]
    species_keys = list(zip(known_species["genus"], known_species["species"]))
    import pandas as pd
    species_counts = pd.Series(species_keys).value_counts()
    print(f"\nUnique (genus, species) taxa: {species_counts.shape[0]}")
    print(f"Specimens per species: mean={species_counts.mean():.1f}, median={species_counts.median():.0f}")

    seqlen = df["sequence"].str.len()
    print(f"\nSequence length: mean={seqlen.mean():.1f}, std={seqlen.std():.1f}, min={seqlen.min()}, max={seqlen.max()}")
    print(f"  Coefficient of variation (std/mean): {seqlen.std()/seqlen.mean():.4f}")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True)
    return p


if __name__ == "__main__":
    run(get_parser().parse_args().data_dir)