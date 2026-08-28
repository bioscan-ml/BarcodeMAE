#!/usr/bin/env python
"""Zero-shot BIN reconstruction (ZSC) for BIOSCAN-5M using a pretrained
BarcodeMamba/BarcodeMamba+ checkpoint. Reuses zsc_evaluation_v2.py's own
zsc_pipeline() (UMAP -> AgglomerativeClustering -> AMI) unmodified; only
embedding extraction differs (see barcodemamba_common.py).

Usage:
    python zsc_barcodemamba.py \
        --barcodemamba-repo /scratch/$USER/BarcodeMamba-dev \
        --checkpoint-dir /scratch/$USER/barcodemamba_checkpoints/BarcodeMamba-plus-BIOSCAN-5M \
        --bpe-tokenizer-path /scratch/$USER/barcodemamba_checkpoints/BarcodeMamba-plus-BIOSCAN-5M/bpe_tokenizer.pkl \
        --data-dir ./BarcodeMAE/data/BIOSCAN-5M \
        --run-name zsc_external_barcodemamba_bioscan5m \
        --results-file results_final/ZSC_external_RESULTS.txt
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from barcodebert.barcodemamba_common import embed_sequences, load_barcodemamba, load_bpe_tokenizer
from barcodebert.zsc_evaluation_v2 import zsc_pipeline


def run(config):
    t_start = time.time()

    print(f"Loading BarcodeMamba checkpoint from: {config.checkpoint_dir}")
    model, bm_config = load_barcodemamba(config.barcodemamba_repo, config.checkpoint_dir, config.checkpoint_name)
    tokenizer_name = bm_config.tokenizer.name
    print(f"  Tokenizer: {tokenizer_name}")
    if tokenizer_name == "bpe":
        tokenizer = load_bpe_tokenizer(config.bpe_tokenizer_path)
    else:
        import sys
        if config.barcodemamba_repo not in sys.path:
            sys.path.insert(0, config.barcodemamba_repo)
        from utils.ssm_dataset import get_tokenizer
        tokenizer = get_tokenizer(tokenizer_name, bm_config.tokenizer)

    model.cuda()
    model.eval()

    # Matches zsc_evaluation_v2.py's exact branching (BIOSCAN-5M-only here).
    # Note: the literal default "bin_uri" does NOT hit the "bin" special case
    # below -- it falls through to the else branch, giving target_level =
    # "bin_uri_index", which is the real column name used in practice.
    if config.taxon.lower() == "bin":
        target_level = "bin_uri"
    elif config.taxon.lower() == "dna_bin":
        target_level = "dna_bin"
    else:
        target_level = f"{config.taxon}_index"

    df_test = pd.read_csv(os.path.join(config.data_dir, "supervised_test.csv"))
    df_unseen = pd.read_csv(os.path.join(config.data_dir, "unseen.csv"))
    df_test = df_test.dropna(subset=[target_level])
    df_unseen = df_unseen.dropna(subset=[target_level])

    print(f"\nExtracting embeddings for supervised_test ({len(df_test)} specimens)...")
    X_test = embed_sequences(model, tokenizer, tokenizer_name, df_test["dna_barcode"], config.max_length)
    y_test = df_test[target_level].to_numpy()

    print(f"Extracting embeddings for unseen ({len(df_unseen)} specimens)...")
    X_unseen = embed_sequences(model, tokenizer, tokenizer_name, df_unseen["dna_barcode"], config.max_length)
    y_unseen = df_unseen[target_level].to_numpy()

    X = np.vstack([X_test, X_unseen])
    y = np.hstack([y_test, y_unseen])
    print(f"\nCombined: {X.shape[0]} samples, {len(np.unique(y))} unique labels, dim={X.shape[1]}")

    ami = 100.0 * zsc_pipeline(X, y, metric=config.metric, n_neighbours=config.n_neighbors,
                                n_clusters=config.n_clusters)
    print(f"\nFinal AMI (%): {ami:.4f}")
    print(f"Total time: {time.time() - t_start:.1f}s")

    model_name = os.path.basename(os.path.normpath(config.checkpoint_dir))
    with open(config.results_file, "a") as f:
        f.write(f"\n{config.run_name}_{model_name}\t{ami:.4f}")


def get_parser():
    p = argparse.ArgumentParser(description="ZSC evaluation for BIOSCAN-5M using a BarcodeMamba/BarcodeMamba+ checkpoint.")
    p.add_argument("--barcodemamba-repo", "--barcodemamba_repo", dest="barcodemamba_repo", required=True)
    p.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", required=True)
    p.add_argument("--checkpoint-name", "--checkpoint_name", dest="checkpoint_name", default=None)
    p.add_argument("--bpe-tokenizer-path", "--bpe_tokenizer_path", dest="bpe_tokenizer_path", default=None)
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="BIOSCAN-5M data directory (supervised_test.csv, unseen.csv).")
    p.add_argument("--taxon", type=str, default="genus",
                    help="Matches this project's other ZSC scripts (bioscan5m_barcodebert_local_sweep.sh,"
                    " bioscan5m_hyenadna_zsc.sh, etc.), which all use --taxon genus. Default: %(default)s")
    p.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=660)
    p.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors", type=int, default=15,
                    help="UMAP neighborhood size. Default: %(default)s")
    p.add_argument("--metric", default="cosine")
    p.add_argument("--n-clusters", "--n_clusters", dest="n_clusters", default=None, type=int)
    p.add_argument("--run-name", "--run_name", dest="run_name", default="zsc_external_barcodemamba")
    p.add_argument("--results-file", "--results_file", dest="results_file",
                    default="results_final/ZSC_external_RESULTS.txt")
    return p


def cli():
    config = get_parser().parse_args()
    run(config)


if __name__ == "__main__":
    cli()