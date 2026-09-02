#!/usr/bin/env python
"""Held-out (seen-to-seen) genus KNN sanity check for a BarcodeMamba/BarcodeMamba+
checkpoint on BIOSCAN-5M's supervised_train.csv.

Purpose: validate that a checkpoint's embeddings carry real genus signal,
without touching the seen-to-unseen generalization gap or any cross-file
label-index alignment issues. Splits supervised_train.csv in half, fits KNN
on half A, predicts genus for half B -- both halves use the SAME genus_index
column, so there is no label-encoding mismatch risk (unlike comparing against
externally precomputed embedding caches with independently-encoded indices).

Usage:
    python held_out_sanity_check.py \
        --barcodemamba-repo /scratch/$USER/BarcodeMamba-dev \
        --checkpoint-dir /scratch/$USER/barcodemamba_checkpoints/BarcodeMamba-plus-BIOSCAN-5M \
        --bpe-tokenizer-path /scratch/$USER/barcodemamba_checkpoints/BarcodeMamba-plus-BIOSCAN-5M/bpe_tokenizer.pkl \
        --data-dir ./BarcodeMAE/data/BIOSCAN-5M \
        --save-embeddings-prefix /scratch/$USER/train_ours
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, ".")
from barcodebert.barcodemamba_common import embed_sequences, load_barcodemamba, load_bpe_tokenizer


def run(config):
    model, bm_config = load_barcodemamba(config.barcodemamba_repo, config.checkpoint_dir, config.checkpoint_name)
    tokenizer_name = bm_config.tokenizer.name
    print(f"Tokenizer: {tokenizer_name}")
    if tokenizer_name == "bpe":
        tokenizer = load_bpe_tokenizer(config.bpe_tokenizer_path)
    else:
        from utils.ssm_dataset import get_tokenizer

        tokenizer = get_tokenizer(tokenizer_name, bm_config.tokenizer)

    model.cuda()
    model.eval()

    df = pd.read_csv(os.path.join(config.data_dir, "supervised_train.csv"))
    print(f"Extracting embeddings for {len(df)} specimens...")
    X = embed_sequences(model, tokenizer, tokenizer_name, df["nucleotides"], config.max_length)
    y_genus = df["genus_index"].to_numpy()

    if config.save_embeddings_prefix:
        np.save(f"{config.save_embeddings_prefix}_feat.npy", X)
        np.save(f"{config.save_embeddings_prefix}_genus.npy", y_genus)
        print(f"Saved embeddings to {config.save_embeddings_prefix}_feat.npy / _genus.npy")

    rng = np.random.RandomState(config.seed)
    idx = rng.permutation(len(X))
    half = len(idx) // 2
    idx_a, idx_b = idx[:half], idx[half:]
    X_a, y_a = X[idx_a], y_genus[idx_a]
    X_b, y_b = X[idx_b], y_genus[idx_b]

    not_in_a = len(set(y_b.tolist()) - set(y_a.tolist()))
    print(f"genus in half B not in half A: {not_in_a} / {len(set(y_b.tolist()))} unique")

    for k in config.n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=k, metric=config.metric)
        clf.fit(X_a, y_a)
        y_pred = clf.predict(X_b)
        acc = 100.0 * sklearn.metrics.accuracy_score(y_b, y_pred)
        print(f"k={k}: held-out (seen-to-seen) genus accuracy = {acc:.2f}%")


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--barcodemamba-repo", "--barcodemamba_repo", dest="barcodemamba_repo", required=True)
    p.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", required=True)
    p.add_argument("--checkpoint-name", "--checkpoint_name", dest="checkpoint_name", default=None)
    p.add_argument("--bpe-tokenizer-path", "--bpe_tokenizer_path", dest="bpe_tokenizer_path", default=None)
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="BIOSCAN-5M data directory (supervised_train.csv).")
    p.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=660)
    p.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors", default=[1, 5], type=int, nargs="+")
    p.add_argument("--metric", default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-embeddings-prefix", "--save_embeddings_prefix", dest="save_embeddings_prefix",
                    default=None, help="If set, saves <prefix>_feat.npy and <prefix>_genus.npy.")
    return p


def cli():
    config = get_parser().parse_args()
    run(config)


if __name__ == "__main__":
    cli()