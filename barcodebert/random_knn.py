#!/usr/bin/env python
"""
KNN evaluation with a RANDOMLY INITIALIZED BertModel encoder.

This script replicates knn_probing.py but skips loading a checkpoint.
The model weights are random — useful as a baseline to confirm that
pretrained models actually learn something meaningful.

Usage:
    python random_knn.py \
        --dataset BIOSCAN-5M \
        --data-dir ./BarcodeMAE/data/ \
        --k-mer 6 \
        --n-layers 6 \
        --n-heads 6 \
        --encoder-embed-dim 768 \
        --taxon genus \
        --n-neighbors 1
"""

import os
import resource
import time
from itertools import product

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.optim
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchtext.vocab import vocab as build_vocab_from_dict
from transformers import BertConfig, BertModel

from barcodebert.datasets import KmerTokenizer, representations_from_df


def build_random_encoder(
    vocab_size,
    n_layers,
    n_heads,
    hidden_size,
    max_position_embeddings,
):
    """Build a randomly initialized BertModel (MAELM encoder architecture)."""
    bert_config = BertConfig(
        vocab_size=vocab_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        hidden_size=hidden_size,
        output_hidden_states=True,
        max_position_embeddings=max_position_embeddings,
    )
    model = BertModel(bert_config)
    model.eval()
    return model


def run(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    print(f"\nConfiguration:\n{config}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    base_pairs = "ACGT"
    if config.use_cls_token:
        specials = ["[MASK]", "[UNK]", "[CLS]"]
    else:
        specials = ["[MASK]", "[UNK]"]

    kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=config.k_mer)]
    kmer_dict = dict.fromkeys(kmers, 1)
    vocab = build_vocab_from_dict(kmer_dict, specials=specials)
    vocab.set_default_index(vocab["[UNK]"])
    tokenizer = KmerTokenizer(
        config.k_mer, vocab, stride=config.stride, padding=True, max_len=config.max_len
    )

    vocab_size = len(vocab)
    # max_position_embeddings must cover the full tokenized sequence length
    max_position_embeddings = config.max_len + 2  # small buffer

    # ── Random model ──────────────────────────────────────────────────────────
    print(
        f"Building randomly initialized BertModel: "
        f"layers={config.n_layers}, heads={config.n_heads}, "
        f"hidden={config.encoder_embed_dim}, vocab={vocab_size}"
    )
    model = build_random_encoder(
        vocab_size=vocab_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        hidden_size=config.encoder_embed_dim,
        max_position_embeddings=max_position_embeddings,
    )
    model = model.to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    df_train = pd.read_csv(os.path.join(config.data_dir, "supervised_train.csv"))
    df_test = pd.read_csv(os.path.join(config.data_dir, "unseen.csv"))

    if config.taxon.lower() == "bin":
        target_level = "bin_uri"
    elif config.dataset == "BIOSCAN-5M":
        target_level = config.taxon + "_index"
    elif config.dataset == "CANADA-1.5M":
        target_level = config.taxon + "_name"
    else:
        raise NotImplementedError(f"Unknown dataset: {config.dataset}")

    # ── Embeddings ────────────────────────────────────────────────────────────
    t0 = time.time()
    print("Generating embeddings for test set ...")
    X_unseen, y_unseen, _ = representations_from_df(
        df_test,
        target_level,
        model,
        tokenizer,
        config.dataset,
        mode="nonmask",
        mask_rate=0.0,
        representation_type=config.representation_type,
        use_cls_token=config.use_cls_token,
    )
    print("Generating embeddings for train set ...")
    X_train, y_train, _ = representations_from_df(
        df_train,
        target_level,
        model,
        tokenizer,
        config.dataset,
        mode="nonmask",
        mask_rate=0.0,
        representation_type=config.representation_type,
        use_cls_token=config.use_cls_token,
    )
    dt_embed = time.time() - t0
    print(f"Embeddings done in {dt_embed:.1f}s")

    # ── KNN ───────────────────────────────────────────────────────────────────
    y_train = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
    y_unseen = y_unseen.to_numpy() if hasattr(y_unseen, "to_numpy") else y_unseen

    max_k = max(config.n_neighbors)
    clf = KNeighborsClassifier(n_neighbors=max_k, metric=config.metric)
    clf.fit(X_train, y_train)

    neigh_dist, neigh_ind = {}, {}
    partitions = [("Train", X_train, y_train), ("Unseen", X_unseen, y_unseen)]
    for name, X_part, _ in partitions:
        neigh_dist[name], neigh_ind[name] = clf.kneighbors(X_part, n_neighbors=max_k)

    for k in config.n_neighbors:
        print(f"\n{'='*50}\nk = {k}\n{'='*50}")
        for name, X_part, y_part in partitions:
            ind_k = neigh_ind[name][:, :k]
            neighbor_labels = clf._y[ind_k]
            majority_idx = np.array([np.bincount(row).argmax() for row in neighbor_labels])
            y_pred = clf.classes_[majority_idx]

            acc = 100.0 * sklearn.metrics.accuracy_score(y_part, y_pred)
            bal = 100.0 * sklearn.metrics.balanced_accuracy_score(y_part, y_pred)
            f1_macro = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="macro")
            print(f"\n{name} (k={k}):")
            print(f"  accuracy ............. {acc:6.2f} %")
            print(f"  accuracy-balanced .... {bal:6.2f} %")
            print(f"  f1-macro ............. {f1_macro:6.2f} %")

    total = time.time() - t0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    print(f"\nTotal time: {int(h)}:{int(m):02d}:{s:02.0f}")


def get_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="KNN evaluation with a randomly initialized maelm-style BertModel encoder."
    )
    # Dataset
    p.add_argument("--dataset", default="BIOSCAN-5M", choices=["BIOSCAN-5M", "CANADA-1.5M"])
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True)
    p.add_argument("--taxon", default="genus")

    # Model architecture
    p.add_argument("--k-mer", "--k_mer", dest="k_mer", type=int, default=6)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-len", "--max_len", dest="max_len", type=int, default=660)
    p.add_argument("--n-layers", "--n_layers", dest="n_layers", type=int, default=6)
    p.add_argument("--n-heads", "--n_heads", dest="n_heads", type=int, default=6)
    p.add_argument(
        "--encoder-embed-dim", "--encoder_embed_dim", dest="encoder_embed_dim", type=int, default=768
    )
    p.add_argument("--use-cls-token", "--use_cls_token", dest="use_cls_token", action="store_true")

    # KNN
    p.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors", type=int, nargs="+", default=[1])
    p.add_argument("--metric", default="cosine")
    p.add_argument(
        "--representation-type",
        "--representation_type",
        dest="representation_type",
        default="tokens",
        choices=["tokens", "tokens_with_cls", "cls", "all_tokens"],
    )

    return p


def cli():
    config = get_parser().parse_args()
    run(config)


if __name__ == "__main__":
    cli()