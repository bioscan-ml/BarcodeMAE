#!/usr/bin/env python
"""
KNN evaluation with a RANDOMLY INITIALIZED encoder (maelm or encoder-only/transformer).

This script replicates knn_probing.py but skips loading a checkpoint.
The model weights are random — useful as a baseline to confirm that
pretrained models actually learn something meaningful.

--arch maelm mirrors the encoder used inside MAELMModel during MAE
pretraining (a plain BertModel, since the decoder never contributes to
downstream embeddings). --arch transformer mirrors the encoder-only
BertForTokenClassification used for vanilla (non-MAE) pretraining, with
its token-classification head stripped to nn.Identity.

Usage:
    python random_knn.py \
        --dataset BIOSCAN-5M \
        --data-dir ./BarcodeMAE/data/ \
        --arch maelm \
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
from transformers import BertConfig, BertForTokenClassification, BertModel

from barcodebert.datasets import KmerTokenizer, representations_from_df
from barcodebert.evaluation import knn_results_path, knn_vote


def build_random_encoder(
    vocab_size,
    n_layers,
    n_heads,
    hidden_size,
    max_position_embeddings,
    arch="maelm",
):
    """Build a randomly initialized encoder.

    arch="maelm" builds a plain BertModel, matching the encoder used inside
    MAELMModel (the decoder never contributes to downstream embeddings, so it
    is omitted here). arch="transformer" builds a BertForTokenClassification
    with its classification head stripped to nn.Identity, matching the
    encoder-only architecture used for vanilla (non-MAE) pretraining.
    """
    bert_config = BertConfig(
        vocab_size=vocab_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        hidden_size=hidden_size,
        output_hidden_states=True,
        max_position_embeddings=max_position_embeddings,
    )
    if arch == "maelm":
        model = BertModel(bert_config)
    elif arch == "transformer":
        model = BertForTokenClassification(bert_config)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Must be 'maelm' or 'transformer'.")
    model.eval()
    return model


def run(config):
    if config.knn_weights == "softmax" and config.metric != "cosine":
        raise ValueError(
            "--knn-weights=softmax requires --metric=cosine (it converts distance to "
            f"similarity via similarity = 1 - distance, which only holds for cosine distance; "
            f"got --metric={config.metric!r})"
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    print(f"\nConfiguration:\n{config}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    base_pairs = "ACGT"
    if config.use_cls_token:
        specials = ["[MASK]", "[UNK]", "[CLS]"]
    else:
        specials = ["[MASK]", "[UNK]"]

    from torchtext.vocab import vocab as build_vocab_from_dict

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
        f"Building randomly initialized {config.arch} encoder: "
        f"layers={config.n_layers}, heads={config.n_heads}, "
        f"hidden={config.encoder_embed_dim}, vocab={vocab_size}"
    )
    model = build_random_encoder(
        vocab_size=vocab_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        hidden_size=config.encoder_embed_dim,
        max_position_embeddings=max_position_embeddings,
        arch=config.arch,
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

    all_results = {}  # k -> {partition -> metrics}
    for k in config.n_neighbors:
        print(f"\n{'='*50}\nk = {k}\n{'='*50}")
        all_results[k] = {}
        for name, X_part, y_part in partitions:
            ind_k = neigh_ind[name][:, :k]
            dist_k = neigh_dist[name][:, :k]
            neighbor_labels = clf._y[ind_k]
            majority_idx = knn_vote(neighbor_labels, dist_k, weights=config.knn_weights, temperature=config.temperature)
            y_pred = clf.classes_[majority_idx]

            res = {
                "accuracy": 100.0 * sklearn.metrics.accuracy_score(y_part, y_pred),
                "accuracy-balanced": 100.0 * sklearn.metrics.balanced_accuracy_score(y_part, y_pred),
                "f1-micro": 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="micro"),
                "f1-macro": 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="macro"),
                "f1-support": 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="weighted"),
            }
            all_results[k][name] = res
            print(f"\n{name} (k={k}):")
            for metric_name, v in res.items():
                print(f"  {metric_name + ' ':.<24s} {v:6.2f} %")

    model_name = f"random_{config.arch}"
    results_file = knn_results_path(config.results_file, config.knn_weights)
    with open(results_file, "a") as f:
        for k, results in all_results.items():
            acc = results["Unseen"]["accuracy"]
            f.write(f"\n{config.run_name}_{model_name}_k{k}\t{acc:.4f}")

    total = time.time() - t0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    print(f"\nTotal time: {int(h)}:{int(m):02d}:{s:02.0f}")


def get_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="KNN evaluation with a randomly initialized (untrained) encoder."
    )
    # Dataset
    p.add_argument("--dataset", default="BIOSCAN-5M", choices=["BIOSCAN-5M", "CANADA-1.5M"])
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True)
    p.add_argument("--taxon", default="genus")

    # Model architecture
    p.add_argument(
        "--arch",
        default="maelm",
        choices=["maelm", "transformer"],
        help="maelm=plain BertModel (MAELM encoder), transformer=encoder-only "
             "BertForTokenClassification with head stripped. Default: %(default)s",
    )
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
        "--knn-weights", "--knn_weights", dest="knn_weights",
        default="uniform", choices=["uniform", "distance", "softmax"],
        help="Vote weighting for kNN label assignment. 'uniform': every neighbor gets one vote"
        " (plain majority vote). 'distance': neighbors weighted by 1/distance ('soft' kNN)."
        " 'softmax': neighbors weighted by softmax(similarity / --temperature), matching"
        " DINOv2's kNN eval; requires --metric=cosine. Default: %(default)s",
    )
    p.add_argument(
        "--temperature", dest="temperature", type=float, default=0.07,
        help="Temperature for --knn-weights=softmax (ignored otherwise). Lower is more"
        " winner-take-all, higher is closer to uniform voting. Default: %(default)s",
    )
    p.add_argument(
        "--representation-type",
        "--representation_type",
        dest="representation_type",
        default="tokens",
        choices=["tokens", "tokens_with_cls", "cls", "all_tokens"],
    )
    p.add_argument(
        "--run-name", "--run_name", dest="run_name", default="random_knn",
        help="Run name prefix for results file. Default: %(default)s",
    )
    p.add_argument(
        "--results-file", "--results_file", dest="results_file", default="RANDOM_KNN_RESULTS.txt",
        help="File to append KNN accuracy results to. Default: %(default)s",
    )

    return p


def cli():
    config = get_parser().parse_args()
    run(config)


if __name__ == "__main__":
    cli()