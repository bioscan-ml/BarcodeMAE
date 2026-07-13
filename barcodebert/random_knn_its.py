#!/usr/bin/env python
"""KNN evaluation on ITS-5M (fungi) with a RANDOMLY INITIALIZED encoder.

Mirrors knn_its.py but skips loading a pretrained checkpoint — the encoder
weights are random. Useful as a baseline to confirm that pretrained models
actually learn something meaningful on the fungi test sets (Yeast,
Filamentous, MycoAI).

--arch maelm builds a plain BertModel, matching the encoder used inside
MAELMModel (the decoder never contributes to downstream embeddings, so it is
omitted here). --arch transformer builds a BertForTokenClassification with
its head stripped to nn.Identity, matching the encoder-only architecture
used for vanilla (non-MAE) pretraining. See random_knn.py for the BIOSCAN-5M
counterpart of this script.

Usage:
    python random_knn_its.py \
        --data-dir ./BarcodeMAE/data/ITS-5M \
        --arch maelm \
        --k-mer 6 \
        --n-layers 6 \
        --n-heads 6 \
        --encoder-embed-dim 768 \
        --n-neighbors 1 3 5 7
"""

import os
import resource
import time
from itertools import product

import numpy as np
import sklearn.metrics
import torch
from sklearn.neighbors import KNeighborsClassifier
from torchtext.vocab import vocab as build_vocab_from_dict

from barcodebert import utils
from barcodebert.datasets import DNADataset, KmerTokenizer
from barcodebert.knn_its import DATASET, TEST_SETS, extract_representations
from barcodebert.random_knn import build_random_encoder


def run(config):
    t_start = time.time()

    if config.log_wandb:
        import wandb

    print("\nConfiguration:\n")
    print(config)
    print(f"\nFound {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.", flush=True)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # ── Build tokenizer ────────────────────────────────────────────────────────
    base_pairs = "ACGT"
    specials = ["[MASK]", "[UNK]", "[CLS]"] if config.use_cls_token else ["[MASK]", "[UNK]"]
    kmers = ["".join(k) for k in product(base_pairs, repeat=config.k_mer)]
    kmer_dict = dict.fromkeys(kmers, 1)
    vocab = build_vocab_from_dict(kmer_dict, specials=specials)
    vocab.set_default_index(vocab["[UNK]"])
    tokenizer = KmerTokenizer(config.k_mer, vocab, stride=config.stride, padding=True, max_len=config.max_len)

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
    model.eval()

    # ── Load train set (gallery) ──────────────────────────────────────────────
    print("\nLoading train set (gallery)...")
    train_path = os.path.join(config.data_dir, "trainset.fasta")
    ds_train = DNADataset(
        file_path=train_path,
        k_mer=config.k_mer, stride=config.stride, max_len=config.max_len,
        dataset_format=DATASET, taxonomic_level="species",
        use_cls_token=config.use_cls_token, filter_unknown_labels=True,
    )
    label2id = ds_train.label2id

    print(f"  {len(ds_train.barcodes)} train sequences | {ds_train.num_labels} species classes")

    t_embed = time.time()
    print("Extracting train embeddings...")
    X_train, y_train = extract_representations(
        ds_train, model, tokenizer, config.representation_type, config.use_cls_token, device
    )

    # ── Load and embed all 3 test sets ────────────────────────────────────────
    test_data = {}
    for name, fname in TEST_SETS:
        fpath = os.path.join(config.data_dir, fname)
        print(f"\nLoading {name} ({fname})...")
        ds_test = DNADataset(
            file_path=fpath,
            k_mer=config.k_mer, stride=config.stride, max_len=config.max_len,
            dataset_format=DATASET, taxonomic_level="species",
            label2id=label2id,
            use_cls_token=config.use_cls_token, filter_unknown_labels=True,
        )
        print(f"Extracting embeddings for {name}...")
        X_test, y_test = extract_representations(
            ds_test, model, tokenizer, config.representation_type, config.use_cls_token, device
        )
        test_data[name] = (X_test, y_test)

    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    dt = time.time() - t_embed
    print(f"\nEmbedding time: {dt:.0f}s | Max memory: {mem:.1f} GB")

    # ── KNN (fit once, evaluate all k) ────────────────────────────────────────
    print("\nFitting KNN...", flush=True)
    max_k = max(config.n_neighbors)
    clf = KNeighborsClassifier(n_neighbors=max_k, metric=config.metric)
    clf.fit(X_train, y_train)

    all_results = {}
    for k in config.n_neighbors:
        print(f"\n{'='*50}\nk = {k}\n{'='*50}")
        all_results[k] = {}
        for name, (X_test, y_test) in test_data.items():
            neigh_dist, neigh_ind = clf.kneighbors(X_test, n_neighbors=k)
            neighbor_labels = clf._y[neigh_ind]
            majority_idx = np.array([np.bincount(row).argmax() for row in neighbor_labels])
            y_pred = clf.classes_[majority_idx]

            res = {
                "count":             len(y_test),
                "accuracy":          100.0 * sklearn.metrics.accuracy_score(y_test, y_pred),
                "accuracy-balanced": 100.0 * sklearn.metrics.balanced_accuracy_score(y_test, y_pred),
                "f1-micro":          100.0 * sklearn.metrics.f1_score(y_test, y_pred, average="micro"),
                "f1-macro":          100.0 * sklearn.metrics.f1_score(y_test, y_pred, average="macro"),
                "f1-support":        100.0 * sklearn.metrics.f1_score(y_test, y_pred, average="weighted"),
            }
            all_results[k][name] = res

            print(f"\n{name} (k={k}):")
            for metric, v in res.items():
                if metric == "count":
                    print(f"  {metric + ' ':.<21s}{v:7d}")
                else:
                    print(f"  {metric + ' ':.<24s} {v:6.2f} %")

    # ── Save results ──────────────────────────────────────────────────────────
    model_name = f"random_{config.arch}"
    with open(config.results_file, "a") as f:
        for k, results in all_results.items():
            for name, res in results.items():
                tag = name.split()[0].lower()  # test1, test2, test3
                f.write(f"\n{config.run_name}_{model_name}_{tag}_k{k}\t{res['accuracy']:.4f}")

    dt_total = time.time() - t_start
    h, rem = divmod(int(dt_total), 3600)
    m, s = divmod(rem, 60)
    print(f"\nFinished in {h}:{m:02d}:{s:02d}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    if config.log_wandb:
        wandb.init(
            name=config.run_name,
            project=config.wandb_project,
            config=vars(config),
            job_type="random_knn_its",
        )
        log_dict = {}
        for k, results in all_results.items():
            for name, res in results.items():
                for metric, v in res.items():
                    log_dict[f"knn_k{k}/{name}/{metric}"] = v
        wandb.log(log_dict)


def get_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="KNN evaluation on ITS-5M (fungi) with a randomly initialized (untrained) encoder."
    )
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="Path to ITS-5M data directory (containing trainset.fasta, test1-3.fasta).")

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
    p.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors",
                    default=[1, 3, 5, 7], type=int, nargs="+",
                    help="KNN neighbor counts. Default: 1 3 5 7")
    p.add_argument("--metric", default="cosine")
    p.add_argument("--representation-type", "--representation_type", dest="representation_type",
                    default="tokens", choices=["tokens", "cls", "tokens_with_cls"],
                    help="Representation type. Default: tokens")

    # Run / logging
    p.add_argument("--run-name", "--run_name", dest="run_name", default="random_knn_its",
                    help="Run name prefix for results file. Default: %(default)s")
    p.add_argument("--results-file", "--results_file", dest="results_file",
                    default="RANDOM_KNN_ITS_RESULTS.txt",
                    help="File to append results to. Default: %(default)s")
    p.add_argument("--log-wandb", "--log_wandb", dest="log_wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="barcodemae_cls")
    p.add_argument("--seed", default=None, type=int)

    return p


def cli():
    config = get_parser().parse_args()
    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)
    run(config)


if __name__ == "__main__":
    cli()