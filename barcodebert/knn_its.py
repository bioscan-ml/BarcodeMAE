#!/usr/bin/env python
"""KNN evaluation for ITS-5M using pretrained encoder representations.

Mirrors knn_probing.py exactly but loads data via DNADataset (ITS-5M format)
and evaluates on all three ITS test sets (Yeast, Filamentous, MycoAI).
"""

import argparse
import os
import resource
import time
from itertools import product

import numpy as np
import sklearn.metrics
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchtext.vocab import vocab as build_vocab_from_dict

from barcodebert import utils
from barcodebert.datasets import DNADataset, KmerTokenizer
from barcodebert.io import load_pretrained_model


DATASET = "ITS-5M"
TEST_SETS = [
    ("Test1 (Yeast)",       "test1.fasta"),
    ("Test2 (Filamentous)", "test2.fasta"),
    ("Test3 (MycoAI)",      "test3.fasta"),
]


def extract_representations(dataset, model, tokenizer, representation_type, use_cls_token, device):
    """Extract embeddings from a DNADataset, skipping unknown-label samples (-1).

    Embedding logic mirrors representations_from_df in datasets.py exactly.
    """
    embeddings = []
    valid_labels = []

    with torch.no_grad():
        for barcode, label in zip(dataset.barcodes, dataset.labels):
            if label == -1:
                continue

            x, att_mask = tokenizer(barcode)

            if use_cls_token:
                cls_token = torch.tensor([2], dtype=x.dtype)
                cls_mask = torch.tensor([1], dtype=att_mask.dtype)
                x = torch.cat([cls_token, x])
                att_mask = torch.cat([cls_mask, att_mask])

            x = x.unsqueeze(0).to(device)
            att_mask = att_mask.unsqueeze(0).to(device)

            output = model(x, att_mask)

            if representation_type == "cls":
                if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                    hidden_states = output.last_hidden_state
                elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hidden_states = output.hidden_states
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]
                else:
                    hidden_states = output[-1] if isinstance(output, tuple) else output
                embedding = hidden_states[:, 0, :]

            elif representation_type == "tokens":
                if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                    hidden_states = output.last_hidden_state
                elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hidden_states = output.hidden_states
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]
                else:
                    hidden_states = output[-1] if isinstance(output, tuple) else output
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]
                seq_mask = att_mask.clone()
                if use_cls_token:
                    seq_mask[:, 0] = 0
                sum_embeddings = (hidden_states * seq_mask.unsqueeze(-1)).sum(1)
                sum_mask = seq_mask.sum(1, keepdim=True)
                embedding = sum_embeddings / sum_mask

            elif representation_type == "tokens_with_cls":
                if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                    hidden_states = output.last_hidden_state
                elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hidden_states = output.hidden_states
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]
                else:
                    hidden_states = output[-1] if isinstance(output, tuple) else output
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]
                sum_embeddings = (hidden_states * att_mask.unsqueeze(-1)).sum(1)
                sum_mask = att_mask.sum(1, keepdim=True)
                embedding = sum_embeddings / sum_mask

            else:
                raise ValueError(
                    f"Unsupported representation_type: {representation_type}. "
                    "Choose from: tokens, cls, tokens_with_cls"
                )

            embeddings.append(embedding.cpu().numpy())
            valid_labels.append(label)

    X = np.squeeze(np.array(embeddings), 1)
    y = np.array(valid_labels)
    print(f"  {len(y)} samples | representation shape: {X.shape}")
    return X, y


def run(config):
    t_start = time.time()

    if config.log_wandb:
        import wandb

    print("\nConfiguration:\n")
    print(config)
    print(f"\nFound {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.", flush=True)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    model, pre_checkpoint = load_pretrained_model(config.pretrained_checkpoint_path, device=device)
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()

    pre_config = pre_checkpoint["config"]
    k_mer       = getattr(pre_config, "k_mer", 6)
    stride      = getattr(pre_config, "stride", k_mer)
    max_len     = getattr(pre_config, "max_len", 660)
    use_cls     = getattr(pre_config, "use_cls_token", False)

    print(f"\nCheckpoint: k_mer={k_mer}, stride={stride}, max_len={max_len}, use_cls_token={use_cls}")

    # ── Build tokenizer (same as knn_probing.py) ──────────────────────────────
    base_pairs = "ACGT"
    specials = ["[MASK]", "[UNK]", "[CLS]"] if use_cls else ["[MASK]", "[UNK]"]
    kmers = ["".join(k) for k in product(base_pairs, repeat=k_mer)]
    kmer_dict = dict.fromkeys(kmers, 1)
    vocab = build_vocab_from_dict(kmer_dict, specials=specials)
    vocab.set_default_index(vocab["[UNK]"])
    tokenizer = KmerTokenizer(k_mer, vocab, stride=stride, padding=True, max_len=max_len)

    # ── Load train set (gallery) ──────────────────────────────────────────────
    print("\nLoading train set (gallery)...")
    train_path = os.path.join(config.data_dir, "trainset.fasta")
    ds_train = DNADataset(
        file_path=train_path,
        k_mer=k_mer, stride=stride, max_len=max_len,
        dataset_format=DATASET, taxonomic_level="species",
        use_cls_token=use_cls, filter_unknown_labels=True,
    )
    label2id = ds_train.label2id

    print(f"  {len(ds_train.barcodes)} train sequences | {ds_train.num_labels} species classes")

    t_embed = time.time()
    print("Extracting train embeddings...")
    X_train, y_train = extract_representations(ds_train, model, tokenizer, config.representation_type, use_cls, device)

    # ── Load and embed all 3 test sets ────────────────────────────────────────
    test_data = {}
    for name, fname in TEST_SETS:
        fpath = os.path.join(config.data_dir, fname)
        print(f"\nLoading {name} ({fname})...")
        ds_test = DNADataset(
            file_path=fpath,
            k_mer=k_mer, stride=stride, max_len=max_len,
            dataset_format=DATASET, taxonomic_level="species",
            label2id=label2id,
            use_cls_token=use_cls, filter_unknown_labels=True,
        )
        print(f"Extracting embeddings for {name}...")
        X_test, y_test = extract_representations(ds_test, model, tokenizer, config.representation_type, use_cls, device)
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
    model_name = os.path.join(*os.path.split(config.pretrained_checkpoint_path)[-2:])
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
            job_type="knn_its",
        )
        log_dict = {}
        for k, results in all_results.items():
            for name, res in results.items():
                for metric, v in res.items():
                    log_dict[f"knn_k{k}/{name}/{metric}"] = v
        wandb.log(log_dict)


def get_parser():
    parser = argparse.ArgumentParser(description="KNN evaluation for ITS-5M pretrained encoders.")

    parser.add_argument("--pretrained-checkpoint", "--pretrained_checkpoint",
                        dest="pretrained_checkpoint_path", required=True,
                        help="Path to pretrained encoder checkpoint.")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                        help="Path to ITS-5M data directory (containing trainset.fasta, test1-3.fasta).")
    parser.add_argument("--run-name", "--run_name", dest="run_name", default="knn_its",
                        help="Run name prefix for results file.")
    parser.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors",
                        default=[1, 3, 5, 7], type=int, nargs="+",
                        help="KNN neighbor counts. Default: 1 3 5 7")
    parser.add_argument("--metric", default="cosine",
                        help="Distance metric for KNN. Default: cosine")
    parser.add_argument("--representation-type", "--representation_type", dest="representation_type",
                        default="tokens", choices=["tokens", "cls", "tokens_with_cls"],
                        help="Representation type. Default: tokens")
    parser.add_argument("--results-file", "--results_file", dest="results_file",
                        default="KNN_ITS_RESULTS.txt",
                        help="File to append results to. Default: KNN_ITS_RESULTS.txt")
    parser.add_argument("--log-wandb", "--log_wandb", dest="log_wandb",
                        action="store_true", default=False)
    parser.add_argument("--wandb-project", "--wandb_project", dest="wandb_project",
                        default="barcodemae_cls")
    parser.add_argument("--seed", default=None, type=int)

    return parser


def cli():
    parser = get_parser()
    config = parser.parse_args()
    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)
    run(config)


if __name__ == "__main__":
    cli()
