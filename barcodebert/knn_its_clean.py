#!/usr/bin/env python
"""Leakage-free, level-aware KNN evaluation for ITS-5M.

Mirrors knn_its.py's checkpoint-loading and embedding-extraction, but fixes
two things knn_its.py gets wrong for a true generalization measurement:

  1. It doesn't exclude leaked query specimens — exact-sequence duplicates of
     training data, or same-physical-read-different-trim ("substring")
     duplicates — which inflate accuracy without testing generalization at
     all (see analyze_its_overlap.py for the audit that found this).
  2. It only ever evaluates at species level via DNADataset's *_labels.csv,
     whose 'species' column is a pre-factorized classification label: any
     species not in the training vocabulary collapses into one shared
     "unknown" bucket, so a species-novel specimen can never even be posed
     as a question, let alone answered.

This script instead:
  - Takes the *_tasks.csv files exported by `analyze_its_overlap.py
    --export-dir` and, for each test set, evaluates BOTH well-posed tasks in
    one pass (sharing embeddings — see below):
      species_level : query species IS in the training vocabulary (same
                       species, genuinely different individual — "identify a
                       new specimen of a known species").
      genus_level    : query species is novel, but its GENUS is in the
                        training vocabulary — species can't be predicted (no
                        gallery entry), genus can.
  - Builds the gallery AND labels directly from mycoai.data.Data's per-file
    UNITE-header parsing (same approach as analyze_its_overlap.py), not
    DNADataset/*_labels.csv, so genus-level evaluation doesn't hit the same
    vocabulary-collapse problem species-level did.
  - Embeds the ~5.2M-specimen gallery and each test set's query specimens
    ONLY ONCE each (embeddings don't depend on which label level you're
    evaluating, only the KNN fit does) — species-level and genus-level KNN
    classifiers are both fit on the same embeddings, just filtered/labelled
    differently. Doubling the (very expensive) gallery embedding pass to
    evaluate two tasks would be pure waste.
  - Supports both a real pretrained checkpoint AND a random-initialized
    encoder (--arch, no --pretrained-checkpoint), for the random baseline.

Usage:
    # Real checkpoint
    python knn_its_clean.py \
        --pretrained-checkpoint path/to/checkpoint_encoder.pt \
        --data-dir ./BarcodeMAE/data/ITS-5M \
        --tasks-dir ./BarcodeMAE/data/ITS-5M/tasks \
        --representation-type tokens \
        --run-name knnclean_myrun --results-file results_final/KNN_ITS_CLEAN_RESULTS.txt

    # Random baseline
    python knn_its_clean.py \
        --arch transformer --k-mer 6 --n-layers 6 --n-heads 6 --encoder-embed-dim 768 \
        --data-dir ./BarcodeMAE/data/ITS-5M \
        --tasks-dir ./BarcodeMAE/data/ITS-5M/tasks \
        --representation-type tokens \
        --run-name knnclean_random --results-file results_final/KNN_ITS_CLEAN_RESULTS.txt
"""

import argparse
import os
import resource
import time
from itertools import product

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from mycoai.data import Data
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchtext.vocab import vocab as build_vocab_from_dict
from tqdm import tqdm

from barcodebert import utils
from barcodebert.datasets import KmerTokenizer
from barcodebert.evaluation import knn_results_path, knn_vote
from barcodebert.io import load_pretrained_model
from barcodebert.random_knn import build_random_encoder

TEST_SETS = [
    ("Test1 (Yeast)", "test1"),
    ("Test2 (Filamentous)", "test2"),
    ("Test3 (MycoAI)", "test3"),
]
TASKS = ["species_level", "genus_level"]
UNKNOWN_STR = "?"


def extract_representations(sequences, model, tokenizer, representation_type, use_cls_token, device):
    """One-sequence-at-a-time embedding extraction — mirrors knn_its.py's
    extract_representations, but takes a plain sequence list (no labels) and
    returns embeddings in the same order, so callers can reuse them for
    multiple label levels without re-embedding."""
    embeddings = []

    with torch.no_grad():
        for seq in tqdm(sequences, desc=f"  embedding ({representation_type})", mininterval=10.0):
            x, att_mask = tokenizer(seq)

            if use_cls_token:
                cls_token = torch.tensor([2], dtype=x.dtype)
                cls_mask = torch.tensor([1], dtype=att_mask.dtype)
                x = torch.cat([cls_token, x])
                att_mask = torch.cat([cls_mask, att_mask])

            x = x.unsqueeze(0).to(device)
            att_mask = att_mask.unsqueeze(0).to(device)

            output = model(x, att_mask)

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

            if representation_type == "cls":
                embedding = hidden_states[:, 0, :]
            elif representation_type == "tokens":
                seq_mask = att_mask.clone()
                if use_cls_token:
                    seq_mask[:, 0] = 0
                sum_embeddings = (hidden_states * seq_mask.unsqueeze(-1)).sum(1)
                sum_mask = seq_mask.sum(1, keepdim=True)
                embedding = sum_embeddings / sum_mask
            elif representation_type == "tokens_with_cls":
                sum_embeddings = (hidden_states * att_mask.unsqueeze(-1)).sum(1)
                sum_mask = att_mask.sum(1, keepdim=True)
                embedding = sum_embeddings / sum_mask
            else:
                raise ValueError(f"Unsupported representation_type: {representation_type}")

            embeddings.append(embedding.cpu().numpy())

    X = np.squeeze(np.array(embeddings), 1)
    print(f"  {len(sequences)} samples | representation shape: {X.shape}")
    return X


def fit_knn(X_all, labels_col, max_k, metric):
    """Filter to rows with a resolved label, fit a KNeighborsClassifier."""
    known_mask = (labels_col != UNKNOWN_STR).to_numpy()
    X = X_all[known_mask]
    y = labels_col[known_mask].to_numpy()
    clf = KNeighborsClassifier(n_neighbors=max_k, metric=metric)
    clf.fit(X, y)
    return clf


def evaluate_task(clf, X_all, labels_col, task_mask, n_neighbors_list, weights="uniform", temperature=0.07):
    """Evaluate one (test set, task) combo: rows selected by task_mask,
    labelled by labels_col, against a KNN classifier already fit on the
    appropriate gallery."""
    mask = task_mask.to_numpy() & (labels_col != UNKNOWN_STR).to_numpy()
    X_query = X_all[mask]
    y_query = labels_col[mask].to_numpy()
    if len(y_query) == 0:
        return {}

    gallery_n = len(clf._y)
    gallery_labels = set(clf.classes_.tolist())
    test_labels = set(np.unique(y_query).tolist())
    label_overlap = test_labels & gallery_labels
    overlap_pct = 100.0 * len(label_overlap) / len(test_labels) if test_labels else float("nan")
    print(f"    gallery: {gallery_n} samples, {len(gallery_labels)} unique labels | "
          f"query: {len(y_query)} samples, {len(test_labels)} unique labels | "
          f"label overlap: {len(label_overlap)}/{len(test_labels)} ({overlap_pct:.1f}%)")
    if overlap_pct < 100.0:
        missing = test_labels - gallery_labels
        print(f"    WARNING: {len(missing)} query label(s) not in gallery (query can never be correct "
              f"for these) — task construction should guarantee 100%, this indicates a bug: {sorted(missing)[:10]}")

    max_k = max(n_neighbors_list)
    neigh_dist, neigh_ind = clf.kneighbors(X_query, n_neighbors=max_k)
    neighbor_labels = clf._y[neigh_ind]

    results = {}
    for k in n_neighbors_list:
        majority_idx = knn_vote(neighbor_labels[:, :k], neigh_dist[:, :k], weights=weights, temperature=temperature)
        y_pred = clf.classes_[majority_idx]
        results[k] = {
            "count": len(y_query),
            "accuracy": 100.0 * sklearn.metrics.accuracy_score(y_query, y_pred),
            "accuracy-balanced": 100.0 * sklearn.metrics.balanced_accuracy_score(y_query, y_pred),
            "f1-macro": 100.0 * sklearn.metrics.f1_score(y_query, y_pred, average="macro"),
        }
    return results


def run(config):
    if config.knn_weights == "softmax" and config.metric != "cosine":
        raise ValueError(
            "--knn-weights=softmax requires --metric=cosine (it converts distance to "
            f"similarity via similarity = 1 - distance, which only holds for cosine distance; "
            f"got --metric={config.metric!r})"
        )

    t_start = time.time()
    if config.log_wandb:
        import wandb

    print("\nConfiguration:\n")
    print(config)
    print(f"\nFound {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.", flush=True)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    results_file = knn_results_path(config.results_file, config.knn_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    if config.pretrained_checkpoint_path:
        model, pre_checkpoint = load_pretrained_model(config.pretrained_checkpoint_path, device=device)
        model.classifier = nn.Identity()
        model = model.to(device)
        model.eval()
        pre_config = pre_checkpoint["config"]
        k_mer = getattr(pre_config, "k_mer", 6)
        stride = getattr(pre_config, "stride", k_mer)
        max_len = getattr(pre_config, "max_len", 660)
        use_cls = getattr(pre_config, "use_cls_token", False)
    else:
        model = None  # built below, once the tokenizer's vocab size is known
        k_mer, stride, max_len = config.k_mer, config.stride, config.max_len
        use_cls = config.use_cls_token

    print(f"\nk_mer={k_mer}, stride={stride}, max_len={max_len}, use_cls_token={use_cls}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    base_pairs = "ACGT"
    specials = ["[MASK]", "[UNK]", "[CLS]"] if use_cls else ["[MASK]", "[UNK]"]
    kmers = ["".join(k) for k in product(base_pairs, repeat=k_mer)]
    kmer_dict = dict.fromkeys(kmers, 1)
    vocab = build_vocab_from_dict(kmer_dict, specials=specials)
    vocab.set_default_index(vocab["[UNK]"])
    tokenizer = KmerTokenizer(k_mer, vocab, stride=stride, padding=True, max_len=max_len)

    if model is None:
        model = build_random_encoder(
            vocab_size=len(vocab), n_layers=config.n_layers, n_heads=config.n_heads,
            hidden_size=config.encoder_embed_dim, max_position_embeddings=max_len + 2, arch=config.arch,
        ).to(device)
        model.eval()

    # ── Gallery: embed once, fit two classifiers (species, genus) ─────────────
    print(f"\nLoading gallery (trainset)...")
    gallery_df_raw = Data(os.path.join(config.data_dir, "trainset.fasta"), allow_duplicates=True).data
    # Drop rows useless for BOTH classifiers up front — no point embedding a
    # specimen with neither species nor genus resolved (fit_knn would filter
    # it out downstream anyway, but only after paying for the embedding).
    is_usable = (gallery_df_raw["species"] != UNKNOWN_STR) | (gallery_df_raw["genus"] != UNKNOWN_STR)
    gallery_df = gallery_df_raw[is_usable].reset_index(drop=True)
    print(f"  {len(gallery_df_raw)} raw specimens -> {len(gallery_df)} with species and/or genus resolved "
          f"({len(gallery_df_raw) - len(gallery_df)} dropped, useless for either classifier)")

    print("Extracting gallery embeddings (once, reused for both label levels)...")
    X_gallery = extract_representations(
        gallery_df["sequence"].tolist(), model, tokenizer, config.representation_type, use_cls, device
    )

    print("Fitting species-level and genus-level KNN classifiers...", flush=True)
    max_k = max(config.n_neighbors)
    clf_species = fit_knn(X_gallery, gallery_df["species"], max_k, config.metric)
    clf_genus = fit_knn(X_gallery, gallery_df["genus"], max_k, config.metric)
    print(f"  species gallery: {len(clf_species._y)} specimens, {len(clf_species.classes_)} classes")
    print(f"  genus gallery:   {len(clf_genus._y)} specimens, {len(clf_genus.classes_)} classes")

    # ── Query per test set: embed once, evaluate both tasks ───────────────────
    # Results are saved incrementally (right after each test set), not batched
    # to the end — a later test set failing (e.g. malformed ids in an external
    # benchmark file) must not lose results already computed for earlier ones.
    model_name = os.path.basename(config.pretrained_checkpoint_path) if config.pretrained_checkpoint_path else f"random_{config.arch}"
    all_results = {}  # all_results[task][k][test_name] = metrics dict, for wandb logging at the end
    for name, tag in TEST_SETS:
        tasks_df = pd.read_csv(os.path.join(config.tasks_dir, f"{tag}_tasks.csv"))
        keep_ids = set(tasks_df.loc[tasks_df["task"].isin(TASKS), "id"])
        if not keep_ids:
            print(f"\n{name}: 0 query specimens across both tasks — skipping (no fasta load needed)")
            continue

        test_df = Data(os.path.join(config.data_dir, f"{tag}.fasta"), allow_duplicates=False).data
        # id -> task via boolean masks + .isin(), NOT .map()/reindex: robust to
        # duplicate or NaN ids in the source fasta (seen in practice for the
        # MycoAI benchmark file), since .isin() only checks membership by value.
        species_ids = set(tasks_df.loc[tasks_df["task"] == "species_level", "id"])
        genus_ids = set(tasks_df.loc[tasks_df["task"] == "genus_level", "id"])
        test_df = test_df.copy()
        test_df["task"] = np.where(
            test_df["id"].isin(species_ids), "species_level",
            np.where(test_df["id"].isin(genus_ids), "genus_level", "other"),
        )

        relevant = test_df[test_df["task"].isin(TASKS)].reset_index(drop=True)
        print(f"\n{name}: {len(relevant)} query specimens across both tasks "
              f"({(relevant['task'] == 'species_level').sum()} species_level, "
              f"{(relevant['task'] == 'genus_level').sum()} genus_level)")
        if len(relevant) == 0:
            continue

        X_query = extract_representations(
            relevant["sequence"].tolist(), model, tokenizer, config.representation_type, use_cls, device
        )

        tag_lower = name.split()[0].lower()
        for task in TASKS:
            print(f"  --- {task} ---")
            clf = clf_species if task == "species_level" else clf_genus
            label_col = relevant["species"] if task == "species_level" else relevant["genus"]
            task_mask = relevant["task"] == task
            res_by_k = evaluate_task(clf, X_query, label_col, task_mask, config.n_neighbors,
                                      weights=config.knn_weights, temperature=config.temperature)

            # Save + log each (task, k) result as soon as it's computed — don't
            # wait for the rest of this test set, let alone the other test sets.
            with open(results_file, "a") as f:
                for k, res in res_by_k.items():
                    all_results.setdefault(task, {}).setdefault(k, {})[name] = res
                    print(f"  [{task}] k={k}: accuracy={res['accuracy']:.2f}% "
                          f"balanced={res['accuracy-balanced']:.2f}% f1-macro={res['f1-macro']:.2f}% "
                          f"(n={res['count']})")
                    f.write(f"\n{config.run_name}_{task}_{model_name}_{tag_lower}_k{k}\t{res['accuracy']:.4f}")
        print(f"  -> saved {name} results to {results_file}")

    dt_total = time.time() - t_start
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"\nFinished in {dt_total/60:.1f} min | Max memory: {mem:.1f} GB")

    if config.log_wandb:
        wandb.init(name=config.run_name, project=config.wandb_project, config=vars(config), job_type="knn_its_clean")
        log_dict = {}
        for task, by_k in all_results.items():
            for k, results in by_k.items():
                for name, res in results.items():
                    for metric, v in res.items():
                        log_dict[f"{task}/knn_k{k}/{name}/{metric}"] = v
        wandb.log(log_dict)


def get_parser():
    p = argparse.ArgumentParser(description="Leakage-free, level-aware KNN evaluation for ITS-5M.")
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="ITS-5M data directory (trainset.fasta, test1-3.fasta).")
    p.add_argument("--tasks-dir", "--tasks_dir", dest="tasks_dir", required=True,
                    help="Directory containing test{1,2,3}_tasks.csv from analyze_its_overlap.py --export-dir.")

    group = p.add_argument_group("Model (pretrained checkpoint)")
    group.add_argument("--pretrained-checkpoint", "--pretrained_checkpoint", dest="pretrained_checkpoint_path",
                        default=None, help="Path to pretrained encoder checkpoint. Omit for random-init baseline.")

    group = p.add_argument_group("Model (random-init baseline — used when --pretrained-checkpoint is omitted)")
    group.add_argument("--arch", default="transformer", choices=["maelm", "transformer"])
    group.add_argument("--k-mer", "--k_mer", dest="k_mer", type=int, default=6)
    group.add_argument("--stride", type=int, default=6)
    group.add_argument("--max-len", "--max_len", dest="max_len", type=int, default=660)
    group.add_argument("--n-layers", "--n_layers", dest="n_layers", type=int, default=6)
    group.add_argument("--n-heads", "--n_heads", dest="n_heads", type=int, default=6)
    group.add_argument("--encoder-embed-dim", "--encoder_embed_dim", dest="encoder_embed_dim", type=int, default=768)
    group.add_argument("--use-cls-token", "--use_cls_token", dest="use_cls_token", action="store_true")

    group = p.add_argument_group("KNN")
    group.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors", default=[1, 3, 5, 7],
                        type=int, nargs="+")
    group.add_argument("--metric", default="cosine")
    group.add_argument("--knn-weights", "--knn_weights", dest="knn_weights",
                        default="uniform", choices=["uniform", "distance", "softmax"],
                        help="Vote weighting for kNN label assignment. 'uniform': every neighbor"
                        " gets one vote. 'distance': neighbors weighted by 1/distance ('soft' kNN)."
                        " 'softmax': neighbors weighted by softmax(similarity / --temperature),"
                        " matching DINOv2's kNN eval; requires --metric=cosine.")
    group.add_argument("--temperature", dest="temperature", type=float, default=0.07,
                        help="Temperature for --knn-weights=softmax (ignored otherwise). Lower is"
                        " more winner-take-all, higher is closer to uniform voting.")
    group.add_argument("--representation-type", "--representation_type", dest="representation_type",
                        default="tokens", choices=["tokens", "cls", "tokens_with_cls"])

    group = p.add_argument_group("Run / logging")
    group.add_argument("--run-name", "--run_name", dest="run_name", default="knn_its_clean")
    group.add_argument("--results-file", "--results_file", dest="results_file",
                        default="KNN_ITS_CLEAN_RESULTS.txt")
    group.add_argument("--log-wandb", "--log_wandb", dest="log_wandb", action="store_true", default=False)
    group.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="barcodemae_cls")
    group.add_argument("--seed", default=None, type=int)

    return p


def cli():
    config = get_parser().parse_args()
    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)
    run(config)


if __name__ == "__main__":
    cli()
