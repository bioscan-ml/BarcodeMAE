#!/usr/bin/env python
"""Leakage-free, level-aware KNN evaluation for ITS-5M using a pretrained
MycoAI-BERT or MycoAI-CNN checkpoint (https://zenodo.org/records/10904344).

Mirrors knn_its_clean.py's gallery/query task-filtering, KNN fitting, and
voting exactly (same *_tasks.csv files, same species_level/genus_level task
definitions, same knn_vote/results-file conventions) -- see that script's
docstring for the full leakage-filtering rationale. The only thing that
differs is embedding extraction: a MycoAI checkpoint is a full
torch.load'd `mycoai.modules.seq_class_network.SeqClassNetwork` object (not
a HuggingFace AutoModel), with its own `.latent_space(data_obj)` method that
internally tokenizes, batches, and pools -- so this script builds a
`mycoai.data.Data` object per gallery/query subset (by constructing it
normally from the real fasta, then overwriting `.data` with our
leakage-filtered subset) and calls `.latent_space()` on it directly, instead
of knn_its_clean.py's tokenizer(seq)->model(ids,mask) per-sequence loop.

Usage:
    python knn_its_mycoai.py \
        --checkpoint /scratch/$USER/mycoai_models/MycoAI-BERT.pt \
        --data-dir ./BarcodeMAE/data/ITS-5M \
        --tasks-dir ./BarcodeMAE/data/ITS-5M/tasks \
        --run-name mycoai_bert --results-file results_final/KNN_ITS_external_RESULTS.txt
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from mycoai.data import Data
from sklearn.neighbors import KNeighborsClassifier

from barcodebert.evaluation import knn_results_path, knn_vote

TEST_SETS = [
    ("Test1 (Yeast)", "test1"),
    ("Test2 (Filamentous)", "test2"),
]
ALL_TASKS = ["species_level", "genus_level"]
UNKNOWN_STR = "?"


def embed(model, fasta_path, df):
    """df: a subset of a mycoai.data.Data's .data (id, phylum..species,
    sequence columns), in the row order embeddings should be returned in.
    Constructs a real Data object from fasta_path (for any non-.data state
    encode_dataset/latent_space might rely on) then swaps in our filtered
    subset before embedding, so we embed only the leakage-filtered rows,
    not the full file."""
    data_obj = Data(fasta_path, allow_duplicates=True)
    data_obj.data = df.reset_index(drop=True)
    return model.latent_space(data_obj)


def fit_knn(X_all, labels_col, max_k, metric):
    known_mask = (labels_col != UNKNOWN_STR).to_numpy()
    X = X_all[known_mask]
    y = labels_col[known_mask].to_numpy()
    clf = KNeighborsClassifier(n_neighbors=max_k, metric=metric)
    clf.fit(X, y)
    return clf


def evaluate_task(clf, X_all, labels_col, task_mask, n_neighbors_list, weights="uniform", temperature=0.07):
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
            "--knn-weights=softmax requires --metric=cosine, got --metric="
            f"{config.metric!r}"
        )

    t_start = time.time()
    if config.log_wandb:
        import wandb

    print(f"Loading MycoAI checkpoint: {config.checkpoint}")
    model = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    model.eval()

    results_file = knn_results_path(config.results_file, config.knn_weights)
    model_name = os.path.splitext(os.path.basename(config.checkpoint))[0]

    train_fasta = os.path.join(config.data_dir, "trainset.fasta")
    print("\nLoading gallery (trainset)...")
    gallery_df_raw = Data(train_fasta, allow_duplicates=True).data
    needed_label_cols = [c for task, c in (("species_level", "species"), ("genus_level", "genus"))
                          if task in config.tasks]
    is_usable = pd.concat([gallery_df_raw[c] != UNKNOWN_STR for c in needed_label_cols], axis=1).any(axis=1)
    gallery_df = gallery_df_raw[is_usable].reset_index(drop=True)
    print(f"  {len(gallery_df_raw)} raw specimens -> {len(gallery_df)} with a resolved label for "
          f"{'/'.join(needed_label_cols)} ({len(gallery_df_raw) - len(gallery_df)} dropped, "
          f"useless for --tasks {config.tasks})")

    print(f"Extracting gallery embeddings ({len(gallery_df)} specimens)...")
    X_gallery = embed(model, train_fasta, gallery_df)
    print(f"  representation shape: {X_gallery.shape}")

    max_k = max(config.n_neighbors)
    clf_species = (fit_knn(X_gallery, gallery_df["species"], max_k, config.metric)
                   if "species_level" in config.tasks else None)
    clf_genus = (fit_knn(X_gallery, gallery_df["genus"], max_k, config.metric)
                 if "genus_level" in config.tasks else None)
    if clf_species is not None:
        print(f"  species gallery: {len(clf_species._y)} specimens, {len(clf_species.classes_)} classes")
    if clf_genus is not None:
        print(f"  genus gallery:   {len(clf_genus._y)} specimens, {len(clf_genus.classes_)} classes")

    all_results = {}
    for name, tag in TEST_SETS:
        tasks_df = pd.read_csv(os.path.join(config.tasks_dir, f"{tag}_tasks.csv"))
        keep_ids = set(tasks_df.loc[tasks_df["task"].isin(config.tasks), "id"])
        if not keep_ids:
            print(f"\n{name}: 0 query specimens across {config.tasks} — skipping (no fasta load needed)")
            continue

        test_fasta = os.path.join(config.data_dir, f"{tag}.fasta")
        test_df = Data(test_fasta, allow_duplicates=False).data
        species_ids = set(tasks_df.loc[tasks_df["task"] == "species_level", "id"])
        genus_ids = set(tasks_df.loc[tasks_df["task"] == "genus_level", "id"])
        test_df = test_df.copy()
        test_df["task"] = np.where(
            test_df["id"].isin(species_ids), "species_level",
            np.where(test_df["id"].isin(genus_ids), "genus_level", "other"),
        )
        relevant = test_df[test_df["task"].isin(config.tasks)].reset_index(drop=True)
        print(f"\n{name}: {len(relevant)} query specimens across {config.tasks} "
              f"({(relevant['task'] == 'species_level').sum()} species_level, "
              f"{(relevant['task'] == 'genus_level').sum()} genus_level)")
        if len(relevant) == 0:
            continue

        X_query = embed(model, test_fasta, relevant)

        tag_lower = name.split()[0].lower()
        for task in config.tasks:
            print(f"  --- {task} ---")
            clf = clf_species if task == "species_level" else clf_genus
            label_col = relevant["species"] if task == "species_level" else relevant["genus"]
            task_mask = relevant["task"] == task
            res_by_k = evaluate_task(clf, X_query, label_col, task_mask, config.n_neighbors,
                                      weights=config.knn_weights, temperature=config.temperature)

            with open(results_file, "a") as f:
                for k, res in res_by_k.items():
                    all_results.setdefault(task, {}).setdefault(k, {})[name] = res
                    print(f"  [{task}] k={k}: accuracy={res['accuracy']:.2f}% "
                          f"balanced={res['accuracy-balanced']:.2f}% f1-macro={res['f1-macro']:.2f}% "
                          f"(n={res['count']})")
                    f.write(f"\n{config.run_name}_{task}_{model_name}_{tag_lower}_k{k}\t{res['accuracy']:.4f}")
        print(f"  -> saved {name} results to {results_file}")

    dt_total = time.time() - t_start
    print(f"\nFinished in {dt_total/60:.1f} min")

    if config.log_wandb:
        wandb.init(name=config.run_name, project=config.wandb_project, config=vars(config), job_type="knn_its_mycoai")
        log_dict = {}
        for task, by_k in all_results.items():
            for k, results in by_k.items():
                for name, res in results.items():
                    for metric, v in res.items():
                        log_dict[f"{task}/knn_k{k}/{name}/{metric}"] = v
        wandb.log(log_dict)


def get_parser():
    p = argparse.ArgumentParser(description="KNN evaluation for ITS-5M using a MycoAI-BERT/CNN checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to MycoAI-BERT.pt or MycoAI-CNN.pt")
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="ITS-5M data directory (trainset.fasta, test1-2.fasta).")
    p.add_argument("--tasks-dir", "--tasks_dir", dest="tasks_dir", required=True,
                    help="Directory containing test{1,2}_tasks.csv from analyze_its_overlap.py --export-dir.")
    p.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors",
                    default=[1, 3, 5, 7, 10, 15, 20, 25, 50], type=int, nargs="+")
    p.add_argument("--metric", default="cosine")
    p.add_argument("--knn-weights", "--knn_weights", dest="knn_weights", default="uniform",
                    choices=["uniform", "distance", "softmax"],
                    help="Vote weighting. 'softmax' matches DINOv2's kNN eval; requires --metric=cosine.")
    p.add_argument("--temperature", dest="temperature", type=float, default=0.07)
    p.add_argument("--tasks", dest="tasks", default=list(ALL_TASKS), nargs="+", choices=ALL_TASKS)
    p.add_argument("--run-name", "--run_name", dest="run_name", default="knn_its_mycoai")
    p.add_argument("--results-file", "--results_file", dest="results_file",
                    default="results_final/KNN_ITS_external_RESULTS.txt")
    p.add_argument("--log-wandb", "--log_wandb", dest="log_wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="barcodemae_cls")
    return p


def cli():
    config = get_parser().parse_args()
    run(config)


if __name__ == "__main__":
    cli()