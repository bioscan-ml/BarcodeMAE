#!/usr/bin/env python
"""KNN evaluation (uniform + optional softmax temperature sweep) for
BIOSCAN-5M using a pretrained BarcodeMamba/BarcodeMamba+ checkpoint from
https://github.com/bioscan-ml/BarcodeMamba-dev (branch
GTCtech-BarcodeMambaPlus-release).

BarcodeMamba is not a HuggingFace AutoModel -- it's a standalone repo with
its own model code (utils/barcode_mamba.py, Mamba2-based) and its own
char/k-mer/BPE tokenizers. See barcodemamba_common.py for checkpoint/tokenizer
loading (flexible to the checkpoint folder layouts seen in the wild) and
embedding extraction. This script hands those embeddings to OUR OWN KNN
fit/sweep/results-file pipeline (matching every other external baseline in
this paper: same k values, same softmax temperature sweep, same BIOSCAN-5M
supervised_train.csv/unseen.csv gallery-query split), instead of using
BarcodeMamba's own (uniform-only, fixed k=1) knn_probing.py.

Usage (BPE checkpoint):
    python knn_probing_barcodemamba.py \
        --barcodemamba-repo /scratch/$USER/BarcodeMamba-dev \
        --checkpoint-dir /scratch/$USER/barcodemamba_checkpoints/BarcodeMamba-plus-BIOSCAN-5M \
        --bpe-tokenizer-path /scratch/$USER/barcodemamba_checkpoints/BarcodeMamba-plus-BIOSCAN-5M/bpe_tokenizer.pkl \
        --data-dir ./BarcodeMAE/data/BIOSCAN-5M \
        --knn-weights softmax --temperature-sweep 0.01 0.02 0.05 0.07 0.1 0.2 0.5 1.0 \
        --run-name knn_external_barcodemamba_bioscan5m_softmax_sweep \
        --results-file results_final/KNN_external_temp_sweep_RESULTS.txt
"""

import argparse
import os
import time

import pandas as pd
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

from barcodebert.barcodemamba_common import embed_sequences, load_barcodemamba, load_bpe_tokenizer
from barcodebert.evaluation import knn_results_path, knn_vote


def run(config):
    if config.knn_weights == "softmax" and config.metric != "cosine":
        raise ValueError(
            "--knn-weights=softmax requires --metric=cosine, got --metric="
            f"{config.metric!r}"
        )

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

    model_name = os.path.basename(os.path.normpath(config.checkpoint_dir))
    results_file = knn_results_path(config.results_file, config.knn_weights)

    df_train = pd.read_csv(os.path.join(config.data_dir, "supervised_train.csv"))
    df_test = pd.read_csv(os.path.join(config.data_dir, config.query_file))
    target_level = f"{config.taxon}_index"

    print(f"\nExtracting gallery embeddings ({len(df_train)} specimens)...")
    X = embed_sequences(model, tokenizer, tokenizer_name, df_train["nucleotides"], config.max_length)
    y = df_train[target_level].to_numpy()
    print(f"Extracting query embeddings ({len(df_test)} specimens)...")
    X_unseen = embed_sequences(model, tokenizer, tokenizer_name, df_test["nucleotides"], config.max_length)
    y_unseen = df_test[target_level].to_numpy()

    c = sum(1 for label in y_unseen if label not in y)
    print(f"There are {c} genus that are not present during training")

    # kNN =====================================================================
    print("Computing Nearest Neighbors", flush=True)
    n_neighbors_list = config.n_neighbors
    max_k = max(n_neighbors_list)
    clf = KNeighborsClassifier(n_neighbors=max_k, metric=config.metric)
    clf.fit(X, y)

    partitions = [("Train", X, y), ("Unseen", X_unseen, y_unseen)]
    neigh_dist, neigh_ind = {}, {}
    for partition_name, X_part, _ in partitions:
        neigh_dist[partition_name], neigh_ind[partition_name] = clf.kneighbors(X_part, n_neighbors=max_k)

    sweep_temperatures = (
        config.temperature_sweep if (config.knn_weights == "softmax" and config.temperature_sweep)
        else [config.temperature]
    )
    best_combo = None  # (accuracy, temperature, k)
    all_results = {}
    sweep_results = []
    for k in n_neighbors_list:
        for temperature in sweep_temperatures:
            results = {}
            for partition_name, X_part, y_part in partitions:
                ind_k = neigh_ind[partition_name][:, :k]
                dist_k = neigh_dist[partition_name][:, :k]
                neighbor_labels = clf._y[ind_k]
                majority_idx = knn_vote(neighbor_labels, dist_k, weights=config.knn_weights, temperature=temperature)
                y_pred = clf.classes_[majority_idx]
                results[partition_name] = {
                    "count": len(y_part),
                    "accuracy": 100.0 * sklearn.metrics.accuracy_score(y_part, y_pred),
                    "accuracy-balanced": 100.0 * sklearn.metrics.balanced_accuracy_score(y_part, y_pred),
                    "f1-macro": 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="macro"),
                }
            all_results[k] = results
            unseen_acc = results["Unseen"]["accuracy"]
            train_acc = results["Train"]["accuracy"]
            # Train accuracy is a free sanity check, not a reported metric: at
            # k=1 the gallery a training point is queried against includes
            # that exact point (distance ~0), so it should land near 100%.
            # If it's also low, embeddings/extraction are broken, not just
            # generalizing poorly.
            if len(sweep_temperatures) > 1:
                sweep_results.append((temperature, k, unseen_acc))
                print(f"  T={temperature:<6} k={k:<3} Train accuracy={train_acc:.4f}%  Unseen accuracy={unseen_acc:.4f}%")
            else:
                print(f"  k={k:<3} Train accuracy={train_acc:.4f}%  Unseen accuracy={unseen_acc:.4f}%")
            if best_combo is None or unseen_acc > best_combo[0]:
                best_combo = (unseen_acc, temperature, k)

    if len(sweep_temperatures) > 1:
        best_acc, best_t, best_k = best_combo
        print(f"\nBEST: T={best_t}, k={best_k}, Unseen accuracy={best_acc:.4f}%")

    with open(results_file, "a") as f:
        if len(sweep_temperatures) > 1:
            for temperature, k, acc in sweep_results:
                f.write(f"\n{config.run_name}_{model_name}_T{temperature}_k{k}\t {acc:.4f}")
            best_acc, best_t, best_k = best_combo
            f.write(f"\n{config.run_name}_{model_name}_BEST_T{best_t}_k{best_k}\t {best_acc:.4f}")
        else:
            for k, results in all_results.items():
                acc = results["Unseen"]["accuracy"]
                f.write(f"\n{config.run_name}_{model_name}_k{k}\t {acc:.4f}")

    print(f"\nFinished in {(time.time() - t_start) / 60:.1f} min. Results -> {results_file}")


def get_parser():
    p = argparse.ArgumentParser(description="KNN evaluation for BIOSCAN-5M using a BarcodeMamba/BarcodeMamba+ checkpoint.")
    p.add_argument("--barcodemamba-repo", "--barcodemamba_repo", dest="barcodemamba_repo", required=True,
                    help="Path to a local clone of bioscan-ml/BarcodeMamba-dev "
                    "(branch GTCtech-BarcodeMambaPlus-release), so utils.probing_utils/utils.ssm_dataset can be imported.")
    p.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", required=True,
                    help="Folder containing a config (.hydra/config.yaml or config.yaml) and a"
                    " .ckpt file (checkpoints/last.ckpt, last.ckpt, or model.ckpt).")
    p.add_argument("--checkpoint-name", "--checkpoint_name", dest="checkpoint_name", default=None,
                    help="Checkpoint filename within checkpoint-dir/checkpoints/. Default: last.ckpt")
    p.add_argument("--bpe-tokenizer-path", "--bpe_tokenizer_path", dest="bpe_tokenizer_path", default=None,
                    help="Path to bpe_tokenizer.pkl, required when the checkpoint's tokenizer.name is 'bpe'.")
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="BIOSCAN-5M data directory (supervised_train.csv, unseen.csv).")
    p.add_argument("--query-file", "--query_file", dest="query_file", default="unseen.csv")
    p.add_argument("--taxon", default="genus")
    p.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=660)
    p.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors",
                    default=[1, 3, 5, 7, 10, 15, 20, 25, 50], type=int, nargs="+")
    p.add_argument("--metric", default="cosine")
    p.add_argument("--knn-weights", "--knn_weights", dest="knn_weights", default="uniform",
                    choices=["uniform", "distance", "softmax"])
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--temperature-sweep", "--temperature_sweep", dest="temperature_sweep",
                    default=None, type=float, nargs="+")
    p.add_argument("--run-name", "--run_name", dest="run_name", default="knn_external_barcodemamba")
    p.add_argument("--results-file", "--results_file", dest="results_file",
                    default="results_final/KNN_external_RESULTS.txt")
    return p


def cli():
    config = get_parser().parse_args()
    run(config)


if __name__ == "__main__":
    cli()