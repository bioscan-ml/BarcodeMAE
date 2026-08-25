#!/usr/bin/env python
"""Leakage-free, genus-level KNN evaluation of ITS-5M on the pretraining
VALIDATION split (trainset_valid.fasta), for hyperparameter tuning (e.g. the
aux-loss weight sweep) without touching the real test pools (test1/test2/
test3.fasta) that the paper's headline results are computed on.

Reuses knn_its_clean.py's embedding/KNN machinery, but is deliberately a
separate, slimmer script rather than an extension of knn_its_clean.py's
--tasks flag: trainset_valid.fasta has ZERO genus_level-task specimens (see
analyze_its_valtrain_overlap.py -- every resolvable species in it is already
in trainset.fasta's gallery, since it's a random held-out split of the same
pool, not a species-holdout), so there is no way to reproduce a genuine
open-world genus_level evaluation from it. Instead, per-request, this scores
GENUS-level accuracy on the 'species_level'-task query specimens from
trainset_valid_tasks.csv (i.e. clean, non-leaked, closed-world specimens --
species not novel, but that's fine for weight-tuning purposes) -- using
their genus label instead of species, decoupling query-selection from
scoring-level the way knn_its_clean.py's --tasks does not.

Requires trainset_valid_tasks.csv already exported:
    python barcodebert/analyze_its_valtrain_overlap.py --data-dir data/ITS-5M --export-dir data/ITS-5M/tasks

Usage:
    python barcodebert/knn_its_clean_val.py \
        --pretrained-checkpoint path/to/checkpoint_encoder.pt \
        --data-dir data/ITS-5M --tasks-dir data/ITS-5M/tasks \
        --representation-type cls \
        --run-name ablw_its_..._val --results-file results_final/KNN_ITS_val_aux_weight_ablation_RESULTS.txt
"""

import argparse
import os
import resource
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
from mycoai.data import Data
from torch import nn

from barcodebert.evaluation import knn_results_path
from barcodebert.io import load_pretrained_model
from barcodebert.knn_its_clean import UNKNOWN_STR, evaluate_task, extract_representations, fit_knn
from barcodebert.datasets import KmerTokenizer


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

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    results_file = knn_results_path(config.results_file, config.knn_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, pre_checkpoint = load_pretrained_model(config.pretrained_checkpoint_path, device=device)
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()
    pre_config = pre_checkpoint["config"]
    k_mer = getattr(pre_config, "k_mer", 6)
    stride = getattr(pre_config, "stride", k_mer)
    max_len = getattr(pre_config, "max_len", 660)
    use_cls = getattr(pre_config, "use_cls_token", False)
    print(f"\nk_mer={k_mer}, stride={stride}, max_len={max_len}, use_cls_token={use_cls}")

    base_pairs = "ACGT"
    specials = ["[MASK]", "[UNK]", "[CLS]"] if use_cls else ["[MASK]", "[UNK]"]
    kmers = ["".join(k) for k in product(base_pairs, repeat=k_mer)]
    from torchtext.vocab import vocab as build_vocab_from_dict

    kmer_dict = dict.fromkeys(kmers, 1)
    vocab = build_vocab_from_dict(kmer_dict, specials=specials)
    vocab.set_default_index(vocab["[UNK]"])
    tokenizer = KmerTokenizer(k_mer, vocab, stride=stride, padding=True, max_len=max_len)

    # ── Gallery: trainset.fasta, genus classifier only ──────────────────────────
    print("\nLoading gallery (trainset)...")
    gallery_df_raw = Data(os.path.join(config.data_dir, "trainset.fasta"), allow_duplicates=True).data
    gallery_df = gallery_df_raw[gallery_df_raw["genus"] != UNKNOWN_STR].reset_index(drop=True)
    print(f"  {len(gallery_df_raw)} raw specimens -> {len(gallery_df)} with a resolved genus "
          f"({len(gallery_df_raw) - len(gallery_df)} dropped)")

    print("Extracting gallery embeddings...")
    X_gallery = extract_representations(
        gallery_df["sequence"].tolist(), model, tokenizer, config.representation_type, use_cls, device,
        embed_batch_size=config.embed_batch_size,
    )

    max_k = max(config.n_neighbors)
    print("Fitting genus KNN classifier...", flush=True)
    clf_genus = fit_knn(X_gallery, gallery_df["genus"], max_k, config.metric)
    print(f"  genus gallery: {len(clf_genus._y)} specimens, {len(clf_genus.classes_)} classes")

    # ── Query: trainset_valid.fasta, restricted to the clean species_level ─────
    # task from trainset_valid_tasks.csv (no genus_level-task specimens exist
    # for this file -- see module docstring), scored on GENUS label regardless.
    # dtype=str on the id column: pandas otherwise infers int64 for
    # numeric-looking ids on CSV read, while Data()'s freshly-parsed id column
    # may be string/object -- "12345" != 12345 in Python, so .isin() silently
    # matches nothing despite the values "looking" the same. Confirmed: this
    # alone (independent of the allow_duplicates fix below) reproduced
    # "0 clean query specimens" even after that fix was applied.
    tasks_df = pd.read_csv(os.path.join(config.tasks_dir, "trainset_valid_tasks.csv"), dtype={"id": str})
    keep_ids = set(tasks_df.loc[tasks_df["task"] == "species_level", "id"])
    if not keep_ids:
        raise RuntimeError("0 clean species_level query specimens in trainset_valid_tasks.csv -- "
                            "re-run analyze_its_valtrain_overlap.py --export-dir first.")

    # allow_duplicates=True to match analyze_its_valtrain_overlap.py's load_split(),
    # which sets allow_duplicates = "train" in os.path.basename(fasta_path) --
    # "trainset_valid.fasta" contains "train", so that's how trainset_valid_tasks.csv's
    # ids were generated. Loading with allow_duplicates=False here would produce a
    # different row set/id assignment, silently breaking the id match against
    # keep_ids (observed: 0 query specimens survive the .isin() filter).
    query_df_raw = Data(os.path.join(config.data_dir, "trainset_valid.fasta"), allow_duplicates=True).data
    query_df_raw["id"] = query_df_raw["id"].astype(str)
    query_df = query_df_raw[query_df_raw["id"].isin(keep_ids)].reset_index(drop=True)
    if len(query_df) == 0:
        print(f"DEBUG: 0 matches. Sample keep_ids: {sorted(keep_ids)[:5]}", file=sys.stderr)
        print(f"DEBUG: sample query_df_raw ids: {query_df_raw['id'].head(5).tolist()}", file=sys.stderr)
        raise RuntimeError("0 query specimens matched keep_ids by id -- id format mismatch between "
                            "trainset_valid_tasks.csv and the freshly-loaded trainset_valid.fasta "
                            "(see DEBUG lines above for a direct comparison).")
    query_df = query_df[query_df["genus"] != UNKNOWN_STR].reset_index(drop=True)
    print(f"\ntrainset_valid: {len(query_df)} clean query specimens with a resolved genus "
          f"(out of {len(keep_ids)} clean species_level ids)")

    X_query = extract_representations(
        query_df["sequence"].tolist(), model, tokenizer, config.representation_type, use_cls, device,
        embed_batch_size=config.embed_batch_size,
    )

    model_name = os.path.basename(config.pretrained_checkpoint_path)
    task_mask = pd.Series(True, index=query_df.index)
    res_by_k = evaluate_task(clf_genus, X_query, query_df["genus"], task_mask, config.n_neighbors,
                              weights=config.knn_weights, temperature=config.temperature)

    all_results = {}
    with open(results_file, "a") as f:
        for k, res in res_by_k.items():
            all_results[k] = res
            print(f"  [genus_level] k={k}: accuracy={res['accuracy']:.2f}% "
                  f"balanced={res['accuracy-balanced']:.2f}% f1-macro={res['f1-macro']:.2f}% "
                  f"(n={res['count']})")
            f.write(f"\n{config.run_name}_genus_level_{model_name}_trainset_valid_k{k}\t{res['accuracy']:.4f}")
    print(f"  -> saved results to {results_file}")

    dt_total = time.time() - t_start
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"\nFinished in {dt_total/60:.1f} min | Max memory: {mem:.1f} GB")

    if config.log_wandb:
        wandb.init(name=config.run_name, project=config.wandb_project, config=vars(config), job_type="knn_its_clean_val")
        wandb.log({f"genus_level/knn_k{k}/trainset_valid/{metric}": v
                    for k, res in all_results.items() for metric, v in res.items()})


def get_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pretrained-checkpoint", "--pretrained_checkpoint", dest="pretrained_checkpoint_path",
                    required=True, help="Path to pretrained encoder checkpoint.")
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                    help="ITS-5M data directory (trainset.fasta, trainset_valid.fasta).")
    p.add_argument("--tasks-dir", "--tasks_dir", dest="tasks_dir", required=True,
                    help="Directory containing trainset_valid_tasks.csv from "
                         "analyze_its_valtrain_overlap.py --export-dir.")
    p.add_argument("--representation-type", "--representation_type", dest="representation_type",
                    default="cls", choices=["cls", "tokens", "tokens_with_cls"])
    p.add_argument("--n-neighbors", "--n_neighbors", default=[1], type=int, nargs="+")
    p.add_argument("--metric", default="cosine", type=str)
    p.add_argument("--knn-weights", "--knn_weights", default="uniform", type=str,
                    choices=["uniform", "distance", "softmax"])
    p.add_argument("--temperature", default=0.07, type=float)
    p.add_argument("--embed-batch-size", "--embed_batch_size", dest="embed_batch_size", default=32, type=int)
    p.add_argument("--run-name", "--run_name", dest="run_name", required=True)
    p.add_argument("--results-file", "--results_file", dest="results_file", required=True)
    p.add_argument("--log-wandb", "--log_wandb", dest="log_wandb", action="store_true")
    p.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="barcodemae_cls")
    return p


def cli():
    args = get_parser().parse_args()
    run(args)


if __name__ == "__main__":
    cli()
