#!/usr/bin/env python

import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from torch import nn
from torchtext.vocab import vocab as build_vocab

sys.path.append(".")
from barcodebert import utils
from barcodebert.datasets import BPETokenizer, KmerTokenizer, representations_from_df
from barcodebert.io import load_pretrained_model


def zsc_pipeline(X, y_true, metric="cosine", n_neighbours=10, n_clusters=None):
    """
    UMAP dimensionality reduction -> Agglomerative Clustering -> AMI evaluation.

    Parameters
    ----------
    X : np.ndarray (N, D)
    y_true : array-like (N,)
    metric : str
    n_neighbours : int
    n_clusters : int or None
        Number of clusters for AgglomerativeClustering.
        If None, inferred from the number of unique labels in y_true.
    """
    if n_clusters is None:
        n_clusters = len(np.unique(y_true))
        print(f"Auto-detected n_clusters = {n_clusters}")

    print(f"Running UMAP (metric={metric}, n_neighbors={n_neighbours})...")
    umap_reducer = umap.UMAP(n_components=50, random_state=42, metric=metric, n_neighbors=n_neighbours)
    X_reduced = umap_reducer.fit_transform(X)

    print(f"Running AgglomerativeClustering (n_clusters={n_clusters})...")
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    cluster_labels = clustering.fit_predict(X_reduced)

    ami_score = adjusted_mutual_info_score(y_true, cluster_labels)
    print(f"Adjusted Mutual Information (AMI) score: {ami_score:.4f}")
    return ami_score


def run(config):
    r"""
    Run ZSC evaluation using a single GPU worker to create embeddings.

    Parameters
    ----------
    config : argparse.Namespace
    """
    t_start = time.time()

    if config.log_wandb:
        import wandb

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("\nConfiguration:\n")
    print(config)
    print(f"\nFound {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.", flush=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- Load checkpoint ---
    model, pre_checkpoint = load_pretrained_model(config.pretrained_checkpoint_path, device=device)
    if hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()

    # Inherit tokenizer settings from checkpoint
    keys_to_reuse = [
        "k_mer", "stride", "max_len", "tokenizer", "bpe_path",
        "tokenize_n_nucleotide", "predict_n_nucleotide",
        "pretrain_levenshtein", "levenshtein_vectorized",
        "n_layers", "n_heads", "dataset_name", "use_cls_token",
    ]
    default_kwargs = vars(get_parser().parse_args([
        "--pretrained_checkpoint=dummy.pt", "--backbone=dummy", "--data-dir=.", "--dataset=BIOSCAN-5M"
    ]))
    for key in keys_to_reuse:
        ckpt_val = getattr(pre_checkpoint["config"], key, None)
        if ckpt_val is None:
            continue
        cur_val = getattr(config, key, None)
        if cur_val is None or cur_val == default_kwargs.get(key):
            print(f"  Using checkpoint value: {key} = {ckpt_val}")
        setattr(config, key, ckpt_val)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # --- Tokenizer ---
    if config.tokenizer == "kmer":
        base_pairs = "ACGT"
        if getattr(config, "use_cls_token", False):
            specials = ["[MASK]", "[UNK]", "[CLS]"]
        else:
            specials = ["[MASK]", "[UNK]"]
        if getattr(config, "tokenize_n_nucleotide", False):
            base_pairs += "N"
        kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=config.k_mer)]
        if getattr(config, "tokenize_n_nucleotide", False):
            kmers = [k for k in kmers if "N" not in k] + [k for k in kmers if "N" in k]
        vocab = build_vocab(dict.fromkeys(kmers, 1), specials=specials)
        vocab.set_default_index(vocab["[UNK]"])
        tokenizer = KmerTokenizer(config.k_mer, vocab, stride=config.stride,
                                   padding=True, max_len=config.max_len)
    elif config.tokenizer == "bpe":
        tokenizer = BPETokenizer(padding=True, max_tokenized_len=config.max_len,
                                  bpe_path=config.bpe_path)
    else:
        raise ValueError(f"Unknown tokenizer: {config.tokenizer}")

    # --- Determine target column ---
    if config.taxon.lower() == "bin":
        target_level = "bin_uri"
    elif config.taxon.lower() == "dna_bin":
        target_level = "dna_bin"
    elif config.dataset_name == "CANADA-1.5M":
        target_level = config.taxon + "_name"
    elif config.dataset_name == "BIOSCAN-5M":
        target_level = config.taxon + "_index"
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

    # --- Load dataframes ---
    df_test = pd.read_csv(os.path.join(config.data_dir, "supervised_test.csv"))
    df_unseen = pd.read_csv(os.path.join(config.data_dir, "unseen.csv"))

    # Drop rows with missing target labels (e.g. specimens without BIN assignment)
    df_test = df_test.dropna(subset=[target_level])
    df_unseen = df_unseen.dropna(subset=[target_level])

    # --- Extract representations using barcodebert.datasets.representations_from_df ---
    rep_type = getattr(config, "representation_type", "tokens")
    use_cls = getattr(config, "use_cls_token", False)

    print(f"\nExtracting representations (type={rep_type}) for supervised_test...", flush=True)
    X_test, y_test, _ = representations_from_df(
        df_test,
        target_level,
        model,
        tokenizer,
        config.dataset_name,
        "nonmask",
        0.0,
        rep_type,
        use_cls_token=use_cls,
    )

    print(f"Extracting representations (type={rep_type}) for unseen...", flush=True)
    X_unseen, y_unseen, _ = representations_from_df(
        df_unseen,
        target_level,
        model,
        tokenizer,
        config.dataset_name,
        "nonmask",
        0.0,
        rep_type,
        use_cls_token=use_cls,
    )

    X = np.vstack([X_test, X_unseen])
    # y_test and y_unseen may be pandas Series or numpy arrays
    y_test = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.array(y_test)
    y_unseen = y_unseen.to_numpy() if hasattr(y_unseen, "to_numpy") else np.array(y_unseen)
    y = np.hstack([y_test, y_unseen])

    print(f"\nCombined: {X.shape[0]} samples, {len(np.unique(y))} unique labels, dim={X.shape[1]}")

    # --- ZSC pipeline ---
    ami = 100.0 * zsc_pipeline(
        X, y,
        metric=config.metric,
        n_neighbours=config.n_neighbors,
        n_clusters=getattr(config, "n_clusters", None),
    )
    print(f"\nFinal AMI (%): {ami:.4f}")
    print(f"Total time: {time.time() - t_start:.1f}s")

    model_name = os.path.join(*os.path.split(config.pretrained_checkpoint_path)[-2:])
    with open(getattr(config, "results_file", "ZSC_RESULTS.txt"), "a") as f:
        f.write(f"\n{config.run_name}_{model_name}_{rep_type}\t{ami:.4f}")

    if config.log_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.run_name,
            id=getattr(config, "run_id", None),
            config=vars(config),
        )
        wandb.log({"eval/ami": ami})

    return ami


def get_parser():
    import sys
    from barcodebert.pretraining import get_parser as get_pretraining_parser

    parser = get_pretraining_parser()

    prog = os.path.split(sys.argv[0])[1]
    if prog in ("__main__.py", "__main__"):
        prog = os.path.split(__file__)[1]
    parser.prog = prog
    parser.description = "Zero-Shot Clustering (ZSC) evaluation for BarcodeMAE."

    group = parser.add_argument_group("Model")
    group.add_argument(
        "--pretrained-checkpoint", "--pretrained_checkpoint",
        dest="pretrained_checkpoint_path",
        default="", type=str, metavar="PATH", required=True,
        help="Path to pretrained checkpoint (.pt)",
    )
    group.add_argument(
        "--backbone", dest="backbone",
        default="barcodebert", type=str,
        help="Model name (used for logging only)",
    )
    group.add_argument(
        "--representation-type", "--representation_type",
        dest="representation_type",
        default="tokens", type=str,
        choices=["tokens", "tokens_with_cls", "cls", "jumbo", "jumbo_avg",
                 "all_tokens", "tokens_with_registers", "all_with_registers"],
        help="How to extract the sequence embedding from the model",
    )
    group.add_argument(
        "--results-file",
        "--results_file",
        dest="results_file",
        default="ZSC_RESULTS.txt",
        type=str,
        help="File to append ZSC AMI results to. Default: %(default)s",
    )

    group = parser.add_argument_group("ZSC parameters")
    group.add_argument("--taxon", type=str, default="bin_uri",
                       help="Taxonomic level to evaluate. Default: %(default)s")
    group.add_argument("--n-neighbors", "--n_neighbors", dest="n_neighbors",
                       default=5, type=int,
                       help="UMAP neighborhood size. Default: %(default)s")
    group.add_argument("--metric", default="cosine", type=str,
                       help="UMAP distance metric. Default: %(default)s")
    group.add_argument("--n-clusters", "--n_clusters", dest="n_clusters",
                       default=None, type=int,
                       help="Number of clusters (auto-detected from data if not set)")

    return parser


def cli():
    parser = get_parser()
    config = parser.parse_args()
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    return run(config)


if __name__ == "__main__":
    cli()
