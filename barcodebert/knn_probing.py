#!/usr/bin/env python

import os
import resource
import time
from itertools import product

import pandas as pd
import sklearn.metrics
import torch
import torch.optim
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torchtext.vocab import vocab as build_vocab_from_dict

from barcodebert import utils
from barcodebert.datasets import BPETokenizer, KmerTokenizer, representations_from_df
from barcodebert.io import load_pretrained_model


def run(config):
    r"""
    Run kNN job, using a single GPU worker to create the embeddings.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    t_start = time.time()
    timing_stats = {}

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower, but more reproducible.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.", flush=True)

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # LOAD PRE-TRAINED CHECKPOINT =============================================
    # Map model parameters to be load to the specified gpu.
    model, pre_checkpoint = load_pretrained_model(config.pretrained_checkpoint_path, device=device)
    # Override the classifier with an identity function as we only want the embeddings
    model.classifier = nn.Identity()
    model = model.to(device)

    keys_to_reuse = [
        "k_mer",
        "stride",
        "max_len",
        "tokenizer",
        "bpe_path",
        "tokenize_n_nucleotide",
        "predict_n_nucleotide",
        "pretrain_levenshtein",
        "levenshtein_vectorized",
        "n_layers",
        "n_heads",
        "dataset_name",
        "use_cls_token"
    ]
    default_kwargs = vars(get_parser().parse_args(["--pretrained_checkpoint=dummy.pt", "--dataset=foo_bar"]))
    for key in keys_to_reuse:
        if not hasattr(config, key) or getattr(config, key) == getattr(pre_checkpoint["config"], key):
            pass
        elif getattr(config, key) is None or getattr(config, key) == default_kwargs[key]:
            print(
                f"  Overriding default config value {key}={getattr(config, key)}"
                f" with {getattr(pre_checkpoint['config'], key)} from pretained checkpoint."
            )
        elif getattr(config, key) != getattr(pre_checkpoint["config"], key):
            raise ValueError(
                f"config value for {key} differs from pretrained checkpoint:"
                f" {getattr(config, key)} (ours) vs {getattr(pre_checkpoint['config'], key)} (pretrained checkpoint)"
            )
        setattr(config, key, getattr(pre_checkpoint["config"], key, None))

    config.pretrained_run_name = pre_checkpoint["config"].run_name
    config.pretrained_run_id = pre_checkpoint["config"].run_id

    # DATASET =================================================================

    if config.tokenizer == "kmer":
        base_pairs = "ACGT"
        # specials = ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[UNK]"]
        if hasattr(config, "use_cls_token") and config.use_cls_token:
            specials = ["[MASK]", "[UNK]", "[CLS]"]
        else:
            specials = ["[MASK]", "[UNK]"]

        UNK_TOKEN = "[UNK]"

        if config.tokenize_n_nucleotide:
            # Encode kmers which contain N differently depending on where it is
            base_pairs += "N"

        kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=config.k_mer)]

        if config.tokenize_n_nucleotide:
            prediction_kmers = []
            other_kmers = []
            for kmer in kmers:
                if "N" in kmer:
                    other_kmers.append(kmer)
                else:
                    prediction_kmers.append(kmer)

            kmers = prediction_kmers + other_kmers

        kmer_dict = dict.fromkeys(kmers, 1)
        vocab = build_vocab_from_dict(kmer_dict, specials=specials)
        vocab.set_default_index(vocab[UNK_TOKEN])
        tokenizer = KmerTokenizer(config.k_mer, vocab, stride=config.stride, padding=True, max_len=config.max_len)

    elif config.tokenizer == "bpe":
        tokenizer = BPETokenizer(padding=True, max_tokenized_len=config.max_len, bpe_path=config.bpe_path)

    df_train = pd.read_csv(os.path.join(config.data_dir, "supervised_train.csv"))
    df_test = pd.read_csv(os.path.join(config.data_dir, "unseen.csv"))

    if config.taxon.lower() == "bin":
        config.target_level = "bin_uri"
    else:
        if config.dataset_name == "CANADA-1.5M":
            config.target_level = config.taxon + "_name"
        elif config.dataset_name == "BIOSCAN-5M":
            config.target_level = config.taxon + "_index"
        else:
            raise NotImplementedError("Dataset format is not supported. Must be one of CANADA-1.5M or BIOSCAN-5M")

    timing_stats["preamble"] = time.time() - t_start

    # Ensure model is in eval mode
    model.eval()
    t_start_embed = time.time()
    # Generate emebddings for the training and test sets
    print("Generating embeddings for test set", flush=True)
    X_unseen, y_unseen, orders = representations_from_df(
        df_test,
        config.target_level,
        model,
        tokenizer,
        config.dataset_name,
        config.mode,
        config.mask_rate,
        config.representation_type,
        use_cls_token=getattr(config, "use_cls_token", False),
    )
    print("Generating embeddings for train set", flush=True)
    X, y, train_orders = representations_from_df(
        df_train,
        config.target_level,
        model,
        tokenizer,
        config.dataset_name,
        config.mode,
        config.mask_rate,
        config.representation_type,
        use_cls_token=getattr(config, "use_cls_token", False),
    )
    timing_stats["embed"] = time.time() - t_start_embed

    c = 0
    for label in y_unseen:
        if label not in y:
            c += 1
    print(f"There are {c} genus that are not present during training")

    running_info = resource.getrusage(resource.RUSAGE_SELF)
    dt = time.time() - t_start_embed
    hour = dt // 3600
    minutes = (dt - (3600 * hour)) // 60
    seconds = dt - (hour * 3600) - (minutes * 60)
    memory = running_info.ru_maxrss / 1e6
    print(f"Creating embeddings took: {int(hour)}:{int(minutes):02d}:{seconds:02.0f} (hh:mm:ss)\n")
    print(f"Max memory usage: {memory} (GB)")

    # kNN =====================================================================
    print("Computing Nearest Neighbors", flush=True)

    n_neighbors_list = config.n_neighbors  # already a list

    # Fit once with the largest k (reuse for all smaller k via kneighbors())
    t_start_train = time.time()
    max_k = max(n_neighbors_list)
    clf = KNeighborsClassifier(n_neighbors=max_k, metric=config.metric)
    clf.fit(X, y)
    timing_stats["train"] = time.time() - t_start_train

    # Precompute distances once for both partitions
    t_start_test = time.time()
    partitions = [("Train", X, y), ("Unseen", X_unseen, y_unseen)]
    neigh_dist = {}
    neigh_ind = {}
    for partition_name, X_part, _ in partitions:
        neigh_dist[partition_name], neigh_ind[partition_name] = clf.kneighbors(X_part, n_neighbors=max_k)

    # Evaluate for each k
    all_results = {}  # k -> {partition -> metrics}
    for k in n_neighbors_list:
        print(f"\n{'='*50}")
        print(f"k = {k}")
        print(f"{'='*50}")
        results = {}
        for partition_name, X_part, y_part in partitions:
            # Use the k closest neighbors from precomputed distances
            ind_k = neigh_ind[partition_name][:, :k]
            # Majority vote
            neighbor_labels = y[ind_k] if partition_name == "Train" else y[ind_k]
            # For train partition neighbors come from train set (same clf)
            neighbor_labels = clf._y[ind_k]
            from scipy import stats as scipy_stats
            y_pred = scipy_stats.mode(neighbor_labels, axis=1, keepdims=False).mode
            res_part = {}
            res_part["count"] = len(y_part)
            res_part["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_part, y_pred)
            res_part["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_part, y_pred)
            res_part["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="micro")
            res_part["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="macro")
            res_part["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_part, y_pred, average="weighted")
            results[partition_name] = res_part
            print(f"\n{partition_name} evaluation results (k={k}):")
            for metric_name, v in res_part.items():
                if metric_name == "count":
                    print(f"  {metric_name + ' ':.<21s}{v:7d}")
                else:
                    print(f"  {metric_name + ' ':.<24s} {v:6.2f} %")
        all_results[k] = results

    timing_stats["test"] = time.time() - t_start_test

    # Save results -------------------------------------------------------------
    dt = time.time() - t_start
    hour = dt // 3600
    minutes = (dt - (3600 * hour)) // 60
    seconds = dt - (hour * 3600) - (minutes * 60)
    print(f"\nThe code finished after: {int(hour)}:{int(minutes):02d}:{seconds:02.0f} (hh:mm:ss)\n")

    model_name = os.path.join(*os.path.split(config.pretrained_checkpoint_path)[-2:])
    with open("KNN_RESULTS.txt", "a") as f:
        for k, results in all_results.items():
            acc = results["Unseen"]["accuracy"]
            f.write(f"\n{config.run_name}_{model_name}_k{k}\t {acc:.4f}")

    timing_stats["overall"] = time.time() - t_start

    # LOGGING =================================================================
    if config.log_wandb:
        wandb_run_name = config.run_name
        if wandb_run_name is not None and config.run_id is not None:
            wandb_run_name = f"{wandb_run_name}__{config.run_id}"
        EXCLUDED_WANDB_CONFIG_KEYS = [
            "log_wandb",
            "wandb_entity",
            "wandb_project",
            "global_rank",
            "local_rank",
            "run_name",
            "run_id",
            "model_output_dir",
        ]
        job_type = "knn"
        wandb.init(
            name=wandb_run_name,
            id=config.run_id,
            group=config.pretrained_run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=wandb.helper.parse_config(config, exclude=EXCLUDED_WANDB_CONFIG_KEYS),
            job_type=job_type,
            tags=["evaluate", job_type],
        )

        # Log results for all k values to wandb
        log_dict = {**{f"knn/duration/{k}": v for k, v in timing_stats.items()}}
        for k, results in all_results.items():
            for partition, res in results.items():
                for metric_name, v in res.items():
                    log_dict[f"knn_k{k}/{partition}/{metric_name}"] = v
        wandb.log(log_dict)


def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import sys

    from barcodebert.pretraining import get_parser as get_pretraining_parser

    parser = get_pretraining_parser()

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser.prog = prog
    parser.description = "Evaluate with k-nearest neighbors for BarcodeBERT."

    # Model args --------------------------------------------------------------
    group = parser.add_argument_group("Input model")
    group.add_argument(
        "--pretrained-checkpoint",
        "--pretrained_checkpoint",
        dest="pretrained_checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        required=True,
        help="Path to pretrained model checkpoint (required).",
    )
    # kNN args ----------------------------------------------------------------
    group = parser.add_argument_group("kNN parameters")
    group.add_argument(
        "--taxon",
        type=str,
        default="genus",
        help="Taxonomic level to evaluate on. Default: %(default)s",
    )
    group.add_argument(
        "--n-neighbors",
        "--n_neighbors",
        default=[1],
        type=int,
        nargs="+",
        help="Neighborhood size(s) for kNN. Pass one or more values. Default: %(default)s",
    )
    group.add_argument(
        "--metric",
        default="cosine",
        type=str,
        help="Distance metric to use for kNN. Default: %(default)s",
    )

    # Data args ---------------------------------------------------------------
    group.add_argument(
        "--mode",
        default="nonmask",
        type=str,
        help="Mode for generating representations. Default: %(default)s",
    )

    group.add_argument(
        "--mask-rate",
        default=0.5,
        type=float,
        help="Mask rate for masked language model. Default: %(default)s",
    )

    group.add_argument(
        "--representation_type",
        default="tokens",
        type=str,
        choices=["tokens", "tokens_with_cls", "jumbo", "jumbo_avg", "all_tokens", "cls",
                 "tokens_with_registers", "all_with_registers"],
        help=(
            "Type of representation to extract. Options: "
            "'tokens' (mean of sequence tokens, excluding CLS and registers), "
            "'tokens_with_cls' (mean of sequence tokens including CLS, no registers), "
            "'tokens_with_registers' (mean of sequence tokens + registers, excluding CLS), "
            "'all_with_registers' (mean of all tokens: sequence + registers + CLS), "
            "'cls' (CLS token at position 0), "
            "'jumbo' (flattened jumbo tokens, J*D dim), "
            "'jumbo_avg' (average of jumbo tokens), "
            "'all_tokens' (average of jumbo + sequence tokens). "
            "Default: %(default)s"
        ),
    )

    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    return run(config)


if __name__ == "__main__":
    cli()
