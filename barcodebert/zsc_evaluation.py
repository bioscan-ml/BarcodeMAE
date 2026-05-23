#!/usr/bin/env python

import os
import pickle
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
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.append(".")
from barcodebert import utils
from barcodebert.datasets import BPETokenizer, KmerTokenizer
from barcodebert.io import load_pretrained_model


class ModelEmbedder:
    """Wraps a loaded BarcodeMAE checkpoint into the embedder interface expected by DNADataset."""

    def __init__(self, model, tokenizer, name, hidden_size, representation_type="tokens", use_cls_token=False):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.hidden_size = hidden_size
        self.representation_type = representation_type
        self.use_cls_token = use_cls_token


class DNADataset(Dataset):
    def __init__(
    self,
        file_path,
        embedder,
        randomize_offset=False,
        max_length=660,
        dataset_format="CANADA-1.5M",
        target="species",
    ):
        self.randomize_offset = randomize_offset

        df = pd.read_csv(file_path, sep="\t" if file_path.endswith(".tsv") else ",")
        self.barcodes = df["nucleotides"].to_list()

        self.tokenizer = embedder.tokenizer
        self.backbone_name = embedder.name
        self.max_len = max_length
        self.dataset_format = dataset_format
        self.target = target
        self.use_cls_token = getattr(embedder, "use_cls_token", False)

        if dataset_format == "CANADA-1.5M":
            if target not in ["processid", "bin_uri"]:
                target += "_name"
                self.labels, self.label_set = pd.factorize(df[target], sort=True)
            else:
                self.labels = df[target].to_list()
                self.label_set = set(self.labels)
            self.num_labels = len(self.label_set)
        else:  # BIOSCAN-5M
            self.num_labels = 22_622
            # Accept the exact column name if it already exists in the CSV;
            # otherwise fall back to appending _index (baseline convention).
            if target not in df.columns and target not in ["processid", "dna_bin"]:
                target += "_index"
            self.ids = df[target].to_list()
            self.labels = self.ids

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        x = self.barcodes[idx]
        if len(x) > self.max_len:
            x = x[: self.max_len]

        if self.backbone_name == "BarcodeMAE":
            # KmerTokenizer / BPETokenizer returns (tokens_1d, att_mask_1d) tensors directly.
            processed_barcode, att_mask = self.tokenizer(x)
            if self.use_cls_token:
                cls_token = torch.tensor([2], dtype=processed_barcode.dtype)
                cls_mask = torch.tensor([1], dtype=att_mask.dtype)
                processed_barcode = torch.cat([cls_token, processed_barcode])
                att_mask = torch.cat([cls_mask, att_mask])
            processed_barcode = processed_barcode.unsqueeze(0)  # (1, seq_len)
            att_mask = att_mask.unsqueeze(0)

        elif self.backbone_name == "BarcodeBERT":
            encoding_info = self.tokenizer(x, return_tensors="pt", padding=True)
            processed_barcode = encoding_info["input_ids"]
            att_mask = encoding_info["attention_mask"]

        elif self.backbone_name == "Hyena_DNA":
            encoding_info = self.tokenizer(
                x,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=False,
            )
            processed_barcode = encoding_info["input_ids"]
            att_mask = encoding_info["attention_mask"]

        elif self.backbone_name == "DNABERT":
            k = 6
            kmer = [x[i: i + k] for i in range(len(x) + 1 - k)]
            encoding_info = self.tokenizer.encode_plus(
                " ".join(kmer),
                sentence_b=None,
                return_tensors="pt",
                add_special_tokens=False,
                padding="max_length",
                max_length=512,
                return_attention_mask=True,
                truncation=True,
            )
            processed_barcode = encoding_info["input_ids"]
            att_mask = encoding_info["attention_mask"]

        else:
            encoding_info = self.tokenizer(
                x,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                max_length=512,
                add_special_tokens=False,
                padding="max_length",
                truncation=True,
            )
            processed_barcode = encoding_info["input_ids"]
            att_mask = encoding_info["attention_mask"]

        if self.target not in ["processid", "bin_uri", "dna_bin"]:
            label = torch.tensor(self.labels[idx], dtype=torch.int64)
        else:
            label = self.labels[idx]

        return processed_barcode, label, att_mask


def representations_from_df(
    filename,
    embedder,
    batch_size=128,
    save_embeddings=True,
    dataset="BIOSCAN-5M",
    embeddings_folder="embeddings/",
    target="species",
):
    """
    Extract (or load cached) embeddings for all sequences in a CSV file.

    Returns a dict with keys ``"data"`` (np.ndarray, N×D) and ``"ids"`` (np.ndarray, N).
    Embeddings are saved to / loaded from a pickle file under
    ``{embeddings_folder}/{dataset}/{embedder.name}/{stem}.pickle``.
    """
    backbone = embedder.name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_fname = None

    if save_embeddings:
        backbone_folder = os.path.join(embeddings_folder, dataset, backbone)
        os.makedirs(backbone_folder, exist_ok=True)
        prefix = os.path.splitext(os.path.basename(filename))[0]
        out_fname = os.path.join(backbone_folder, f"{prefix}.pickle")
        print(f"Embeddings cache: {out_fname}")
        if os.path.exists(out_fname):
            print("Found cached embeddings — loading from file.")
            with open(out_fname, "rb") as fh:
                return pickle.load(fh)

    dataset_val = DNADataset(
        file_path=filename,
        embedder=embedder,
        randomize_offset=False,
        max_length=660,
        dataset_format=dataset,
        target=target,
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, drop_last=False,
        shuffle=False, pin_memory=True,
    )

    rep_type = getattr(embedder, "representation_type", "tokens")
    use_cls = getattr(embedder, "use_cls_token", False)

    embeddings_list = []
    id_list = []

    with torch.no_grad():
        for sequences, _id, att_mask in tqdm(dataloader_val, desc=f"Embedding ({backbone})"):
            sequences = sequences.view(-1, sequences.shape[-1]).to(device)
            att_mask = att_mask.view(-1, att_mask.shape[-1]).to(device)

            if backbone == "BarcodeMAE":
                output = embedder.model(sequences, att_mask)
                if hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hs = output.hidden_states
                    hidden_states = hs[-1] if isinstance(hs, (list, tuple)) else hs
                elif hasattr(output, "last_hidden_state"):
                    hidden_states = output.last_hidden_state
                else:
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]

                # Mean pooling; exclude CLS position when representation_type="tokens"
                seq_mask = att_mask.float()
                if use_cls and rep_type == "tokens":
                    seq_mask = seq_mask.clone()
                    seq_mask[:, 0] = 0.0
                sum_emb = (hidden_states * seq_mask.unsqueeze(-1)).sum(1)
                sum_mask = seq_mask.sum(1, keepdim=True).clamp(min=1)
                out = sum_emb / sum_mask

            elif backbone == "NT":
                out = embedder.model(sequences, output_hidden_states=True, attention_mask=att_mask)[
                    "hidden_states"
                ][-1]
                n_emb = att_mask.sum(axis=1)
                mask_exp = att_mask.unsqueeze(2).expand(-1, -1, embedder.hidden_size)
                out = torch.div((out * mask_exp).sum(axis=1).t(), n_emb).t()

            elif backbone == "Hyena_DNA":
                out = embedder.model(sequences)
                n_emb = att_mask.sum(axis=1)
                mask_exp = att_mask.unsqueeze(2).expand(-1, -1, embedder.hidden_size)
                out = torch.div((out * mask_exp).sum(axis=1).t(), n_emb).t()

            elif backbone in ["DNABERT", "DNABERT-2", "DNABERT-S"]:
                out = embedder.model(sequences, attention_mask=att_mask)[0]
                n_emb = att_mask.sum(axis=1)
                mask_exp = att_mask.unsqueeze(2).expand(-1, -1, embedder.hidden_size)
                out = torch.div((out * mask_exp).sum(axis=1).t(), n_emb).t()

            elif backbone == "BarcodeBERT":
                out = embedder.model(sequences, att_mask).hidden_states[-1]
                n_emb = att_mask.sum(axis=1)
                mask_exp = att_mask.unsqueeze(2).expand(-1, -1, embedder.hidden_size)
                out = torch.div((out * mask_exp).sum(axis=1).t(), n_emb).t()

            else:
                raise ValueError(f"Unknown backbone: {backbone!r}")

            embeddings_list.append(out.cpu().numpy())
            id_list.append(_id)

    all_embeddings = np.vstack(embeddings_list)
    # id_list entries are either tensors (integer labels) or lists of strings
    if isinstance(id_list[0], torch.Tensor):
        all_ids = torch.cat(id_list).numpy()
    else:
        all_ids = np.array([item for batch in id_list for item in batch])

    result = {"data": all_embeddings, "ids": all_ids}

    if save_embeddings:
        with open(out_fname, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return result


def zsc_pipeline(X, y_true, metric="cosine", n_neighbours=10, n_clusters=None):
    """
    UMAP dimensionality reduction → Agglomerative Clustering → AMI evaluation.

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
        run_name = f"zsc_{os.path.basename(os.path.dirname(config.pretrained_checkpoint_path))}"
        wandb.init(project="BarcodeBERT", name=run_name, config=vars(config))

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
        from torchtext.vocab import vocab as build_vocab
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
    elif config.dataset_name == "CANADA-1.5M":
        target_level = config.taxon + "_name"
    elif config.dataset_name == "BIOSCAN-5M":
        target_level = config.taxon + "_index"
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")

    # --- Wrap model + tokenizer into the embedder interface ---
    rep_type = getattr(config, "representation_type", "tokens")
    use_cls = getattr(config, "use_cls_token", False)
    bert_cfg = pre_checkpoint.get("bert_config", None)
    if bert_cfg is None:
        hidden_size = 768
    elif isinstance(bert_cfg, dict):
        hidden_size = bert_cfg.get("hidden_size", 768)
    else:
        hidden_size = bert_cfg.hidden_size
    embedder = ModelEmbedder(
        model=model,
        tokenizer=tokenizer,
        name="BarcodeMAE",
        hidden_size=hidden_size,
        representation_type=rep_type,
        use_cls_token=use_cls,
    )

    # --- Extract representations (batched, with pickle caching) ---
    embeddings_dir = getattr(config, "embeddings_dir", "embeddings/")
    print(f"\nExtracting representations (type={rep_type}) for supervised_test...", flush=True)
    emb_test = representations_from_df(
        os.path.join(config.data_dir, "supervised_test.csv"),
        embedder,
        dataset=config.dataset_name,
        embeddings_folder=embeddings_dir,
        target=target_level,
    )
    print(f"Extracting representations (type={rep_type}) for unseen...", flush=True)
    emb_unseen = representations_from_df(
        os.path.join(config.data_dir, "unseen.csv"),
        embedder,
        dataset=config.dataset_name,
        embeddings_folder=embeddings_dir,
        target=target_level,
    )

    X = np.vstack([emb_test["data"], emb_unseen["data"]])
    y = np.hstack([emb_test["ids"], emb_unseen["ids"]])

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
    rep_type = getattr(config, "representation_type", "tokens")
    with open("ZSC_RESULTS.txt", "a") as f:
        f.write(f"\n{config.run_name}_{model_name}_{rep_type}\t{ami:.4f}")

    if config.log_wandb:
        import wandb
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
    group.add_argument(
        "--embeddings-dir", "--embeddings_dir",
        dest="embeddings_dir",
        default="embeddings/", type=str, metavar="PATH",
        help="Directory for caching extracted embeddings. Default: %(default)s",
    )
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