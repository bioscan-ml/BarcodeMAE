#!/usr/bin/env python

"""
Scan model_checkpoints/<dataset>/*/finetune/*.pt, find checkpoints that have
completed at least --min-epochs epochs of finetuning, infer the taxonomic level
and representation type from the filename, then run evaluation on all test sets
and save per-checkpoint JSON results.

Usage (from BarcodeMAE/ directory):
    python barcodebert/eval_scan_and_test.py \
        --data-dir ./data/ITS-5M \
        --checkpoint-base ./model_checkpoints/ITS-5M \
        --results-dir ./eval_results \
        --min-epochs 10
"""

import argparse
import json
import os
import re

import torch
import torch.nn as nn

from barcodebert import utils
from barcodebert.datasets import DNADataset
from barcodebert.evaluation import evaluate
from barcodebert.finetuning import ClassificationModel
from barcodebert.io import load_pretrained_model

TAXA_CHOICES = ["genus", "species", "family"]
REPR_CHOICES = ["tokens", "cls", "jumbo_avg", "jumbo"]


def infer_taxa_repr(filename):
    """Parse taxonomic level and representation type from checkpoint filename.
    Expects pattern: ..._ft_<taxa>_<repr>.pt
    """
    stem = filename.replace(".pt", "")
    # Try every combination, longest repr name first to avoid partial matches
    for taxa in TAXA_CHOICES:
        for repr_type in sorted(REPR_CHOICES, key=len, reverse=True):
            suffix = f"_ft_{taxa}_{repr_type}"
            if f"_ft_{taxa}_{repr_type}" in stem:
                return taxa, repr_type
    return None, None


def check_epoch(ckpt_path, min_epochs, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        epoch = ckpt.get("epoch", 0)
        return epoch >= min_epochs, epoch, ckpt
    except Exception as e:
        print(f"  WARNING: could not load {ckpt_path}: {e}")
        return False, 0, None


def evaluate_checkpoint(ckpt_path, ckpt, taxa, repr_type, data_dir, batch_size, cpu_workers, device):
    ft_config = ckpt.get("config", None)

    # Resolve tokenization settings from stored config
    k_mer = getattr(ft_config, "k_mer", 6)
    stride = getattr(ft_config, "stride", k_mer)
    max_len = getattr(ft_config, "max_len", 660)
    tokenizer = getattr(ft_config, "tokenizer", "kmer")
    bpe_path = getattr(ft_config, "bpe_path", "./")
    tokenize_n = getattr(ft_config, "tokenize_n_nucleotide", False)
    use_cls = getattr(ft_config, "use_cls_token", False)

    dataset_args = {
        "k_mer": k_mer,
        "stride": stride,
        "max_len": max_len,
        "tokenizer": tokenizer,
        "bpe_path": bpe_path,
        "tokenize_n_nucleotide": tokenize_n,
        "dataset_format": "ITS-5M",
        "use_cls_token": use_cls,
    }

    # Build datasets with shared label mapping from train
    dataset_train = DNADataset(
        file_path=os.path.join(data_dir, "trainset.fasta"),
        randomize_offset=False,
        **dataset_args,
        taxonomic_level=taxa,
        filter_unknown_labels=True,
    )
    shared_label2id = dataset_train.label2id
    num_labels = dataset_train.num_labels

    splits = {
        "Val":                 "trainset_valid.fasta",
        "Test1 (Yeast)":       "test1.fasta",
        "Test2 (Filamentous)": "test2.fasta",
        "Test3 (MycoAI)":      "test3.fasta",
    }
    dl_kwargs = {
        "batch_size": batch_size,
        "drop_last": False,
        "shuffle": False,
        "worker_init_fn": utils.worker_seed_fn,
        "num_workers": cpu_workers,
        "pin_memory": device.type != "cpu",
    }
    dataloaders = {}
    for name, fname in splits.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping {name}")
            continue
        ds = DNADataset(
            file_path=fpath,
            randomize_offset=False,
            **dataset_args,
            taxonomic_level=taxa,
            label2id=shared_label2id,
            filter_unknown_labels=True,
        )
        dataloaders[name] = torch.utils.data.DataLoader(ds, **dl_kwargs)

    # Strip DDP 'module.' prefix from state dict up front
    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # Read num_labels from the saved classifier head so we don't depend on the local
    # dataset having identical label coverage to the training run.
    if "classifier.weight" in state_dict:
        ckpt_num_labels = state_dict["classifier.weight"].shape[0]
    else:
        ckpt_num_labels = num_labels

    # Detect whether the finetuned model used a jumbo encoder — ground truth from state_dict.
    is_jumbo = "base_model.jumbo_cls_tokens" in state_dict

    # Build the encoder (pre_model).  Try the pretrained path first; if it either doesn't
    # exist or returns the wrong architecture (e.g. non-jumbo when the finetuned weights
    # have jumbo keys), fall back to reconstructing the encoder from the bert_config stored
    # in the finetuned checkpoint plus jumbo params inferred directly from tensor shapes.
    pretrained_path = getattr(ft_config, "pretrained_checkpoint_path", None)
    pre_model = None
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            pre_model, _ = load_pretrained_model(pretrained_path, device=device)
            # Sanity-check: if the checkpoint has jumbo weights the encoder must also be jumbo.
            if is_jumbo and not hasattr(pre_model, "jumbo_cls_tokens"):
                print(
                    "  WARNING: pretrained path returned a non-jumbo encoder but the "
                    "finetuned state dict has jumbo weights — rebuilding encoder from config."
                )
                pre_model = None
        except Exception as exc:
            print(f"  WARNING: load_pretrained_model failed ({exc}); rebuilding encoder from config.")
            pre_model = None

    if pre_model is None:
        bert_config_dict = ckpt.get("bert_config")
        if bert_config_dict is None:
            raise ValueError(
                "No pretrained checkpoint path found in config and no bert_config in "
                "checkpoint. Cannot reconstruct encoder architecture."
            )
        from transformers import BertConfig, BertForTokenClassification, BertModel

        from barcodebert.jumbo_transformer import create_jumbo_transformer_model

        _bert_cfg = BertConfig(**bert_config_dict)
        if is_jumbo:
            # Infer jumbo params directly from state_dict tensor shapes — more reliable
            # than trusting the finetuning config, which may not have --jumbo set.
            jumbo_multiplier = state_dict["base_model.jumbo_cls_tokens"].shape[1]
            mlp_w_key = "base_model.bert.encoder.layer.0.jumbo_handler.jumbo_mlp.1.weight"
            if mlp_w_key in state_dict:
                jumbo_width = jumbo_multiplier * _bert_cfg.hidden_size
                mlp_expansion_factor = state_dict[mlp_w_key].shape[0] // jumbo_width
            else:
                mlp_expansion_factor = getattr(ft_config, "jumbo_mlp_expansion", 2)
            share_jumbo_layers = getattr(ft_config, "share_jumbo_layers", False)
            pre_model = create_jumbo_transformer_model(
                _bert_cfg,
                jumbo_multiplier=jumbo_multiplier,
                share_jumbo_mlp_across_layers=share_jumbo_layers,
                mlp_expansion_factor=mlp_expansion_factor,
            )
        elif getattr(ft_config, "arch", "maelm") == "maelm":
            pre_model = BertModel(_bert_cfg)
        else:
            pre_model = BertForTokenClassification(_bert_cfg)

    pre_model.classifier = nn.Identity()
    model = ClassificationModel(pre_model, ckpt_num_labels, representation_type=repr_type)

    # For "jumbo" repr_type the classifier is lazy-initialized on the first forward pass,
    # so it is absent from model.state_dict().  Pre-create it from the checkpoint shape
    # so that load_state_dict() has a matching key to write into.
    if repr_type == "jumbo" and "classifier.weight" in state_dict:
        flat_dim = state_dict["classifier.weight"].shape[1]
        model.classifier = nn.Linear(flat_dim, ckpt_num_labels)
        model._classifier_initialized = True
        model.hidden_size = flat_dim

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    results = {}
    for name, dl in dataloaders.items():
        print(f"    [{name}] evaluating {len(dl.dataset)} samples...", flush=True)
        stats = evaluate(dataloader=dl, model=model, device=device, partition_name=name, is_distributed=False)
        results[name] = stats
        print(f"      accuracy={stats['accuracy']:.2f}%  f1-macro={stats['f1-macro']:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Scan and evaluate all completed finetuned checkpoints.")
    parser.add_argument("--data-dir", "--data_dir", required=True)
    parser.add_argument("--checkpoint-base", "--checkpoint_base",
                        default="./model_checkpoints/ITS-5M",
                        help="Root directory containing run subfolders.")
    parser.add_argument("--results-dir", "--results_dir", default="./eval_results")
    parser.add_argument("--min-epochs", "--min_epochs", type=int, default=10,
                        help="Minimum completed epochs to consider a checkpoint done.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=128)
    parser.add_argument("--cpu-workers", "--cpu_workers", type=int, default=4)
    parser.add_argument("--no-cuda", "--no_cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-evaluate even if JSON result already exists.")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.results_dir, exist_ok=True)

    # Scan for all finetuned checkpoints
    candidates = []
    for run_name in sorted(os.listdir(args.checkpoint_base)):
        finetune_dir = os.path.join(args.checkpoint_base, run_name, "finetune")
        if not os.path.isdir(finetune_dir):
            continue
        for fname in sorted(os.listdir(finetune_dir)):
            if not fname.endswith(".pt"):
                continue
            # Skip encoder-only or best-copy checkpoints
            if "best_finetune" in fname or fname.endswith("_encoder.pt"):
                continue
            ckpt_path = os.path.join(finetune_dir, fname)
            taxa, repr_type = infer_taxa_repr(fname)
            if taxa is None:
                print(f"SKIP (can't parse taxa/repr): {fname}")
                continue
            candidates.append((run_name, ckpt_path, fname, taxa, repr_type))

    print(f"\nFound {len(candidates)} finetuned checkpoint(s) to check.\n")

    evaluated = 0
    skipped_epoch = 0
    skipped_exists = 0

    for run_name, ckpt_path, fname, taxa, repr_type in candidates:
        out_fname = fname.replace(".pt", ".json")
        out_path = os.path.join(args.results_dir, out_fname)

        if os.path.exists(out_path) and not args.overwrite:
            print(f"SKIP (result exists): {out_fname}")
            skipped_exists += 1
            continue

        done, epoch, ckpt = check_epoch(ckpt_path, args.min_epochs, device)
        if not done:
            print(f"SKIP (only {epoch} epochs < {args.min_epochs}): {fname}")
            skipped_epoch += 1
            continue

        print(f"\n{'='*60}")
        print(f"Run:   {run_name}")
        print(f"File:  {fname}")
        print(f"Taxa:  {taxa} | Repr: {repr_type} | Epoch: {epoch}")
        print(f"{'='*60}")

        try:
            results = evaluate_checkpoint(
                ckpt_path=ckpt_path,
                ckpt=ckpt,
                taxa=taxa,
                repr_type=repr_type,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                cpu_workers=args.cpu_workers,
                device=device,
            )
            results["_meta"] = {
                "run_name": run_name,
                "checkpoint": ckpt_path,
                "taxa": taxa,
                "repr_type": repr_type,
                "epoch": epoch,
            }
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved: {out_path}")
            evaluated += 1
        except Exception as e:
            print(f"  ERROR evaluating {fname}: {e}")

    print(f"\n{'='*60}")
    print(f"Done. Evaluated: {evaluated} | Skipped (done): {skipped_exists} | Skipped (epochs): {skipped_epoch}")
    print(f"Results in: {args.results_dir}/")


if __name__ == "__main__":
    main()