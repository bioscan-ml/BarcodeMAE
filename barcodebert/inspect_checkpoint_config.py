#!/usr/bin/env python
"""One-off diagnostic: print the training config (hyperparameters) stored
inside one of our own checkpoints, to check what values were actually used
(e.g. aux-loss-weight, cls-taxonomy-loss-weight, learning rate, num-pairs)
vs. what the SLURM scripts currently say and what the manuscript states.

Checkpoints save the full training argparse.Namespace under ckpt["config"]
(see barcodebert/io.py's safe_save_model), so this works for both
checkpoint.pt and checkpoint_encoder.pt.

Usage:
    python inspect_checkpoint_config.py --checkpoint main_checkpoints_final/ITS-5M/final_its_k6_6L6H_6DL6DH_maelm_cls_binary/checkpoint_encoder.pt
    python inspect_checkpoint_config.py --checkpoint main_checkpoints_final/BIOSCAN-5M/final_k6_6L6H_6DL6DH_maelm_cls_ce/checkpoint_encoder.pt
"""

import argparse

import torch

RELEVANT_KEYS = [
    "aux_loss_weight", "aux_loss_warmup_epochs", "cls_taxonomy_loss_weight",
    "aux_loss_type", "enable_cls_taxonomy", "use_cls_token",
    "triplet_margin", "taxonomy_level", "taxonomy_max_pairs",
    "k_classes", "m_per_class",
    "lr", "weight_decay", "batch_size", "epochs",
    "masked_loss_weight", "mask_token_ratio", "random_token_ratio",
    "arch", "k_mer", "stride", "n_layers", "n_heads",
    "decoder_n_layers", "decoder_n_heads",
]


def cli():
    p = argparse.ArgumentParser(description="Print a checkpoint's stored training config.")
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    print(f"Loading {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if "config" not in ckpt:
        print("No 'config' key found in this checkpoint!")
        return

    cfg = ckpt["config"]
    print("=" * 80)
    print("Relevant hyperparameters found in checkpoint config:")
    print("=" * 80)
    for key in RELEVANT_KEYS:
        val = getattr(cfg, key, "<not present>")
        print(f"  {key:<28} = {val}")

    print("\n" + "=" * 80)
    print("Full config (all attributes):")
    print("=" * 80)
    for key, val in sorted(vars(cfg).items()):
        print(f"  {key:<28} = {val}")


if __name__ == "__main__":
    cli()