#!/usr/bin/env python
"""One-off inspection tool: print the structure of a MycoAI-BERT.pt /
MycoAI-CNN.pt checkpoint (from https://zenodo.org/records/10904344) so we
can figure out how to extract embeddings (not just final classify()
predictions) for the external-baseline KNN pipeline.

These checkpoints are full pickled Python objects (custom classes from the
mycoai package, e.g. a wrapper around mycoai.modules.transformers.BERT or
mycoai.modules.cnns.SimpleCNN/ResNet), not a HuggingFace-style state_dict,
so torch.load() needs `mycoai` importable to unpickle them -- run this
inside the venv that has mycoai-its installed.

Usage:
    python inspect_mycoai_checkpoint.py --checkpoint data/ITS-5M/models/MycoAI-BERT.pt
    python inspect_mycoai_checkpoint.py --checkpoint data/ITS-5M/models/MycoAI-CNN.pt
"""

import argparse

import torch


def describe(obj, prefix=""):
    print(f"{prefix}type: {type(obj)}")
    print(f"{prefix}module: {type(obj).__module__}")

    if hasattr(obj, "__dict__"):
        print(f"{prefix}instance attributes:")
        for k, v in vars(obj).items():
            if isinstance(v, torch.nn.Module):
                print(f"{prefix}  {k}: nn.Module -> {type(v).__name__}")
            elif torch.is_tensor(v):
                print(f"{prefix}  {k}: Tensor{tuple(v.shape)}")
            else:
                v_repr = repr(v)
                if len(v_repr) > 120:
                    v_repr = v_repr[:120] + "..."
                print(f"{prefix}  {k}: {v_repr}")

    print(f"{prefix}public methods/callables:")
    for name in dir(obj):
        if name.startswith("_"):
            continue
        attr = getattr(obj, name, None)
        if callable(attr):
            print(f"{prefix}  .{name}(...)")

    if isinstance(obj, torch.nn.Module):
        print(f"{prefix}--- nn.Module structure (print(model)) ---")
        print(obj)


def get_parser():
    p = argparse.ArgumentParser(description="Inspect a MycoAI .pt checkpoint's structure.")
    p.add_argument("--checkpoint", required=True, help="Path to MycoAI-BERT.pt or MycoAI-CNN.pt")
    return p


def cli():
    args = get_parser().parse_args()
    print(f"Loading {args.checkpoint} ...")
    obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print("=" * 100)
    describe(obj)
    print("=" * 100)


if __name__ == "__main__":
    cli()