#!/usr/bin/env python
"""One-off diagnostic: print the exact tensor shapes DNABERT-2's forward pass
returns for a small batch, to debug the (32) vs (660) shape mismatch seen
under batched embedding extraction in knn_its_clean.py.

Usage:
    python inspect_dnabert2_shapes.py
"""

import torch

from barcodebert.external_models import load_external_model

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, tokenizer = load_external_model("zhihan1996/DNABERT-2-117M", device=device, max_length=660, model_cls="auto")

sequences = [
    "ACGT" * 165,  # 660 bp
    "TGCA" * 165,
    "AACCGGTT" * 82 + "AACC",  # 660 bp
]

ids_list, mask_list = [], []
for seq in sequences:
    x, att_mask = tokenizer(seq)
    print(f"tokenizer output: x.shape={tuple(x.shape)}, att_mask.shape={tuple(att_mask.shape)}")
    ids_list.append(x)
    mask_list.append(att_mask)

x = torch.stack(ids_list).to(device)
att_mask = torch.stack(mask_list).to(device)
print(f"\nstacked batch: x.shape={tuple(x.shape)}, att_mask.shape={tuple(att_mask.shape)}")

with torch.no_grad():
    output = model(x, att_mask)

print(f"\noutput type: {type(output)}")
if hasattr(output, "last_hidden_state"):
    lhs = output.last_hidden_state
    print(f"output.last_hidden_state: {'None' if lhs is None else tuple(lhs.shape)}")
if hasattr(output, "hidden_states"):
    hs = output.hidden_states
    if hs is None:
        print("output.hidden_states: None")
    else:
        print(f"output.hidden_states: tuple of {len(hs)}, each shape {tuple(hs[0].shape)}")
        print(f"output.hidden_states[-1].shape: {tuple(hs[-1].shape)}")
try:
    print(f"output[-1] shape (if indexable): {tuple(output[-1].shape)}")
except Exception as e:
    print(f"output[-1] failed: {e}")

from barcodebert.datasets import _extract_last_hidden_states
hidden_states = _extract_last_hidden_states(output)
print(f"\n_extract_last_hidden_states(output).shape: {tuple(hidden_states.shape)}")
print(f"att_mask.shape: {tuple(att_mask.shape)}")