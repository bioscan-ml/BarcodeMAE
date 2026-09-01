"""Shared loading/embedding utilities for evaluating BarcodeMamba / BarcodeMamba+
checkpoints (https://github.com/bioscan-ml/BarcodeMamba-dev, branch
GTCtech-BarcodeMambaPlus-release) inside our own KNN/ZSC pipeline.

BarcodeMamba is not a HuggingFace AutoModel -- it is a standalone repo with its
own Mamba2-based model code (utils/barcode_mamba.py) and its own char/k-mer/BPE
tokenizers. Checkpoint folders in the wild have been seen in two different
layouts (neither matches utils/probing_utils.py::get_pretrained_barcodemamba's
hardcoded <dir>/.hydra/config.yaml + <dir>/checkpoints/last.ckpt assumption
exactly), so this module resolves paths flexibly instead of assuming one:
  - BIOSCAN-5M release (Google Drive): <dir>/.hydra/config.yaml, <dir>/last.ckpt
    (NOT inside a checkpoints/ subfolder), <dir>/bpe_tokenizer.pkl
  - UNITE/ITS release (GitHub v0.2.0): <dir>/config.yaml, <dir>/model.ckpt, and
    a single bpe_tokenizer.pkl shared by both size variants (release-level
    asset, not per-checkpoint).

BPE tokenizer note: the released bpe_tokenizer.pkl was pickled under the class
path "utils.mycoai.data.encoders.BytePairEncoder", but that submodule does not
actually exist anywhere in this repo's vendored utils/mycoai/ copy (confirmed
against the full repo tree on the GTCtech-BarcodeMambaPlus-release branch --
its own merge comment claiming the data/ subtree was included is stale). The
real mycoai-its PyPI package (already installed for the MycoAI-BERT/CNN
baselines elsewhere in this project) has a byte-for-byte compatible
mycoai.data.encoders.BytePairEncoder (same vocab_size/sp/length attributes,
same .encode() implementation), so load_bpe_tokenizer() aliases
utils.mycoai.data.encoders -> the real mycoai.data.encoders in sys.modules
before unpickling, sidestepping the vendored gap entirely.
"""

import glob
import importlib.util
import os
import sys

import torch


def _find_first(dir_path, candidates):
    for rel in candidates:
        p = os.path.join(dir_path, rel)
        if os.path.isfile(p):
            return p
    return None


def load_barcodemamba(repo_path, checkpoint_dir, checkpoint_name=None):
    """Returns (model, config). repo_path must be a local clone of
    bioscan-ml/BarcodeMamba-dev (branch GTCtech-BarcodeMambaPlus-release), used
    only for the utils.barcode_mamba.BarcodeMamba class (needs mamba_ssm)."""
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    # Running this as a script (python barcodebert/knn_probing_barcodemamba.py)
    # puts barcodebert/ itself on sys.path (right after repo_path), which has
    # its own flat utils.py. BarcodeMamba-dev's utils/ has no __init__.py (a
    # namespace package), and namespace-package resolution only wins if NO
    # regular module of that name is found anywhere else on sys.path first --
    # since barcodebert/utils.py IS a regular module and sits right after
    # repo_path, Python's finder hits it and stops, discarding the namespace
    # candidate entirely. This happens on a fresh import too (not just a stale
    # cache), so clearing sys.modules alone doesn't fix it. Instead, force
    # "utils" to resolve to repo_path/utils by constructing and registering
    # the module ourselves, bypassing sys.path search order altogether.
    for mod_name in list(sys.modules):
        if mod_name == "utils" or mod_name.startswith("utils."):
            del sys.modules[mod_name]
    utils_dir = os.path.join(repo_path, "utils")
    init_path = os.path.join(utils_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "utils",
        init_path if os.path.isfile(init_path) else None,
        submodule_search_locations=[utils_dir],
    )
    utils_pkg = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = utils_pkg
    if spec.loader is not None:
        spec.loader.exec_module(utils_pkg)

    from omegaconf import OmegaConf as o
    from utils.barcode_mamba import BarcodeMamba

    try:
        o.register_new_resolver("eval", eval)
        o.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)
    except Exception:
        pass  # already registered by an earlier call in this process

    config_path = _find_first(checkpoint_dir, [".hydra/config.yaml", "config.yaml"])
    if config_path is None:
        raise FileNotFoundError(f"No .hydra/config.yaml or config.yaml found under {checkpoint_dir}")
    config = o.load(config_path)

    if checkpoint_name is not None:
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
    else:
        ckpt_path = _find_first(checkpoint_dir, ["checkpoints/last.ckpt", "last.ckpt", "model.ckpt"])
        if ckpt_path is None:
            matches = glob.glob(os.path.join(checkpoint_dir, "*.ckpt")) + \
                glob.glob(os.path.join(checkpoint_dir, "checkpoints", "*.ckpt"))
            ckpt_path = matches[0] if matches else None
    if ckpt_path is None:
        raise FileNotFoundError(f"No .ckpt file found under {checkpoint_dir}")

    model = BarcodeMamba(**config.model, use_head=config.dataset.phase)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    model_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_dict, strict=False)
    print(f"  Loaded {ckpt_path}: {len(missing)} missing, {len(unexpected)} unexpected keys "
          f"(expect 0 missing; unexpected is normal for pl trainer/metric bookkeeping keys)")
    return model, config


def load_bpe_tokenizer(pkl_path):
    """Unpickles a BarcodeMamba+ bpe_tokenizer.pkl, aliasing the pickled class
    path utils.mycoai.data.encoders.BytePairEncoder to the real, already-
    installed mycoai-its package (see module docstring)."""
    import pickle
    import types

    import mycoai
    import mycoai.data
    import mycoai.data.encoders

    if "utils" not in sys.modules:
        sys.modules["utils"] = types.ModuleType("utils")
    if "utils.mycoai" not in sys.modules:
        sys.modules["utils.mycoai"] = mycoai
        sys.modules["utils"].mycoai = mycoai
    sys.modules["utils.mycoai.data"] = mycoai.data
    sys.modules["utils.mycoai.data.encoders"] = mycoai.data.encoders

    with open(pkl_path, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer


def embed_sequences(model, tokenizer, tokenizer_name, sequences, max_length=660):
    """Extracts mean-pooled BarcodeMamba hidden states for a sequence iterable.
    Mirrors the repo's own embedding logic per tokenizer type:
      - bpe: tokenizer.encode(seq) already returns a padded LongTensor
        (CLS + BPE tokens + SEP + PAD, to tokenizer.length) -- no attention
        mask, no separate padding call needed.
      - char: HF-style tokenizer(seq, ...)["input_ids"], padded/truncated to
        max_length.
      - k_mer: tokenizer(seq) -> (ids, attention_mask).
    """
    assert tokenizer_name in ("bpe", "char", "k_mer"), f"Unsupported tokenizer_name: {tokenizer_name!r}"
    from tqdm import tqdm
    import numpy as np

    embeddings = []
    with torch.no_grad():
        for seq in tqdm(sequences):
            if tokenizer_name == "bpe":
                x = tokenizer.encode(seq)
            elif tokenizer_name == "char":
                tokenizer.pad_token = "N"
                x = tokenizer(
                    seq, add_special_tokens=False, padding="max_length",
                    max_length=max_length, truncation=True,
                )["input_ids"]
                x = torch.tensor(x, dtype=torch.int64)
            else:  # k_mer
                x, _ = tokenizer(seq)
                x = torch.tensor(x, dtype=torch.int64)
            x = x.unsqueeze(0).cuda()
            h = model.get_hidden_states(x)
            h = h.mean(1)
            embeddings.append(h.cpu().numpy())
    return np.squeeze(np.array(embeddings), 1)