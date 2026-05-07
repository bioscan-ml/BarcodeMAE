"""
Input/output utilities.
"""

import os
from inspect import getsourcefile

import torch
from torch import nn
from transformers import BertConfig, BertForTokenClassification, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from barcodebert.jumbo_transformer import create_jumbo_transformer_model
from barcodebert.jumbo_transformer_with_taxonomy import (
    create_jumbo_transformer_with_taxonomy,
)
from .utils import remove_extra_pre_fix

PACKAGE_DIR = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))


class RegisterBertModel(nn.Module):
    """
    Wraps a BertModel to prepend learnable register tokens before the transformer layers,
    replicating the forward pass used in MAELMModel during pretraining with n_registers > 0.
    """

    def __init__(self, bert_model, register_tokens):
        super().__init__()
        self.bert = bert_model
        self.register_tokens = nn.Parameter(register_tokens.clone())
        self.config = bert_model.config
        self.n_registers = register_tokens.shape[1]

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Embed the input sequence (word + position), matching MAELMModel line 146-148
        seq_embeds = self.bert.embeddings(input_ids=input_ids, position_ids=position_ids)

        # Prepend register tokens with no positional embedding, matching MAELMModel line 152-153
        registers = self.register_tokens.expand(batch_size, -1, -1)
        combined_embeds = torch.cat([registers, seq_embeds], dim=1)

        # Build attention mask: registers always attend, matching MAELMModel line 155-165
        reg_mask = torch.ones(batch_size, self.n_registers, device=device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([reg_mask, attention_mask], dim=1)
        extended_mask = self.bert.get_extended_attention_mask(
            combined_mask, (batch_size, combined_mask.shape[1]), device=device
        )

        # Run through transformer layers directly, matching MAELMModel line 168
        encoder_out = self.bert.encoder(combined_embeds, attention_mask=extended_mask)
        all_hidden = encoder_out.last_hidden_state  # (B, n_registers + seq_len, D)

        # Strip register positions, matching MAELMModel line 172
        seq_hidden = all_hidden[:, self.n_registers:, :]  # (B, seq_len, D)

        output = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=seq_hidden,
            hidden_states=(seq_hidden,),
        )
        # Expose register outputs for downstream representation extraction
        output.register_hidden_states = all_hidden[:, : self.n_registers, :]  # (B, R, D)
        return output


def get_project_root() -> str:
    return os.path.dirname(PACKAGE_DIR)


def safe_save_model(modules, checkpoint_path=None, config=None, decoder_config=None, **kwargs):
    """
    Save a model to a checkpoint file, along with any additional data.

    Parameters
    ----------
    modules : dict
        A dictionary of modules to save. The keys are the names of the modules
        and the values are the modules themselves.
    checkpoint_path : str, optional
        Path to the checkpoint file. If not provided, the path will be taken
        from the config object.
    config : :class:`argparse.Namespace`, optional
        A configuration object containing the checkpoint path.
    **kwargs
        Additional data to save to the checkpoint file.
    """
    if checkpoint_path is not None:
        pass
    elif config is not None and hasattr(config, "checkpoint_path"):
        checkpoint_path = config.checkpoint_path
    else:
        raise ValueError("No checkpoint path provided")
    print(f"\nSaving model to {checkpoint_path}")
    directory = os.path.dirname(checkpoint_path)
    os.makedirs(directory, exist_ok=True)
    # Save to a temporary file first, then move the temporary file to the target
    # destination. This is to prevent clobbering the checkpoint with a partially
    # saved file, in the event that the saving process is interrupted. Saving
    # the checkpoint takes a little while and can be disrupted by preemption,
    # whereas moving the file is an atomic operation.
    tmp_a, tmp_b = os.path.split(checkpoint_path)
    tmp_fname = os.path.join(tmp_a, ".tmp." + tmp_b)
    data = {k: v.state_dict() if hasattr(v, "state_dict") else v for k, v in modules.items()}
    data.update(kwargs)
    if config is not None:
        data["config"] = config
    if decoder_config is not None:
        data["decoder_config"] = decoder_config
    torch.save(data, tmp_fname)
    os.rename(tmp_fname, checkpoint_path)
    print(f"Saved model to  {checkpoint_path}")


def load_pretrained_model(checkpoint_path, device=None):
    """
    Load a pretrained model from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the pretrained checkpoint file.

    Returns
    -------
    model : torch.nn.Module
        The pretrained model.
    ckpt : dict
        The contents of the checkpoint file.
    """
    print(f"\nLoading model from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    bert_config = BertConfig(**ckpt["bert_config"])
    #print(bert_config)

    # Check if this is a jumbo transformer model
    if hasattr(ckpt["config"], "jumbo") and ckpt["config"].jumbo:
        # Get jumbo configuration parameters
        jumbo_multiplier = getattr(ckpt["config"], "jumbo_multiplier", 6)
        share_jumbo_layers = getattr(ckpt["config"], "share_jumbo_layers", False)
        enable_genus_classification = getattr(ckpt["config"], "enable_genus_classification", False)
        mlp_expansion_factor = getattr(ckpt["config"], "jumbo_mlp_expansion", 2)
        # Check if taxonomy classification is enabled
        if enable_genus_classification and ckpt["config"].arch != "maelm":
            print("Loading JumboTransformerWithTaxonomy")
            model = create_jumbo_transformer_with_taxonomy(
                bert_config,
                jumbo_multiplier=jumbo_multiplier,
                share_jumbo_mlp_across_layers=share_jumbo_layers,
                enable_taxonomy_classification=True,
                mlp_expansion_factor=mlp_expansion_factor,
            )
        else:
            print("Loading JumboBertForTokenClassification")
            model = create_jumbo_transformer_model(
                bert_config,
                jumbo_multiplier=jumbo_multiplier,
                share_jumbo_mlp_across_layers=share_jumbo_layers,
                mlp_expansion_factor=mlp_expansion_factor,
            )
    elif ckpt["config"].arch == "maelm":
        model = BertModel(bert_config)
    elif ckpt["config"].arch == "transformer":
        model = BertForTokenClassification(bert_config)
    else:
        raise ValueError(f"Unknown model architecture: {ckpt['config'].arch}")

    state_dict = remove_extra_pre_fix(ckpt["model"])

    # Full MAELMModel checkpoint: extract only encoder weights
    register_tokens = None
    if ckpt["config"].arch == "maelm" and "decoder_config" in ckpt:
        register_tokens = state_dict.get("register_tokens", None)  # save before stripping
        state_dict = {k[len("encoder.") :]: v for k, v in state_dict.items() if k.startswith("encoder.")}

    model.load_state_dict(state_dict)

    # Wrap BertModel with register token prepending if the model was trained with registers
    n_registers = getattr(ckpt["config"], "n_registers", 0)
    if ckpt["config"].arch == "maelm" and n_registers > 0 and register_tokens is not None:
        print(f"Wrapping encoder with RegisterBertModel ({n_registers} register tokens)")
        model = RegisterBertModel(model, register_tokens)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    cfg = ckpt["config"]
    total_step = ckpt.get("total_step", "N/A")
    steps_per_epoch = getattr(cfg, "steps_per_epoch", None)
    if total_step != "N/A" and steps_per_epoch:
        epochs_trained = total_step / steps_per_epoch
    else:
        epochs_trained = "N/A"
    print("\n--- Checkpoint Diagnostics ---")
    print(f"  Total steps trained:       {total_step}")
    print(f"  Epochs trained:            {epochs_trained}")
    print(f"  use_cls_token:             {getattr(cfg, 'use_cls_token', False)}")
    print(f"  enable_genus_classification:      {getattr(cfg, 'enable_genus_classification', False)}")
    print(f"  enable_taxonomy_classification:   {getattr(cfg, 'enable_taxonomy_classification', False)}")
    print(f"  enable_cls_taxonomy:              {getattr(cfg, 'enable_cls_taxonomy', False)}")
    jumbo = getattr(cfg, "jumbo", False)
    jumbo_multiplier = getattr(cfg, "jumbo_multiplier", 0) if jumbo else 0
    n_registers = getattr(cfg, "n_registers", 0) if not jumbo else 0
    print(f"  jumbo:                     {jumbo}")
    print(f"  number of jumbo tokens (J):{jumbo_multiplier}")
    print(f"  number of register tokens: {n_registers}")
    print("------------------------------\n")

    return model, ckpt


def load_pretrained_encoder(checkpoint_path, device=None):
    """
    Load ONLY the encoder from a pretrained checkpoint — no classifier, no
    taxonomy head, no decoder. Mirrors the loading path used by
    inspect_checkpoint.py (direct torch.load + rebuild BertConfig), then builds
    the minimal encoder architecture and loads just the encoder weights.

    Use this for finetuning when you want a clean backbone and a fresh
    classification head. Returns (encoder, ckpt).

    The returned encoder has:
      - jumbo pretraining: JumboBertForTokenClassification (its `.classifier`
        is untrained/unused; take embeddings from `.jumbo_tokens` or
        `.hidden_states` in the forward output)
      - maelm pretraining: BertModel (possibly wrapped in RegisterBertModel)
      - vanilla transformer pretraining: BertModel
    """
    print(f"\nLoading encoder from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ckpt["config"]
    bert_config = BertConfig(**ckpt["bert_config"])
    jumbo = getattr(cfg, "jumbo", False)

    # ── 1. Build encoder-only architecture ────────────────────────────────
    if jumbo:
        model = create_jumbo_transformer_model(
            bert_config,
            jumbo_multiplier=getattr(cfg, "jumbo_multiplier", 6),
            share_jumbo_mlp_across_layers=getattr(cfg, "share_jumbo_layers", False),
            mlp_expansion_factor=getattr(cfg, "jumbo_mlp_expansion", 2),
        )
        print("Built JumboBertForTokenClassification (encoder only — no taxonomy head)")
    elif cfg.arch == "maelm":
        model = BertModel(bert_config)
        print("Built BertModel (MAELM encoder)")
    else:
        model = BertModel(bert_config)
        print("Built BertModel (vanilla transformer encoder)")

    # ── 2. Strip head weights from state dict ─────────────────────────────
    state_dict = remove_extra_pre_fix(ckpt["model"])

    # MAELM: keep only encoder.* entries
    register_tokens = None
    if cfg.arch == "maelm" and "decoder_config" in ckpt:
        register_tokens = state_dict.get("register_tokens", None)
        state_dict = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}

    # Drop any head-only keys that aren't part of the encoder architecture
    target_keys = set(model.state_dict().keys())
    filtered = {}
    dropped = []
    for k, v in state_dict.items():
        if k in target_keys:
            filtered[k] = v
        else:
            dropped.append(k)

    if dropped:
        # Show a concise summary so the user can see what was stripped
        heads = sorted({k.split(".")[0] for k in dropped})
        print(f"Dropped {len(dropped)} non-encoder param groups: {heads}")

    missing = target_keys - set(filtered.keys())
    if missing:
        print(f"Warning: {len(missing)} target-model params have no checkpoint value "
              f"(will stay at init): {sorted(list(missing))[:10]}...")

    model.load_state_dict(filtered, strict=False)

    # ── 3. Register-token wrap for MAELM ──────────────────────────────────
    n_registers = getattr(cfg, "n_registers", 0)
    if cfg.arch == "maelm" and n_registers > 0 and register_tokens is not None:
        print(f"Wrapping encoder with RegisterBertModel ({n_registers} register tokens)")
        model = RegisterBertModel(model, register_tokens)

    model.eval()

    # ── 4. Diagnostics (same info as inspect_checkpoint) ──────────────────
    print("--- Encoder checkpoint summary ---")
    print(f"  arch ................ {getattr(cfg, 'arch', '?')}")
    print(f"  jumbo ............... {jumbo}")
    print(f"  use_cls_token ....... {getattr(cfg, 'use_cls_token', False)}")
    print(f"  epochs trained ...... {ckpt.get('epoch', '?')}")
    print(f"  total_step .......... {ckpt.get('total_step', '?')}")
    print(f"  max_accuracy ........ {ckpt.get('max_accuracy', '?')}")
    print(f"  best_epoch .......... {ckpt.get('best_epoch', '?')}")
    print("-----------------------------------\n")

    return model, ckpt
