"""Unified DNABERT-2 / BarcodeMAE training entrypoint.

Architectures
-------------
  --architecture bert              standard BERT MLM (bert_layers.py)
  --architecture maelm             basic MAELM encoder-decoder (maelm_model.py)
  --architecture maelm --jumbo     BarcodeMAELMModel with Jumbo CLS + taxonomy loss

Quick usage
-----------
  # BERT
  torchrun --nproc_per_node=4 main_train.py --architecture bert \
      --batch-size 64 --total-batch-size 4096

  # MAELM baseline
  torchrun --nproc_per_node=1 main_train.py --architecture maelm \
      --batch-size 64 --total-batch-size 4096

  # MAELM + Jumbo CLS + taxonomy (32 pos / 32 neg pairs, batch 128)
  torchrun --nproc_per_node=1 main_train.py --architecture maelm --jumbo \
      --n-layers 6 --n-heads 6 --jumbo-multiplier 6 \
      --k-classes 32 --m-per-class 2 \
      --batch-size 128 --species-vocab shards/species_vocab.json
"""

import argparse
import contextlib
import datetime
import glob
import inspect
import io
import json
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import webdataset as wds
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertConfig, get_cosine_schedule_with_warmup

try:
    import wandb
except ImportError:
    wandb = None

# Find barcodebert in parent directory in case it was not installed
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
# sys.path.append()
try:
    from barcodebert.maelm_model import MAELMModel as BarcodeMAELMModel
    from barcodebert.jumbo_taxonomy_classifier import compute_taxonomy_classification_loss
    from barcodebert.cls_taxonomy_classifier import (
        CLSTaxonomyClassifier,
        compute_cls_taxonomy_classification_loss,
    )
    BARCODEBERT_AVAILABLE = True
except ImportError:
    BarcodeMAELMModel = None
    compute_taxonomy_classification_loss = None
    CLSTaxonomyClassifier = None
    compute_cls_taxonomy_classification_loss = None
    BARCODEBERT_AVAILABLE = False

import monitor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_SAMPLES_PER_EPOCH = 32_387_832
TRUE_EPOCH_STEPS = 7_908


@dataclass
class TrainingConfig:
    # Architecture
    architecture: str = "bert"          # "bert" | "maelm"

    # Batch / sequence
    total_batch_size: int = 4096
    batch_size: int = 8
    max_seq_length: int = 512
    alibi_starting_size: int = 512

    # Optimiser / schedule
    max_lr: float = 5e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = int(TRUE_EPOCH_STEPS*0.15)
    max_steps: int = TRUE_EPOCH_STEPS
    weight_decay: float = 0.1
    mask_ratio: float = 0.15
    scheduler: str = "auto"             # "auto" | "linear" | "cosine"

    # Encoder / decoder depth
    n_layers: int = 12
    n_heads: int = 12
    decoder_n_layers: Optional[int] = None   # defaults to n_layers
    decoder_n_heads: Optional[int] = None    # defaults to n_heads
    maelm_model_type: str = "maelm_v2"

    # Jumbo CLS + taxonomy
    jumbo: bool = False
    jumbo_multiplier: int = 6
    jumbo_mlp_expansion: int = 2
    share_jumbo_layers: bool = True
    cls_loss_weight: float = 1.0
    species_vocab: Optional[str] = None

    # Standard CLS token + taxonomy
    use_cls_token: bool = False

    # Balanced sampling (streaming KClassMSample equivalent)
    k_classes: int = 0                  # 0 = disabled
    m_per_class: int = 2

    # Checkpointing / logging
    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "log"
    log_interval: int = 100
    train_shards_pattern: str = "shards/train-*.tar"
    shuffle_buffer: int = 10000
    num_workers: int = 4

    # Hardware
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"

    # Tokenizer / vocab
    num_special_tokens: int = 5
    mask_token_id: int = 4
    vocab_size: int = 4096

    # Misc
    finetune_data_path: Optional[str] = None

    # W&B
    use_wandb: bool = True
    wandb_project: str = "barcodemae"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    wandb_existing_run_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_num_cpu_available():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        warnings.warn("Unable to determine available CPUs, falling back to os.cpu_count().", RuntimeWarning, stacklevel=2)
        return os.cpu_count()


def npy_decoder(data: bytes) -> torch.Tensor:
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.int32)
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Dataset / data loading
# ---------------------------------------------------------------------------

def make_dataset(
    pattern,
    shuffle_buf: int = 10_000,
    resampled: bool = False,
    world_size: int = 1,
    require_label: bool = False,
):
    if world_size > 1:
        ds = wds.WebDataset(pattern, resampled=resampled, nodesplitter=wds.shardlists.split_by_node)
    else:
        ds = wds.WebDataset(pattern, resampled=resampled, shardshuffle=True)

    if resampled:
        ds = ds.with_epoch(TRAIN_SAMPLES_PER_EPOCH)
        ds = ds.with_length(TRAIN_SAMPLES_PER_EPOCH)

    ds = ds.shuffle(shuffle_buf)
    if require_label:
        ds = ds.to_tuple("__key__", "tokens", "attention_mask", "label").map_tuple(
            lambda x: x, npy_decoder, npy_decoder, npy_decoder
        )
    else:
        ds = ds.to_tuple("__key__", "tokens", "attention_mask").map_tuple(
            lambda x: x, npy_decoder, npy_decoder
        )
    return ds


def balanced_batch_pipeline(src, k: int, m: int, batch_size: int, buffer_size: int = None):
    """
    Streaming class-balanced batch builder for WebDataset.

    Each batch:
      - k*m labeled slots: k species × m samples  →  k*C(m,2) positive pairs
      - (batch_size - k*m) fill slots: random samples (labeled or unlabeled)

    E.g. k=32, m=2, batch_size=128  →  32 positive + 64 fill per batch.
    """
    n_labeled = k * m
    n_fill = batch_size - n_labeled
    if buffer_size is None:
        buffer_size = batch_size * 40

    species_buf = defaultdict(list)
    general_buf = deque(maxlen=buffer_size)

    for sample in src:
        key, tokens, mask, label_raw = sample
        label_val = int(label_raw.reshape(-1)[0])

        item = (key, tokens, mask, label_val)
        if label_val >= 0:
            species_buf[label_val].append(item)
        general_buf.append(item)

        eligible = [sp for sp, buf in species_buf.items() if len(buf) >= m]
        if len(eligible) < k:
            continue

        chosen = random.sample(eligible, k)
        labeled, labeled_keys = [], set()
        for sp in chosen:
            picked = random.sample(species_buf[sp], m)
            labeled.extend(picked)
            labeled_keys.update(x[0] for x in picked)
            for p in picked:
                species_buf[sp].remove(p)

        fill_candidates = [x for x in general_buf if x[0] not in labeled_keys]
        if len(fill_candidates) < n_fill:
            for item_ in labeled:
                species_buf[item_[3]].append(item_)
            continue

        fill = random.sample(fill_candidates, n_fill)
        batch = labeled + fill
        random.shuffle(batch)

        yield (
            [x[0] for x in batch],
            torch.stack([x[1] for x in batch]),
            torch.stack([x[2] for x in batch]),
            torch.tensor([x[3] for x in batch], dtype=torch.long),
        )


def setup_data_loader(config: TrainingConfig, ddp_world_size: int, device: str,
                      master_process: bool, resolved_shards):
    assert config.total_batch_size % (config.batch_size * ddp_world_size) == 0
    grad_accum_steps = config.total_batch_size // (config.batch_size * ddp_world_size)

    if master_process:
        print(f"total desired batch size: {config.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_dataset = make_dataset(
        resolved_shards,
        config.shuffle_buffer,
        resampled=True,
        world_size=ddp_world_size,
        require_label=(config.k_classes > 0),
    )
    cpu_workers = min(get_num_cpu_available(), config.num_workers)

    if ddp_world_size > 1:
        dataloader = iter(wds.WebLoader(
            train_dataset.batched(config.batch_size, partial=False),
            num_workers=cpu_workers,
            batch_size=None,
        ))
    else:
        dataloader = iter(torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            drop_last=True,
            sampler=None,
            shuffle=False,
            num_workers=cpu_workers,
            pin_memory=device != "cpu",
        ))

    return dataloader, grad_accum_steps


def process_batch_mlm(batch, config: TrainingConfig):
    """Returns (masked_input, att_mask, targets, mask_positions, species_labels)."""
    if len(batch) == 4:
        _, input_ids, att_mask, label_raw = batch
    else:
        _, input_ids, att_mask = batch
        label_raw = None

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if not isinstance(att_mask, torch.Tensor):
        att_mask = torch.as_tensor(att_mask)
    if label_raw is not None and not isinstance(label_raw, torch.Tensor):
        label_raw = torch.as_tensor(label_raw)

    input_ids = input_ids.long()
    att_mask = att_mask.long()

    if input_ids.size(1) > config.max_seq_length:
        input_ids = input_ids[:, :config.max_seq_length]
        att_mask = att_mask[:, :config.max_seq_length]

    if (input_ids >= config.vocab_size).any():
        input_ids = torch.clamp(input_ids, max=config.vocab_size - 1)

    targets = input_ids.clone()
    att_mask = att_mask.to(dtype=torch.float32)

    valid_token_mask = targets > config.num_special_tokens
    random_mask = torch.rand(targets.shape)
    mask_positions = (random_mask < config.mask_ratio) & valid_token_mask

    masked_input = targets.detach().clone()
    masked_input.masked_fill_(mask_positions, config.mask_token_id)
    att_mask.masked_fill_(mask_positions, 0.0)

    if label_raw is None:
        species_labels = torch.full((masked_input.size(0),), -1, dtype=torch.long)
    else:
        species_labels = label_raw.view(label_raw.size(0)).long()

    return masked_input, att_mask, targets, mask_positions, species_labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model(config: TrainingConfig, device: str, ddp: bool, ddp_local_rank: int):
    torch.set_float32_matmul_precision("high")

    dec_n_layers = config.decoder_n_layers if config.decoder_n_layers is not None else config.n_layers
    dec_n_heads = config.decoder_n_heads if config.decoder_n_heads is not None else config.n_heads

    base_cfg = dict(
        _name_or_path="zhihan1996/DNABERT-2-117M",
        alibi_starting_size=config.alibi_starting_size,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=None,
        gradient_checkpointing=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        # Decoder sees max_seq_length + jumbo_multiplier tokens; set buffer large enough.
        # ALiBi models don't use position embeddings, so this is only a buffer size.
        max_position_embeddings=config.max_seq_length + config.jumbo_multiplier + 8,
        position_embedding_type="absolute",
        torch_dtype="float32",
        transformers_version="4.29.0",
        type_vocab_size=2,
        use_cache=True,
        vocab_size=config.vocab_size,
        num_labels=config.vocab_size,
    )

    if config.architecture == "bert" and config.jumbo:
        if not BARCODEBERT_AVAILABLE:
            raise RuntimeError("barcodebert package not found. Run `pip install -e .` from BarcodeMAE/.")
        from barcodebert.jumbo_transformer_with_taxonomy import create_jumbo_transformer_with_taxonomy
        encoder_config = BertConfig(**{**base_cfg, "num_hidden_layers": 12, "num_attention_heads": 12})
        model = create_jumbo_transformer_with_taxonomy(
            encoder_config,
            jumbo_multiplier=config.jumbo_multiplier,
            share_jumbo_mlp_across_layers=config.share_jumbo_layers,
            enable_taxonomy_classification=(config.species_vocab is not None),
            mlp_expansion_factor=config.jumbo_mlp_expansion,
        )
        model.enable_genus_classification = (config.species_vocab is not None)

    elif config.architecture == "bert" and config.use_cls_token:
        # BERT-base + standard CLS token taxonomy head
        if not BARCODEBERT_AVAILABLE:
            raise RuntimeError("barcodebert package not found. Run `pip install -e .` from BarcodeMAE/.")
        encoder_config = BertConfig(**{**base_cfg, "num_hidden_layers": 12, "num_attention_heads": 12})
        from bert_layers import BertForMaskedLM
        model = BertForMaskedLM(encoder_config)
        num_species = None
        if config.species_vocab:
            with open(config.species_vocab) as f:
                num_species = len(json.load(f))
        model.taxonomy_classifier = CLSTaxonomyClassifier(
            hidden_dim=encoder_config.hidden_size
        ) if num_species is not None else None
        model.enable_genus_classification = num_species is not None

    elif config.architecture == "bert":
        # BERT-base: fixed at 12L/12H/768/3072 — not configurable via CLI
        encoder_config = BertConfig(**{**base_cfg, "num_hidden_layers": 12, "num_attention_heads": 12})
        from bert_layers import BertForMaskedLM
        model = BertForMaskedLM(encoder_config)

    elif config.architecture == "maelm":
        encoder_config = BertConfig(**{**base_cfg,
                                       "num_hidden_layers": config.n_layers,
                                       "num_attention_heads": config.n_heads})
        decoder_config = BertConfig(**{**base_cfg,
                                       "num_hidden_layers": dec_n_layers,
                                       "num_attention_heads": dec_n_heads})

        num_species = None
        if config.species_vocab:
            with open(config.species_vocab) as f:
                num_species = len(json.load(f))

        if config.jumbo:
            if not BARCODEBERT_AVAILABLE:
                raise RuntimeError("barcodebert package not found. Run `pip install -e .` from BarcodeMAE/.")
            model = BarcodeMAELMModel(
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                jumbo=True,
                jumbo_multiplier=config.jumbo_multiplier,
                share_jumbo_layers=config.share_jumbo_layers,
                enable_genus_classification=(num_species is not None),
                mlp_expansion_factor=config.jumbo_mlp_expansion,
            )
        elif config.use_cls_token:
            if not BARCODEBERT_AVAILABLE:
                raise RuntimeError("barcodebert package not found. Run `pip install -e .` from BarcodeMAE/.")
            model = BarcodeMAELMModel(
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                jumbo=False,
                jumbo_multiplier=1,
                share_jumbo_layers=False,
                enable_genus_classification=False,
                use_cls_token=True,
            )
            model.taxonomy_classifier = CLSTaxonomyClassifier(
                hidden_dim=encoder_config.hidden_size
            ) if num_species is not None else None
            model.enable_genus_classification = num_species is not None
        else:
            from maelm_model import MAELMModel
            model = MAELMModel(encoder_config, decoder_config)

    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")

    model.to(device)

    # Keep a reference to the uncompiled model for attribute access
    original_model = model

    # torch.compile is incompatible with Jumbo MAE: the model uses integer
    # attributes (jumbo_multiplier) in tensor ops and has dynamic sequence
    # lengths from masking — both break dynamo's symbolic shape tracing.
    can_compile = config.use_compile and not config.jumbo
    if can_compile:
        model = torch.compile(model, mode=config.compile_mode)
    elif config.use_compile and config.jumbo:
        print("Note: torch.compile disabled for Jumbo MAE (dynamic shapes incompatible)")

    return model, original_model, encoder_config


# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------

def configure_optimizers(model, config: TrainingConfig, device_type: str, master_process: bool):
    if config.architecture == "maelm":
        if master_process:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Optimizer: AdamW on {n_params:,} trainable parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"Using fused AdamW: {use_fused}")
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay,
            fused=use_fused if use_fused else False,
        )

    # BERT: separate decay / no-decay groups
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    if master_process:
        print(f"Decay params: {len(decay_params)} tensors / {sum(p.numel() for p in decay_params):,} params")
        print(f"No-decay params: {len(nodecay_params)} tensors / {sum(p.numel() for p in nodecay_params):,} params")

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"Using fused AdamW: {use_fused}")
    return torch.optim.AdamW(optim_groups, lr=config.max_lr, betas=(0.9, 0.95), eps=1e-8,
                             fused=use_fused if use_fused else False)


def get_lr(step: int, config: TrainingConfig):
    min_lr = config.max_lr * config.min_lr_ratio
    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    if step > config.max_steps:
        return min_lr
    return config.max_lr + ((min_lr - config.max_lr) / (config.max_steps - config.warmup_steps)) * (step - config.warmup_steps)


def build_scheduler(optimizer, config: TrainingConfig, start_step: int):
    use_cosine = config.scheduler == "cosine" or (config.scheduler == "auto" and config.architecture == "maelm")
    if not use_cosine:
        return None
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
        last_epoch=start_step - 1,
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def setup_checkpoint_dirs(config: TrainingConfig):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    return os.path.join(config.checkpoint_dir, "latest_checkpoint.pt"), \
           os.path.join(config.log_dir, "log.txt")


def save_checkpoint(step, raw_model, optimizer, loss, dnabert_config, checkpoint_path, config, master_process):
    if not master_process:
        return
    torch.save({
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": dnabert_config.to_dict(),
        "architecture": config.architecture,
        "maelm_model_type": config.maelm_model_type,
    }, checkpoint_path)
    print(f"Checkpoint saved at step {step}")

    if config.architecture == "maelm":
        step_dir = os.path.join(config.checkpoint_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        torch.save(raw_model.state_dict(), os.path.join(step_dir, "pytorch_model.bin"))
        dnabert_config.save_pretrained(step_dir)
        torch.save({"optimizer": optimizer.state_dict(), "step": step, "architecture": "maelm"},
                   os.path.join(step_dir, "training_state.pt"))
        latest = os.path.join(config.checkpoint_dir, "latest")
        if os.path.islink(latest) or os.path.exists(latest):
            try:
                os.remove(latest)
            except OSError:
                pass
        os.symlink(os.path.abspath(step_dir), latest)


def load_checkpoint(raw_model, optimizer, checkpoint_path, checkpoint_dir, device, master_process):
    if os.path.exists(checkpoint_path):
        if master_process:
            print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        if master_process:
            print(f"Resumed from step {start_step}, previous loss: {ckpt['loss']:.6f}")
        return start_step

    legacy_latest = os.path.join(checkpoint_dir, "latest")
    legacy_ts = os.path.join(legacy_latest, "training_state.pt")
    legacy_ms = os.path.join(legacy_latest, "pytorch_model.bin")
    if os.path.islink(legacy_latest) and os.path.exists(legacy_ts) and os.path.exists(legacy_ms):
        if master_process:
            print(f"Loading legacy checkpoint from {legacy_latest}")
        ts = torch.load(legacy_ts, map_location=device)
        ms = torch.load(legacy_ms, map_location=device)
        raw_model.load_state_dict(ms)
        optimizer.load_state_dict(ts["optimizer"])
        start_step = int(ts["step"]) + 1
        if master_process:
            print(f"Resumed from legacy checkpoint at step {start_step}")
        return start_step

    if master_process:
        print("No checkpoint found, starting from step 0")
    return 0


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def maybe_init_wandb(config: TrainingConfig, master_process: bool):
    if not (config.use_wandb and master_process) or wandb is None:
        return None

    run_config = {k: getattr(config, k) for k in (
        "architecture", "total_batch_size", "batch_size", "max_seq_length",
        "max_lr", "warmup_steps", "max_steps", "weight_decay", "mask_ratio",
        "n_layers", "n_heads", "jumbo", "jumbo_multiplier", "share_jumbo_layers",
        "k_classes", "m_per_class", "cls_loss_weight",
    )}

    init_kwargs = dict(project=config.wandb_project, entity=config.wandb_entity,
                       name=config.wandb_run_name, mode=config.wandb_mode, config=run_config)

    if config.wandb_existing_run_path:
        api = wandb.Api()
        existing = api.run(config.wandb_existing_run_path)
        for k, v in run_config.items():
            existing.config[k] = v
        existing.update()
        parts = config.wandb_existing_run_path.split("/")
        init_kwargs.update(id=existing.id, resume="allow")
        if len(parts) >= 3:
            init_kwargs.update(entity=parts[0], project=parts[1])

    return wandb.init(**init_kwargs)


def maybe_log_wandb(metrics: dict, step: int, run):
    if run is not None:
        run.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend="nccl", timeout=datetime.timedelta(hours=3))
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master = rank == 0
        device_type = "cuda"
    else:
        rank = local_rank = 0
        world_size = 1
        master = True
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        print(f"Using device: {device}")
        torch.manual_seed(123)
        if device_type == "cuda":
            torch.cuda.manual_seed(123)

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True

    return ddp, rank, local_rank, world_size, device, device_type, master


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: TrainingConfig):
    ddp, rank, local_rank, world_size, device, device_type, master = setup_distributed()

    # Resolve shard glob once
    resolved_shards = sorted(glob.glob(config.train_shards_pattern)) or config.train_shards_pattern

    # Species vocab
    num_species = None
    if config.jumbo and config.species_vocab:
        with open(config.species_vocab) as f:
            num_species = len(json.load(f))
        if master:
            print(f"Loaded species vocab: {num_species:,} species")

    # Data loader
    if config.k_classes > 0:
        k, m = config.k_classes, config.m_per_class
        if k * m > config.batch_size:
            raise ValueError(f"k*m={k*m} exceeds batch_size={config.batch_size}")
        assert config.total_batch_size % (config.batch_size * world_size) == 0
        grad_accum_steps = config.total_batch_size // (config.batch_size * world_size)
        if master:
            print(f"Balanced sampling: k={k} × m={m} = {k*m} labeled + {config.batch_size-k*m} fill "
                  f"→ {k*(m*(m-1)//2)} positive pairs per batch")
            print(f"total desired batch size: {config.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        raw_ds = make_dataset(
            resolved_shards,
            config.shuffle_buffer,
            resampled=True,
            world_size=world_size,
            require_label=True,
        )
        train_loader = balanced_batch_pipeline(iter(raw_ds), k=k, m=m, batch_size=config.batch_size)
    else:
        train_loader, grad_accum_steps = setup_data_loader(config, world_size, device, master, resolved_shards)

    # Model
    model, original_model, dnabert_config = create_model(config, device, ddp, local_rank)
    if ddp and world_size > 1:
        find_unused = config.architecture == "maelm"
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused)

    if master:
        enc_l = config.n_layers
        enc_h = config.n_heads
        dec_l = config.decoder_n_layers or config.n_layers
        dec_h = config.decoder_n_heads or config.n_heads
        jumbo_str = f" jumbo=J{config.jumbo_multiplier}×{config.jumbo_mlp_expansion} shared={config.share_jumbo_layers}" if config.jumbo else ""
        print(f"Model: {config.architecture} enc={enc_l}L/{enc_h}H dec={dec_l}L/{dec_h}H{jumbo_str}")
        print(f"Compiled: {config.use_compile} (mode={config.compile_mode})")

    optimizer = configure_optimizers(original_model, config, device_type, master)

    checkpoint_path, log_file = setup_checkpoint_dirs(config)
    start_step = load_checkpoint(original_model, optimizer, checkpoint_path, config.checkpoint_dir, device, master)
    scheduler = build_scheduler(optimizer, config, start_step)

    criterion = nn.CrossEntropyLoss() if config.architecture == "bert" else None
    wandb_run = maybe_init_wandb(config, master)

    if master:
        print(f"Starting training from step {start_step} → {config.max_steps} "
              f"| accum={grad_accum_steps} | jumbo={config.jumbo}")

    for step in range(start_step, config.max_steps):
        t0 = time.time()
        last_step = (step == config.max_steps - 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_accum = torch.tensor(0.0, device=device)
        tax_loss_accum = 0.0
        num_pos_pairs = 0
        num_neg_pairs = 0

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(train_loader)
            except (StopIteration, Exception) as e:
                if master:
                    print(f"DataLoader issue ({type(e).__name__}), recreating...")
                raw_ds = make_dataset(
                    resolved_shards,
                    config.shuffle_buffer,
                    resampled=True,
                    world_size=world_size,
                    require_label=(config.k_classes > 0),
                )
                if config.k_classes > 0:
                    train_loader = balanced_batch_pipeline(
                        iter(raw_ds), k=config.k_classes, m=config.m_per_class,
                        batch_size=config.batch_size
                    )
                else:
                    train_loader, _ = setup_data_loader(config, world_size, device, master, resolved_shards)
                batch = next(train_loader)

            masked_input, att_mask, targets, mask_positions, species_labels = process_batch_mlm(batch, config)
            masked_input = masked_input.to(device)
            att_mask = att_mask.to(device)
            targets = targets.to(device)
            mask_positions = mask_positions.to(device)
            species_labels = species_labels.to(device)
            has_species_labels = bool((species_labels >= 0).any().item())

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                if config.architecture == "bert" and config.jumbo:
                    # BERT encoder + Jumbo CLS + taxonomy head
                    outputs = model(input_ids=masked_input, attention_mask=att_mask)
                    masked_idx = masked_input == config.mask_token_id
                    loss = criterion(
                        outputs.logits.view(-1, config.vocab_size)[masked_idx.view(-1)],
                        targets.view(-1)[masked_idx.view(-1)],
                    ) if masked_idx.any() else outputs.logits.new_tensor(0.0)
                    _raw = model.module if hasattr(model, "module") else original_model
                    if _raw.enable_genus_classification and has_species_labels and outputs.jumbo_tokens is not None:
                        tax_loss, _, _, n_same, n_diff = compute_taxonomy_classification_loss(
                            outputs.jumbo_tokens, species_labels, _raw.taxonomy_classifier,
                            same_ratio=0.5, max_pairs=32, debug_print=False,
                        )
                        num_pos_pairs += n_same
                        num_neg_pairs += n_diff
                        if tax_loss is not None:
                            tax_loss_accum += tax_loss.detach() / grad_accum_steps
                            loss = loss + config.cls_loss_weight * tax_loss

                elif config.architecture == "bert" and config.use_cls_token:
                    # bert_layers.py hardcodes hidden_states=None in output, so we
                    # call the BERT encoder and MLM head separately to get both.
                    _raw = model.module if hasattr(model, "module") else original_model
                    hs = _raw.bert(masked_input, attention_mask=att_mask)[0]  # (B, N, D)
                    logits = _raw.cls(hs)                                      # (B, N, vocab)
                    masked_idx = masked_input == config.mask_token_id
                    loss = criterion(
                        logits.view(-1, config.vocab_size)[masked_idx.view(-1)],
                        targets.view(-1)[masked_idx.view(-1)],
                    ) if masked_idx.any() else logits.new_tensor(0.0)
                    if _raw.enable_genus_classification and has_species_labels and _raw.taxonomy_classifier is not None:
                        tax_loss, _, _, n_same, n_diff = compute_cls_taxonomy_classification_loss(
                            hs, species_labels, _raw.taxonomy_classifier,
                            same_ratio=0.5, max_pairs=32, debug_print=False,
                        )
                        num_pos_pairs += n_same
                        num_neg_pairs += n_diff
                        if tax_loss is not None:
                            tax_loss_accum += tax_loss.detach() / grad_accum_steps
                            loss = loss + config.cls_loss_weight * tax_loss

                elif config.architecture == "bert":
                    output = model(masked_input, attention_mask=att_mask)
                    masked_idx = masked_input == config.mask_token_id
                    loss = criterion(
                        output.logits.view(-1, config.vocab_size)[masked_idx.view(-1)],
                        targets.view(-1)[masked_idx.view(-1)],
                    ) if masked_idx.any() else output.logits.new_tensor(0.0)

                elif config.jumbo:
                    outputs = model(input_ids=masked_input, attention_mask=att_mask,
                                    mask_positions=mask_positions)
                    loss = F.cross_entropy(
                        outputs.logits.view(-1, dnabert_config.vocab_size)[mask_positions.view(-1)],
                        targets.view(-1)[mask_positions.view(-1)],
                    )
                    _raw = model.module if hasattr(model, "module") else original_model
                    if _raw.enable_genus_classification and has_species_labels and outputs.jumbo_tokens is not None:
                        tax_loss, _, _, n_same, n_diff = compute_taxonomy_classification_loss(
                            outputs.jumbo_tokens, species_labels, _raw.taxonomy_classifier,
                            same_ratio=0.5, max_pairs=32, debug_print=False,
                        )
                        num_pos_pairs += n_same
                        num_neg_pairs += n_diff
                        if tax_loss is not None:
                            tax_loss_accum += tax_loss.detach() / grad_accum_steps
                            loss = loss + config.cls_loss_weight * tax_loss

                elif config.use_cls_token:  # maelm + CLS taxonomy
                    outputs = model(input_ids=masked_input, attention_mask=att_mask,
                                    mask_positions=mask_positions)
                    loss = F.cross_entropy(
                        outputs.logits.view(-1, dnabert_config.vocab_size)[mask_positions.view(-1)],
                        targets.view(-1)[mask_positions.view(-1)],
                    )
                    _raw = model.module if hasattr(model, "module") else original_model
                    if _raw.enable_genus_classification and has_species_labels and _raw.taxonomy_classifier is not None:
                        cls_hidden = getattr(outputs, "cls_token", None)
                        if cls_hidden is not None:
                            tax_loss, _, _, n_same, n_diff = compute_cls_taxonomy_classification_loss(
                                cls_hidden, species_labels, _raw.taxonomy_classifier,
                                same_ratio=0.5, max_pairs=32, debug_print=False,
                            )
                            num_pos_pairs += n_same
                            num_neg_pairs += n_diff
                            if tax_loss is not None:
                                tax_loss_accum += tax_loss.detach() / grad_accum_steps
                                loss = loss + config.cls_loss_weight * tax_loss

                else:  # basic maelm
                    output = model(input_ids=masked_input, attention_mask=att_mask,
                                   labels=targets, model_type=config.maelm_model_type)
                    loss = output.loss if hasattr(output, "loss") else (
                        output[0] if isinstance(output, (tuple, list)) else output
                    )

            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp and world_size > 1:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scheduler is None:
            lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

        if device_type == "cuda":
            torch.cuda.synchronize()

        dt = time.time() - t0
        seq_per_sec = (masked_input.shape[0] * grad_accum_steps * world_size) / max(dt, 1e-6)

        if step % config.log_interval == 0 and master:
            loss_val = loss_accum.item()
            norm_val = norm.item() if isinstance(norm, torch.Tensor) else float(norm)

            avg_pos = num_pos_pairs // grad_accum_steps
            avg_neg = num_neg_pairs // grad_accum_steps
            tax_str = (f" | TaxLoss: {tax_loss_accum.item():.4f} | Pairs: {avg_pos}pos/{avg_neg}neg"
                       if isinstance(tax_loss_accum, torch.Tensor) else "")

            print(f"step {step:5d} | loss: {loss_val:.4f}{tax_str} | "
                  f"lr: {lr:.4e} | norm: {norm_val:.4f} | seq/s: {seq_per_sec:.0f}")

            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_val:.6f}{tax_str} lr {lr:.4e} norm {norm_val:.4f}\n")

            wandb_metrics = {"train/loss": loss_val, "train/lr": lr,
                             "train/grad_norm": norm_val, "train/seq_per_sec": seq_per_sec}
            if isinstance(tax_loss_accum, torch.Tensor):
                wandb_metrics["train/tax_loss"] = tax_loss_accum.item()
                wandb_metrics["train/pos_pairs"] = num_pos_pairs
                wandb_metrics["train/neg_pairs"] = num_neg_pairs
            maybe_log_wandb(wandb_metrics, step, wandb_run)

        if ((step + 1) % config.checkpoint_interval == 0 or last_step) and master:
            save_checkpoint(step, original_model, optimizer, loss_accum.item(),
                            dnabert_config, checkpoint_path, config, master)

            if config.finetune_data_path:
                history_path = os.path.join(config.log_dir, "finetune_history.csv")
                finetune_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.py")
                model_type = "maelm" if config.architecture == "maelm" else "bert"
                metrics = monitor.run_finetune(finetune_script, checkpoint_path,
                                               config.finetune_data_path, config.log_dir, step,
                                               model_type=model_type)
                if metrics:
                    monitor.update_finetune_history(history_path, step, metrics)
                    monitor.plot_curves(log_file, history_path, config.log_dir)
                    maybe_log_wandb({f"finetune/{k}": v for k, v in metrics.items()}, step, wandb_run)

        if ddp and world_size > 1 and ((step + 1) % config.checkpoint_interval == 0 or last_step):
            dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()
    if ddp:
        destroy_process_group()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified trainer for DNABERT-2 / BarcodeMAE",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    parser.add_argument("--architecture", type=str, choices=["bert", "maelm"], default="bert")

    # Batch / sequence
    parser.add_argument("--total-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--alibi-starting-size", type=int, default=512)

    # Optimiser / schedule
    parser.add_argument("--max-lr", "--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=TRUE_EPOCH_STEPS)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--scheduler", type=str, default="auto", choices=["auto", "linear", "cosine"])

    # Encoder / decoder architecture
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--decoder-n-layers", type=int, default=None)
    parser.add_argument("--decoder-n-heads", type=int, default=None)
    parser.add_argument("--maelm-model-type", type=str, default="maelm_v2",
                        choices=["maelm_v1", "maelm_v2", "baseline"])

    # Jumbo CLS + taxonomy
    parser.add_argument("--jumbo", action="store_true")
    parser.add_argument("--jumbo-multiplier", type=int, default=6)
    parser.add_argument("--jumbo-mlp-expansion", type=int, default=2)
    parser.add_argument("--share-jumbo-layers", action="store_true", default=True)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--species-vocab", type=str, default=None)

    # Balanced sampling
    parser.add_argument("--use-cls-token", action="store_true", default=False,
                        help="Use standard CLS token taxonomy head instead of Jumbo tokens")

    parser.add_argument("--k-classes", type=int, default=0,
                        help="Species per balanced block (0=disabled). k=32 + --m-per-class=2 → 32 positive pairs")
    parser.add_argument("--m-per-class", type=int, default=2)

    # Paths / logging
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="log")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--train-shards-pattern", type=str, default="shards/train-*.tar")
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--finetune-data-path", type=str, default=None)

    # Hardware
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"])

    # W&B
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="dnabert2-training")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-existing-run-path", type=str, default=None)

    args, unknown = parser.parse_known_args()
    if unknown and int(os.environ.get("RANK", 0)) == 0:
        print(f"Ignoring unknown arguments: {unknown}")
    return args


def main():
    args = parse_args()
    config = TrainingConfig(
        architecture=args.architecture,
        total_batch_size=args.total_batch_size,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        alibi_starting_size=args.alibi_starting_size,
        max_lr=args.max_lr,
        min_lr_ratio=args.min_lr_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        scheduler=args.scheduler,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        decoder_n_layers=args.decoder_n_layers,
        decoder_n_heads=args.decoder_n_heads,
        maelm_model_type=args.maelm_model_type,
        jumbo=args.jumbo,
        jumbo_multiplier=args.jumbo_multiplier,
        jumbo_mlp_expansion=args.jumbo_mlp_expansion,
        share_jumbo_layers=args.share_jumbo_layers,
        cls_loss_weight=args.cls_loss_weight,
        species_vocab=args.species_vocab,
        use_cls_token=args.use_cls_token,
        k_classes=args.k_classes,
        m_per_class=args.m_per_class,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        train_shards_pattern=args.train_shards_pattern,
        shuffle_buffer=args.shuffle_buffer,
        num_workers=args.num_workers,
        finetune_data_path=args.finetune_data_path,
        use_compile=not args.no_compile,
        compile_mode=args.compile_mode,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_existing_run_path=args.wandb_existing_run_path,
    )
    train(config)


if __name__ == "__main__":
    main()
