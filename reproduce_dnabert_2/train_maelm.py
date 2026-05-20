import io
import json
import os
import time
import math
import argparse
import random
import glob
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import webdataset as wds
import numpy as np
import warnings
import monitor
from transformers import AutoTokenizer, BertConfig, get_cosine_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

# Local MAELMModel (no jumbo) kept for backward compatibility
try:
    from maelm_model import MAELMModel
except ImportError:
    print("Warning: local maelm_model.py not found.")
    MAELMModel = None

# Barcodebert MAELMModel with full jumbo + taxonomy classification support
try:
    from barcodebert.maelm_model import MAELMModel as BarcodeMAELMModel
    from barcodebert.jumbo_taxonomy_classifier import compute_taxonomy_classification_loss
    BARCODEBERT_AVAILABLE = True
except ImportError:
    print("Warning: barcodebert package not found. Jumbo mode unavailable.")
    BarcodeMAELMModel = None
    compute_taxonomy_classification_loss = None
    BARCODEBERT_AVAILABLE = False


def get_num_cpu_available():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        warnings.warn("Unable to determine number of available CPUs. Falling back to os.cpu_count().")
        return os.cpu_count()


def npy_decoder(data: bytes) -> torch.Tensor:
    arr = np.load(io.BytesIO(data), allow_pickle=False)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.int32)
    return torch.from_numpy(arr)


def make_dataset(pattern: str, shuffle_buf: int = 10_000, resampled: bool = False, world_size: int = 1, rank: int = 0):
    if world_size > 1:
        ds = wds.WebDataset(
            pattern,
            resampled=resampled,
            nodesplitter=wds.shardlists.split_by_node
        )
    else:
        ds = wds.WebDataset(pattern, resampled=resampled, shardshuffle=True)

    if resampled:
        ds = ds.with_epoch(1000000)
        ds = ds.with_length(1000000)

    ds = ds.shuffle(shuffle_buf)
    ds = ds.to_tuple("__key__", "tokens", "attention_mask", "label").map_tuple(
        lambda x: x, npy_decoder, npy_decoder, npy_decoder
    )
    return ds


def setup_data_loaders(batch_size, total_train_batch_size, train_shards_pattern, world_size, rank, device, workers=4):
    assert total_train_batch_size % (batch_size * world_size) == 0
    grad_accum_steps = total_train_batch_size // (batch_size * world_size)

    if rank == 0:
        print(f"total desired batch size: {total_train_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_dataset_patterns = sorted(glob.glob(train_shards_pattern))
    if not train_dataset_patterns:
        train_dataset_patterns = train_shards_pattern

    train_dataset = make_dataset(train_dataset_patterns, 10000, resampled=True,
                                 world_size=world_size, rank=rank)

    cpu_workers = workers

    if world_size > 1:
        dataloader_train = iter(wds.WebLoader(
            train_dataset.batched(batch_size, partial=False),
            num_workers=cpu_workers,
            batch_size=None
        ))
    else:
        dl_train_kwargs = {
            "batch_size": batch_size,
            "drop_last": True,
            "sampler": None,
            "shuffle": False,
            "num_workers": cpu_workers,
            "pin_memory": device.type != "cpu"
        }
        dataloader_train = iter(torch.utils.data.DataLoader(train_dataset, **dl_train_kwargs))

    return dataloader_train, grad_accum_steps


def balanced_batch_pipeline(src, k, m, batch_size, buffer_size=None):
    """
    Streaming class-balanced batch builder for WebDataset.

    Buffers samples by species label and yields collated batches containing:
      - k*m labeled slots: k species × m samples each
          → k * C(m,2) guaranteed same-species (positive) pairs
          → e.g. k=32, m=2: exactly 32 positive pairs
      - (batch_size - k*m) fill slots: random samples (labeled or unlabeled)
          → serve as negatives in the contrastive loss

    Compatible with any WebDataset stream; no index-based access required.
    """
    n_labeled = k * m
    n_fill = batch_size - n_labeled
    if buffer_size is None:
        buffer_size = batch_size * 40

    species_buf = defaultdict(list)       # label -> [(key, tokens, mask, label), ...]
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
        labeled = []
        labeled_keys = set()
        for sp in chosen:
            picked = random.sample(species_buf[sp], m)
            labeled.extend(picked)
            labeled_keys.update(x[0] for x in picked)
            for p in picked:
                species_buf[sp].remove(p)

        fill_candidates = [x for x in general_buf if x[0] not in labeled_keys]
        if len(fill_candidates) < n_fill:
            # Not enough fill yet; restore species buffers and keep streaming
            for item_ in labeled:
                species_buf[item_[3]].append(item_)
            continue

        fill = random.sample(fill_candidates, n_fill)
        batch = labeled + fill
        random.shuffle(batch)

        keys_b = [x[0] for x in batch]
        tokens_b = torch.stack([x[1] for x in batch])
        masks_b = torch.stack([x[2] for x in batch])
        labels_b = torch.tensor([x[3] for x in batch], dtype=torch.long)

        yield keys_b, tokens_b, masks_b, labels_b


def process_batch_mlm(batch, tokenizer, mask_ratio=0.15):
    """
    Process batch for masked language modeling.
    Returns: masked_input, att_mask, targets, mask_positions, species_labels
    """
    _, input_ids, att_mask, label_raw = batch

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.as_tensor(input_ids)
    if not isinstance(att_mask, torch.Tensor):
        att_mask = torch.as_tensor(att_mask)
    if not isinstance(label_raw, torch.Tensor):
        label_raw = torch.as_tensor(label_raw)

    input_ids = input_ids.long()
    att_mask = att_mask.long()

    if input_ids.size(1) > 512:
        input_ids = input_ids[:, :512]
        att_mask = att_mask[:, :512]

    if (input_ids >= 4096).any():
        input_ids = torch.clamp(input_ids, max=4095)

    targets = input_ids.clone()
    att_mask = att_mask.to(dtype=torch.float32)

    num_special_tokens = 5
    mask_token_id = int(tokenizer.mask_token_id) if tokenizer.mask_token_id is not None else 4

    valid_token_mask = targets > num_special_tokens
    random_mask = torch.rand(targets.shape)
    mask_positions = random_mask < mask_ratio
    mask_positions &= valid_token_mask

    masked_input = targets.detach().clone()
    masked_input.masked_fill_(mask_positions, mask_token_id)
    att_mask.masked_fill_(mask_positions, 0.0)

    # species_labels: (B,) int64, -1 for unlabeled sequences
    species_labels = label_raw.view(label_raw.size(0)).long()

    return masked_input, att_mask, targets, mask_positions, species_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAELM Model (with optional Jumbo CLS tokens)")
    # DDP args
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))

    # Model/Training args
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--total-batch-size", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--warmup-steps", type=int, default=30000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Print loss every N steps (use 1 for smoke tests)")

    # Encoder / decoder architecture
    parser.add_argument("--n-layers", type=int, default=12, help="Encoder hidden layers")
    parser.add_argument("--n-heads", type=int, default=12, help="Encoder attention heads")
    parser.add_argument("--decoder-n-layers", type=int, default=None,
                        help="Decoder hidden layers (defaults to --n-layers)")
    parser.add_argument("--decoder-n-heads", type=int, default=None,
                        help="Decoder attention heads (defaults to --n-heads)")

    # Jumbo CLS token args (uses barcodebert MAELMModel)
    parser.add_argument("--jumbo", action="store_true",
                        help="Use Jumbo CLS tokens in the encoder (requires barcodebert package)")
    parser.add_argument("--jumbo-multiplier", type=int, default=6,
                        help="Number of Jumbo CLS tokens (J)")
    parser.add_argument("--jumbo-mlp-expansion", type=int, default=2,
                        help="MLP width multiplier for Jumbo CLS token layers")
    parser.add_argument("--share-jumbo-layers", action="store_true", default=True,
                        help="Share Jumbo MLP weights across encoder layers")
    parser.add_argument("--species-vocab", type=str, default=None,
                        help="Path to species_vocab.json written by write_shards.py")
    parser.add_argument("--cls-loss-weight", type=float, default=1.0,
                        help="Weight for taxonomy classification loss relative to MAE loss")

    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="log")
    parser.add_argument("--train-shards-pattern", type=str, default="shards/train-*.tar")
    parser.add_argument("--mask-ratio", type=float, default=0.15)

    # Fine-tuning / Monitoring
    parser.add_argument("--finetune-data-path", type=str, default=None)

    # Compatibility args
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--alibi-starting-size", type=int, default=512)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (ignored when --no-compile is set)")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable automatic mixed precision (fp16 on CUDA)")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable automatic mixed precision")
    parser.add_argument("--num-workers", type=int, default=4)

    # Balanced sampling (streaming KClassMSample equivalent for WebDataset)
    parser.add_argument("--k-classes", type=int, default=0,
                        help="Species per balanced block per batch (0 = disabled). "
                             "k=32 with --m-per-class=2 gives 32 positive pairs.")
    parser.add_argument("--m-per-class", type=int, default=2,
                        help="Samples per species in the balanced block (default 2).")

    return parser.parse_known_args()[0]


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, step, args, rank):
    if rank != 0:
        return

    step_dir = os.path.join(args.checkpoint_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(step_dir, "pytorch_model.bin"))

    if hasattr(model_to_save, "encoder") and hasattr(model_to_save.encoder, "config"):
        model_to_save.encoder.config.save_pretrained(step_dir)
    elif hasattr(model_to_save, "config"):
        model_to_save.config.save_pretrained(step_dir)

    train_state = {
        'optimizer': optimizer.state_dict(),
        'step': step,
        'args': args
    }
    torch.save(train_state, os.path.join(step_dir, "training_state.pt"))

    latest_link = os.path.join(args.checkpoint_dir, "latest")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.abspath(step_dir), latest_link)


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+ GPUs
    else:
        device = torch.device("cpu")

    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    # Load species vocabulary if provided (required for taxonomy loss)
    num_species = None
    if args.jumbo and args.species_vocab:
        with open(args.species_vocab) as f:
            species_vocab = json.load(f)
        num_species = len(species_vocab)
        if rank == 0:
            print(f"Loaded species vocab: {num_species:,} species from {args.species_vocab}")

    # Resolve shard glob once — setup_data_loaders does this internally but
    # make_dataset (used by the balanced path) gets the raw pattern otherwise.
    resolved_shards = sorted(glob.glob(args.train_shards_pattern)) or args.train_shards_pattern

    # Data Loading
    if args.k_classes > 0:
        k, m = args.k_classes, args.m_per_class
        if k * m > args.batch_size:
            raise ValueError(f"k*m={k}*{m}={k*m} exceeds batch_size={args.batch_size}")
        assert args.total_batch_size % (args.batch_size * world_size) == 0
        acc_steps = args.total_batch_size // (args.batch_size * world_size)
        if rank == 0:
            print(f"Balanced sampling: k={k} species × m={m} samples = {k*m} labeled "
                  f"+ {args.batch_size - k*m} fill per batch → {k * (m*(m-1)//2)} positive pairs")
            print(f"total desired batch size: {args.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {acc_steps}")
        raw_dataset = make_dataset(
            resolved_shards, 10000, resampled=True,
            world_size=world_size, rank=rank
        )
        train_loader = balanced_batch_pipeline(
            iter(raw_dataset), k=k, m=m, batch_size=args.batch_size
        )
    else:
        train_loader, acc_steps = setup_data_loaders(
            args.batch_size,
            args.total_batch_size,
            args.train_shards_pattern,
            world_size,
            rank,
            device,
            workers=args.num_workers
        )

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    dec_n_layers = args.decoder_n_layers if args.decoder_n_layers is not None else args.n_layers
    dec_n_heads = args.decoder_n_heads if args.decoder_n_heads is not None else args.n_heads

    encoder_config = BertConfig(**{
        "vocab_size": 4096,
        "hidden_size": 768,
        "num_hidden_layers": args.n_layers,
        "num_attention_heads": args.n_heads,
        "intermediate_size": 3072,
        "max_position_embeddings": args.max_seq_length + args.jumbo_multiplier + 8,
        "type_vocab_size": 2,
        "_name_or_path": "zhihan1996/DNABERT-2-117M",
        "alibi_starting_size": args.alibi_starting_size,
        "num_labels": 4096,
    })
    decoder_config = BertConfig(**{
        "vocab_size": 4096,
        "hidden_size": 768,
        "num_hidden_layers": dec_n_layers,
        "num_attention_heads": dec_n_heads,
        "intermediate_size": 3072,
        "max_position_embeddings": args.max_seq_length + args.jumbo_multiplier + 8,
        "type_vocab_size": 2,
        "_name_or_path": "zhihan1996/DNABERT-2-117M",
        "alibi_starting_size": args.alibi_starting_size,
        "num_labels": 4096,
    })
    config = encoder_config  # kept for loss computation references below

    # Build model
    if args.jumbo:
        if not BARCODEBERT_AVAILABLE:
            raise RuntimeError(
                "barcodebert package not found. Run `pip install -e .` from the BarcodeMAE directory."
            )
        if rank == 0:
            print(f"Initializing Jumbo MAELMModel "
                  f"(enc={args.n_layers}L/{args.n_heads}H "
                  f"dec={dec_n_layers}L/{dec_n_heads}H "
                  f"J={args.jumbo_multiplier} mlp_exp={args.jumbo_mlp_expansion} "
                  f"shared={args.share_jumbo_layers} "
                  f"taxonomy={'yes' if num_species else 'no'})...")
        raw_model = BarcodeMAELMModel(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            jumbo=True,
            jumbo_multiplier=args.jumbo_multiplier,
            share_jumbo_layers=args.share_jumbo_layers,
            enable_genus_classification=(num_species is not None),
            mlp_expansion_factor=args.jumbo_mlp_expansion,
        )
    else:
        if rank == 0:
            print("Initializing MAELMModel (no jumbo)...")
        raw_model = MAELMModel(config, config)

    raw_model.to(device)

    if not args.no_compile:
        if rank == 0:
            print(f"Compiling model with torch.compile (mode={args.compile_mode})...")
        if args.jumbo:
            print("Note: torch.compile disabled for Jumbo MAE (dynamic shapes incompatible)")
        else:
            raw_model = torch.compile(raw_model, mode=args.compile_mode)

    model = raw_model

    if world_size > 1:
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    model.train()
    step = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    try:
        if rank == 0:
            print(f"MAELM Training | Steps: {args.max_steps} | Accum: {acc_steps} | Jumbo: {args.jumbo}")

        for step in range(0, args.max_steps):
            loss_accum = 0.0
            tax_loss_accum = 0.0
            num_pos_pairs = 0
            num_neg_pairs = 0

            for micro_step in range(acc_steps):
                try:
                    batch = next(train_loader)
                except (StopIteration, Exception) as e:
                    if rank == 0:
                        print(f"DataLoader issue ({type(e).__name__}), recreating...")
                        if isinstance(e, (BrokenPipeError, ConnectionResetError)):
                            time.sleep(10)
                    if args.k_classes > 0:
                        raw_dataset = make_dataset(
                            args.train_shards_pattern, 10000, resampled=True,
                            world_size=world_size, rank=rank
                        )
                        train_loader = balanced_batch_pipeline(
                            iter(raw_dataset), k=args.k_classes, m=args.m_per_class,
                            batch_size=args.batch_size
                        )
                    else:
                        train_loader, _ = setup_data_loaders(
                            args.batch_size, args.total_batch_size, args.train_shards_pattern,
                            world_size, rank, device
                        )
                    batch = next(train_loader)

                masked_input, attention_mask, targets, mask_positions, species_labels = \
                    process_batch_mlm(batch, tokenizer, args.mask_ratio)

                masked_input = masked_input.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)
                species_labels = species_labels.to(device)

                if world_size > 1:
                    model.require_backward_grad_sync = (micro_step == acc_steps - 1)

                amp_ctx = torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda")
                with amp_ctx:
                    if args.jumbo:
                        mask_positions = mask_positions.to(device)
                        outputs = model(
                            input_ids=masked_input,
                            attention_mask=attention_mask,
                            mask_positions=mask_positions,
                        )

                        # MAE reconstruction loss: only on masked positions
                        loss = F.cross_entropy(
                            outputs.logits.view(-1, config.vocab_size)[mask_positions.view(-1)],
                            targets.view(-1)[mask_positions.view(-1)],
                        )

                        # Taxonomy classification loss via Jumbo CLS tokens
                        _raw = model.module if hasattr(model, "module") else model
                        if _raw.enable_genus_classification and outputs.jumbo_tokens is not None:
                            tax_loss, _, _, n_same, n_diff = compute_taxonomy_classification_loss(
                                outputs.jumbo_tokens,
                                species_labels,
                                _raw.taxonomy_classifier,
                                same_ratio=0.5,
                                max_pairs=32,
                                debug_print=False,
                            )
                            num_pos_pairs += n_same
                            num_neg_pairs += n_diff
                            if tax_loss is not None:
                                tax_loss_accum += tax_loss.detach() / acc_steps
                                loss = loss + args.cls_loss_weight * tax_loss
                    else:
                        outputs = model(input_ids=masked_input, attention_mask=attention_mask, labels=targets)
                        loss = outputs.loss if hasattr(outputs, 'loss') else (
                            outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                        )

                loss = loss / acc_steps
                loss_accum += loss.detach()
                scaler.scale(loss).backward()

            if world_size > 1:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0 and rank == 0:
                elapsed = time.time() - start_time
                if elapsed == 0:
                    elapsed = 1e-5
                seq_per_sec = (args.total_batch_size * args.max_seq_length) / elapsed
                lr = scheduler.get_last_lr()[0]
                tax_str = (
                    f" | TaxLoss: {tax_loss_accum.item():.4f} | Pairs: {num_pos_pairs}pos/{num_neg_pairs}neg"
                    if isinstance(tax_loss_accum, torch.Tensor) else ""
                )
                print(f"Step {step} | Loss: {loss_accum.item():.4f}{tax_str} | LR: {lr:.2e} | Tok/s: {seq_per_sec:.0f}")
                with open(os.path.join(args.log_dir, "log.txt"), "a") as f:
                    f.write(f"{step} train {loss_accum.item():.4f}{tax_str} lr {lr:.2e} tok/s {seq_per_sec:.0f}\n")
                start_time = time.time()

            if (step + 1) % args.checkpoint_interval == 0 and rank == 0:
                save_checkpoint(model, optimizer, step, args, rank)

                if args.finetune_data_path and monitor:
                    print(f"Step {step}: Triggering fine-tuning monitoring...")
                    finetune_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.py")
                    step_dir = os.path.join(args.checkpoint_dir, f"step_{step}")
                    history_path = os.path.join(args.log_dir, "finetune_history.csv")

                    try:
                        metrics = monitor.run_finetune(
                            finetune_script,
                            step_dir,
                            args.finetune_data_path,
                            args.log_dir,
                            step,
                            model_type="maelm"
                        )
                        if metrics:
                            print(f"Step {step} finetune metrics: {metrics}")
                            monitor.update_finetune_history(history_path, step, metrics)
                            monitor.plot_curves(os.path.join(args.log_dir, "log.txt"), history_path, args.log_dir)
                    except Exception as e:
                        print(f"Monitoring failed: {e}")

            if step >= args.max_steps:
                break

    except Exception as e:
        print(f"Error during training on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()