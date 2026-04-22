from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
import glob
import time
import io
import numpy as np
import torch
import webdataset as wds
from torch import nn
import os
import inspect
import warnings
import argparse
import sys
import monitor
import datetime
from dataclasses import dataclass
from typing import Optional
import math

def get_num_cpu_available():
    r"""
    Get the number of available CPU cores.

    Uses :func:`os.sched_getaffinity` if available, otherwise falls back to
    :func:`os.cpu_count`.

    Returns
    -------
    ncpus : int
        The number of available CPU cores.
    """
    try:
        # This is the number of CPUs available to this process, which may
        # be smaller than the number of CPUs on the system.
        # This command is only available on Unix-like systems.
        return len(os.sched_getaffinity(0))
    except Exception:
        # Fall-back for Windows or other systems which don't support sched_getaffinity
        warnings.warn(
            "Unable to determine number of available CPUs available to this python"
            " process specifically. Falling back to the total number of CPUs on the"
            " system.",
            RuntimeWarning,
            stacklevel=2,
        )
        return os.cpu_count()


def npy_decoder(data: bytes) -> torch.Tensor:
    return torch.from_numpy(np.load(io.BytesIO(data), allow_pickle=False))


@dataclass
class TrainingConfig:
    # Data and model parameters
    total_batch_size: int = 4096
    batch_size: int = 8
    max_seq_length: int = 512
    alibi_starting_size: int = 512
    
    # Training parameters
    max_lr: float = 5e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 30000
    max_steps: int = 500000
    weight_decay: float = 0.1
    mask_ratio: float = 0.15
    
    # Checkpoint and logging
    checkpoint_interval: int = 2000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "log"
    
    # Data paths
    train_shards_pattern: str = "shards/train-*.tar"
    
    # Model configuration
    use_compile: bool = True
    shuffle_buffer: int = 10000
    
    # Special tokens
    num_special_tokens: int = 5
    mask_token_id: int = 4
    vocab_size: int = 4096

    # Monitoring
    finetune_data_path: Optional[str] = None


def make_dataset(pattern: str, shuffle_buf: int = 10_000, resampled: bool = False, world_size: int = 1, rank: int = 0):
    # Create WebDataset with proper nodesplitter for distributed training
    if world_size > 1:
        # Use explicit nodesplitter for multi-GPU/multi-node training
        ds = wds.WebDataset(
            pattern, 
            resampled=resampled,
            nodesplitter=wds.shardlists.split_by_node
        )
    else:
        # Single GPU training
        ds = wds.WebDataset(pattern, resampled=resampled, shardshuffle=True)
    
    # Add epoch and length for continuous training
    if resampled:
        ds = ds.with_epoch(1000000)  # Large epoch size for continuous training
        ds = ds.with_length(1000000)  # Ensure dataset has sufficient length
    
    # Shuffle and decode
    ds = ds.shuffle(shuffle_buf)  # shuffle *within* the streaming buffer
    ds = ds.to_tuple("__key__", "tokens", "attention_mask").map_tuple(lambda x:x, npy_decoder, npy_decoder)
    return ds


def setup_distributed():
    """Setup distributed training configuration"""
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA is required for distributed data training"
        # Increase timeout to 3 hours to allow for long fine-tuning steps on rank 0
        init_process_group(backend='nccl', timeout=datetime.timedelta(hours=3))
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        device_type = "cuda"
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        
        device = "cpu"
        use_cuda = False
        if torch.cuda.is_available():
            device = "cuda"
            use_cuda = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
        
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        
        torch.manual_seed(123)
        if use_cuda:
            torch.cuda.manual_seed(123)
    
    return ddp, ddp_local_rank, ddp_world_size, device, device_type, master_process

def setup_data_loaders(config: TrainingConfig, ddp_world_size: int, device: str, master_process: bool):
    """Setup training and validation data loaders"""
    assert config.total_batch_size % (config.batch_size * ddp_world_size) == 0
    grad_accum_steps = config.total_batch_size // (config.batch_size * ddp_world_size)
    
    if master_process:
        print(f"total desired batch size: {config.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    
    # Setup datasets
    train_dataset_pattern = sorted(glob.glob(config.train_shards_pattern))
    train_dataset = make_dataset(train_dataset_pattern, config.shuffle_buffer, resampled=True, 
                                world_size=ddp_world_size, rank=int(os.environ.get('RANK', 0)))
    
    # Use WebLoader for WebDataset (better integration than PyTorch DataLoader)
    cpu_workers = get_num_cpu_available()
    cpu_workers = min(cpu_workers, 4)
    
    if ddp_world_size > 1:
        # For distributed training, use WebLoader
        dataloader_train = iter(wds.WebLoader(
            train_dataset.batched(config.batch_size, partial=False),
            num_workers=cpu_workers,
            batch_size=None  # Batching handled by WebDataset
        ))
    else:
        # Single GPU can still use regular DataLoader
        dl_train_kwargs = {
            "batch_size": config.batch_size,
            "drop_last": True,
            "sampler": None,
            "shuffle": False,
            "num_workers": cpu_workers,
            "pin_memory": device != "cpu"
        }
        dataloader_train = iter(torch.utils.data.DataLoader(train_dataset, **dl_train_kwargs))
    
    return dataloader_train, grad_accum_steps



def process_batch_mlm(batch, config: TrainingConfig):
    """Process batch for masked language modeling"""
    _, input_ids, att_mask = batch
    
    targets = input_ids.long()
    att_mask = att_mask.to(dtype=torch.float32)
    
    valid_token_mask = targets > config.num_special_tokens
    random_mask = torch.rand(targets.shape)
    input_maskout = random_mask < config.mask_ratio
    input_maskout &= valid_token_mask  # Cannot mask the special tokens including [UNK]
    
    masked_input = targets.detach().clone()
    masked_input[input_maskout] = config.mask_token_id
    att_mask[input_maskout] = 0
    
    return masked_input, att_mask, targets
    
# print(process_batch_mlm(sample_batch))

def create_model(config: TrainingConfig, device: str, ddp: bool, ddp_local_rank: int):
    """Create and setup the BERT model"""
    from bert_layers import BertForMaskedLM
    from transformers import BertConfig
    
    torch.set_float32_matmul_precision('high')
    
    dnabert_config = BertConfig(**{
        "_name_or_path": "zhihan1996/DNABERT-2-117M",
        "alibi_starting_size": config.alibi_starting_size,
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": None,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": config.max_seq_length,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "position_embedding_type": "absolute",
        "torch_dtype": "float32",
        "transformers_version": "4.29.0",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": config.vocab_size
    })
    
    model = BertForMaskedLM(dnabert_config)
    print(model)
    
    model.to(device)
    if config.use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    
    return model, raw_model, dnabert_config

def get_lr(step: int, config: TrainingConfig):
    """Get learning rate for current step"""
    min_lr = config.max_lr * config.min_lr_ratio
    
    # Linear warmup
    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    # Return min learning rate after max_steps
    if step > config.max_steps:
        return min_lr
    # Linear decay
    return config.max_lr + ((min_lr - config.max_lr) / (config.max_steps - config.warmup_steps)) * (step - config.warmup_steps) 

def configure_optimizers(model, config: TrainingConfig, device_type: str, master_process: bool):
    """Configure AdamW optimizer with weight decay"""
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    
    if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"using fused AdamW: {use_fused}")
    
    optimizer = torch.optim.AdamW(optim_groups, lr=config.max_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

def setup_checkpoint_dirs(config: TrainingConfig):
    """Setup checkpoint and log directories"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest_checkpoint.pt")
    log_file = os.path.join(config.log_dir, "log.txt")
    return checkpoint_path, log_file

def save_checkpoint(step: int, raw_model, optimizer, loss: float, dnabert_config, checkpoint_path: str, master_process: bool):
    """Save checkpoint with model, optimizer state, and step number"""
    if master_process:
        checkpoint = {
            'step': step,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': dnabert_config.to_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at step {step}")

def load_checkpoint(model, optimizer, checkpoint_path: str, device: str, master_process: bool):
    """Load checkpoint if it exists, return starting step"""
    if os.path.exists(checkpoint_path):
        if master_process:
            print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        if master_process:
            print(f"Resumed from step {start_step}, previous loss: {checkpoint['loss']:.6f}")
        return start_step
    else:
        if master_process:
            print("No checkpoint found, starting from step 0")
        return 0

def train(config: TrainingConfig):
    """Main training function"""
    
    
    # Setup distributed training
    ddp, ddp_local_rank, ddp_world_size, device, device_type, master_process = setup_distributed()
    
    # Setup data loaders
    dataloader_train, grad_accum_steps = setup_data_loaders(config, ddp_world_size, device, master_process)
    
    # Create model
    model, raw_model, dnabert_config = create_model(config, device, ddp, ddp_local_rank)
    
    # Setup optimizer
    optimizer = configure_optimizers(raw_model, config, device_type, master_process)
    
    # Setup checkpointing
    checkpoint_path, log_file = setup_checkpoint_dirs(config)
    start_step = load_checkpoint(raw_model, optimizer, checkpoint_path, device, master_process)
    
    # Initialize log file
    log_mode = "a" if start_step > 0 else "w"
    with open(log_file, log_mode, encoding='utf-8') as f:
        if start_step == 0:
            pass  # Clear file for new training
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    if master_process:
        print(f"Starting training from step {start_step} to {config.max_steps}")
    
    for step in range(start_step, config.max_steps):
        t0 = time.time()
        last_step = (step == config.max_steps - 1)
        
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            try:
                batch = next(dataloader_train)
                input_ids, attention_mask, targets = process_batch_mlm(batch, config)
            except StopIteration:
                if master_process:
                    print("Warning: DataLoader exhausted, recreating...")
                # This shouldn't happen with resampled=True, but recreate if needed
                train_dataset_pattern = sorted(glob.glob(config.train_shards_pattern))
                train_dataset = make_dataset(train_dataset_pattern, config.shuffle_buffer, resampled=True,
                                           world_size=ddp_world_size, rank=int(os.environ.get('RANK', 0)))
                
                cpu_workers = min(get_num_cpu_available(), 4)
                if ddp_world_size > 1:
                    # Recreate WebLoader for distributed training
                    dataloader_train = iter(wds.WebLoader(
                        train_dataset.batched(config.batch_size, partial=False),
                        num_workers=cpu_workers,
                        batch_size=None
                    ))
                else:
                    # Recreate regular DataLoader for single GPU
                    dl_kwargs = {"batch_size": config.batch_size, "drop_last": True, "sampler": None, "shuffle": False}
                    if device != "cpu":
                        dl_kwargs.update({"num_workers": cpu_workers, "pin_memory": True})
                    dataloader_train = iter(torch.utils.data.DataLoader(train_dataset, **dl_kwargs))
                batch = next(dataloader_train)
                input_ids, attention_mask, targets = process_batch_mlm(batch, config)
            
            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(dtype=torch.float32, device=device)
            targets = targets.to(device)
            masked_tokens_idx = input_ids == config.mask_token_id
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                output = model(input_ids, attention_mask=attention_mask)
                loss = criterion(
                    output.logits.view(-1, config.vocab_size)[masked_tokens_idx.view(-1)],
                    targets.view(-1)[masked_tokens_idx.view(-1)],
                )
            
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        
        t1 = time.time()
        dt = t1 - t0
        #tokens_processed = input_ids.shape[0] * input_ids.shape[1] * grad_accum_steps * ddp_world_size
        #tokens_per_sec = tokens_processed / dt
        sequences_processed = input_ids.shape[0] * grad_accum_steps * ddp_world_size
        sequences_per_sec = sequences_processed / dt

        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {sequences_per_sec:.2f}")
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
        
        # Save checkpoint
        if (step + 1) % config.checkpoint_interval == 0 or last_step:
            save_checkpoint(step, raw_model, optimizer, loss_accum.item(), dnabert_config, checkpoint_path, master_process)
            
            # Monitor and Fine-tune logic
            if master_process and config.finetune_data_path:
                print(f"Triggering fine-tuning evaluation at step {step}...")
                history_path = os.path.join(config.log_dir, "finetune_history.csv")
                
                # Locate finetune.py
                finetune_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune.py")
                
                # Run finetune
                metrics = monitor.run_finetune(
                    finetune_script, 
                    checkpoint_path, 
                    config.finetune_data_path, 
                    config.log_dir, # output dir
                    step
                )
                
                if metrics:
                    print(f"Fine-tuning step {step} completed with metrics: {metrics}")
                    monitor.update_finetune_history(history_path, step, metrics)
                    monitor.plot_curves(log_file, history_path, config.log_dir)
                else:
                    print(f"Fine-tuning step {step} failed or returned no metrics.")

            if ddp:
                dist.barrier()
    
    if ddp:
        destroy_process_group()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train DNA-BERT 2 model")
    
    # Data and model parameters
    parser.add_argument("--total-batch-size", type=int, default=4096,
                        help="Total batch size across all GPUs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--alibi-starting-size", type=int, default=512,
                        help="ALiBi starting size (should match max-seq-length)")
    
    # Training parameters
    parser.add_argument("--max-lr", type=float, default=5e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="Minimum learning rate as ratio of max-lr")
    parser.add_argument("--warmup-steps", type=int, default=30000,
                        help="Number of warmup steps")
    parser.add_argument("--max-steps", type=int, default=500000,
                        help="Maximum training steps")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--mask-ratio", type=float, default=0.15,
                        help="Masking ratio for MLM")
    
    # Checkpoint and logging
    parser.add_argument("--checkpoint-interval", type=int, default=2000,
                        help="Steps between checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="log",
                        help="Directory to save logs")
    
    # Data paths
    parser.add_argument("--train-shards-pattern", type=str, default="shards/train-*.tar",
                        help="Glob pattern for training shards")
    
    # Other options
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--shuffle-buffer", type=int, default=10000,
                        help="Shuffle buffer size for WebDataset")
    
    # Monitoring
    parser.add_argument("--finetune-data-path", type=str, default=None,
                        help="Path to data for fine-tuning evaluation")

    return parser.parse_args()


def main():
    """Minimal entry point"""
    args = parse_args()
    
    config = TrainingConfig(
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
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        train_shards_pattern=args.train_shards_pattern,
        use_compile=not args.no_compile,
        shuffle_buffer=args.shuffle_buffer,
        finetune_data_path=args.finetune_data_path,
    )
    
    train(config)


if __name__ == "__main__":
    main()

