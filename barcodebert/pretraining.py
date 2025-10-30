#!/usr/bin/env python

import builtins
import math
import os
import shutil
import sys
import time
from datetime import datetime
from socket import gethostname

import torch
import torch.distributed as dist
import torch.optim
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig, BertForTokenClassification

from barcodebert import levenshtein, utils
from barcodebert.datasets import DNADataset
from barcodebert.io import safe_save_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from barcodebert.maelm_model import MAELMModel
from biological_masker import CompatibleBiologicalMasker, get_biological_replacements

BASE_BATCH_SIZE = 64


def run(config):
    r"""
    Run training job (one worker if using distributed training).

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The configuration for this experiment.
    """
    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    if config.seed is not None:
        utils.set_rng_seeds_fixed(config.seed)

    if config.deterministic:
        print("Running in deterministic cuDNN mode. Performance may be slower, but more reproducible.")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # DISTRIBUTION ============================================================
    # Setup for distributed training
    utils.setup_slurm_distributed()
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.distributed = utils.check_is_distributed()
    if config.world_size > 1 and not config.distributed:
        raise EnvironmentError(
            f"WORLD_SIZE is {config.world_size}, but not all other required"
            " environment variables for distributed training are set."
        )
    # Work out the total batch size depending on the number of GPUs we are using
    config.batch_size = config.batch_size_per_gpu * config.world_size

    if config.distributed:
        # For multiprocessing distributed training, gpu rank needs to be
        # set to the global rank among all the processes.
        config.global_rank = int(os.environ["RANK"])
        config.local_rank = int(os.environ["LOCAL_RANK"])
        print(
            f"Rank {config.global_rank} of {config.world_size} on {gethostname()}"
            f" (local GPU {config.local_rank} of {torch.cuda.device_count()})."
            f" Communicating with master at {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        dist.init_process_group(backend="nccl")
    else:
        config.global_rank = 0

    # Suppress printing if this is not the master process for the node
    if config.distributed and config.global_rank != 0:

        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    print()
    print("Configuration:")
    print()
    print(config)
    print()
    print(f"Found {torch.cuda.device_count()} GPUs and {utils.get_num_cpu_available()} CPUs.")

    # Check which device to use
    use_cuda = not config.no_cuda and torch.cuda.is_available()

    if config.distributed and not use_cuda:
        raise EnvironmentError("Distributed training with NCCL requires CUDA.")
    if not use_cuda:
        device = torch.device("cpu")
    elif config.local_rank is not None:
        device = f"cuda:{config.local_rank}"
    else:
        device = "cuda"

    print(f"Using device {device}", flush=True)

    # ==========================================
    # Initialize biological masker
    biological_masker = None
    try:
        biological_masker = CompatibleBiologicalMasker.from_cache_dir(
            cache_dir='./kmer_cache',
            k_mer_size=config.k_mer,
            tokenize_n_nucleotide=config.tokenize_n_nucleotide,
            device=device
        )
        print("✅ Biological masking enabled")
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        print("Using uniform random masking instead")
        biological_masker = None
    # ==========================================

    # LOAD PRE-EMPTION CHECKPOINT =============================================
    checkpoint = None
    config.model_output_dir = None
    if config.checkpoint_path:
        config.model_output_dir = os.path.dirname(config.checkpoint_path)
    if not config.checkpoint_path:
        # Not trying to resume from a checkpoint
        pass
    elif not os.path.isfile(config.checkpoint_path):
        # Looks like we're trying to resume from the checkpoint that this job
        # will itself create. Let's assume this is to let the job resume upon
        # preemption, and it just hasn't been preempted yet.
        print(f"Skipping premature resumption from preemption: no checkpoint file found at '{config.checkpoint_path}'")
        if config.checkpoint_path_resume and os.path.isfile(config.checkpoint_path_resume):
            # Resume from another checkpoint instead
            print(f"Loading resumption checkpoint '{config.checkpoint_path_resume}'", flush=True)
            checkpoint = torch.load(config.checkpoint_path_resume, map_location=device)
    else:
        print(f"Loading resumption checkpoint '{config.checkpoint_path}'", flush=True)
        # Map model parameters to be load to the specified gpu.
        checkpoint = torch.load(config.checkpoint_path, map_location=device)

    if checkpoint is None:
        # Our epochs go from 1 to n_epoch, inclusive
        start_epoch = 1
    else:
        # Continue from where we left off
        start_epoch = checkpoint["epoch"] + 1
        if config.seed is not None:
            # Make sure we don't get the same behaviour as we did on the
            # first epoch repeated on this resumed epoch.
            utils.set_rng_seeds_fixed(config.seed + start_epoch, all_gpu=False)

    # DATASET =================================================================

    if config.dataset_name not in ["CANADA-1.5M", "BIOSCAN-5M"]:
        raise NotImplementedError(f"Dataset {config.dataset_name} not supported.")

    # Handle default stride dynamically set to equal k-mer size
    if config.stride is None:
        config.stride = config.k_mer

    dataset_args = {
        "k_mer": config.k_mer,
        "stride": config.stride,
        "max_len": config.max_len,
        "tokenizer": config.tokenizer,
        "bpe_path": config.bpe_path,
        "tokenize_n_nucleotide": config.tokenize_n_nucleotide,
        "dataset_format": config.dataset_name,
    }

    dataset_train = DNADataset(
        file_path=os.path.join(config.data_dir, "pre_training.csv"),
        randomize_offset=True,
        **dataset_args,
    )
    dataset_val = DNADataset(
        file_path=os.path.join(config.data_dir, "supervised_train.csv"),
        randomize_offset=False,
        **dataset_args,
    )
    eval_set = "Val"

    # Dataloader --------------------------------------------------------------
    dl_train_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": True,
        "sampler": None,
        "shuffle": True,
        "worker_init_fn": utils.worker_seed_fn,
    }
    dl_val_kwargs = {
        "batch_size": config.batch_size_per_gpu,
        "drop_last": False,
        "sampler": None,
        "shuffle": False,
        "worker_init_fn": utils.worker_seed_fn,
    }
    if config.cpu_workers is None:
        config.cpu_workers = utils.get_num_cpu_available()
    if use_cuda:
        cuda_kwargs = {"num_workers": config.cpu_workers, "pin_memory": True}
        dl_train_kwargs.update(cuda_kwargs)
        dl_val_kwargs.update(cuda_kwargs)

    if config.distributed:
        # The DistributedSampler breaks up the dataset across the GPUs
        dl_train_kwargs["sampler"] = DistributedSampler(
            dataset_train,
            shuffle=True,
            seed=config.seed if config.seed is not None else 0,
            drop_last=False,
        )
        dl_train_kwargs["shuffle"] = None
        dl_val_kwargs["sampler"] = DistributedSampler(
            dataset_val,
            shuffle=False,
            drop_last=False,
        )
        dl_val_kwargs["shuffle"] = None

    dataloader_train = torch.utils.data.DataLoader(dataset_train, **dl_train_kwargs)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, **dl_val_kwargs)

    # MODEL ===================================================================
    base_pairs = "ACGT"
    if config.predict_n_nucleotide:
        base_pairs += "N"

    if config.tokenizer == "kmer":
        max_position_embeddings = max(512, math.ceil(1536 / config.stride))
        n_output_tokens = len(base_pairs) ** config.k_mer
        n_special_tokens = len(dataset_train.special_tokens)
        n_all_tokens = n_output_tokens + n_special_tokens
    elif config.tokenizer == "bpe":
        max_position_embeddings = config.max_len
        n_output_tokens = dataset_train.vocab_size
        n_special_tokens = 5
        n_all_tokens = n_output_tokens
    else:
        raise NotImplementedError(f"Tokenizer {config.tokenizer} is not supported.")

    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    bert_config = BertConfig(
        vocab_size=dataset_train.vocab_size,
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_heads,
        num_labels=n_output_tokens,
        output_hidden_states=True,
        max_position_embeddings=max_position_embeddings,
    )

    if config.arch == "maelm":

        decoder_config = BertConfig(
            vocab_size=dataset_train.vocab_size,
            num_hidden_layers=config.decoder_n_layers,
            num_attention_heads=config.decoder_n_heads,
            num_labels=n_output_tokens,
            max_position_embeddings=max_position_embeddings,
            hidden_size=config.decoder_embed_dim,
        )
        model = MAELMModel(bert_config, decoder_config)

    elif config.arch == "transformer":
        model = BertForTokenClassification(bert_config)

    # Configure model for distributed training --------------------------------
    print("\nModel architecture:")
    print(model, flush=True)
    print()

    if not use_cuda:
        print("Using CPU (this will be slow)", flush=True)
    elif config.distributed:
        # Convert batchnorm into SyncBN, using stats computed from all GPUs
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, the DistributedDataParallel
        # constructor should always set a single device scope, otherwise
        # DistributedDataParallel will use all available devices.
        model = model.to(device)
        torch.cuda.set_device(device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=True
        )
    else:
        if config.local_rank is not None:
            torch.cuda.set_device(config.local_rank)
        model = model.to(device)

    # OPTIMIZATION ============================================================
    # Optimizer ---------------------------------------------------------------
    # Set up the optimizer

    # Bigger batch sizes mean better estimates of the gradient, so we can use a
    # bigger learning rate. See https://arxiv.org/abs/1706.02677
    # Hence we scale the learning rate linearly with the total batch size.
    config.lr = config.lr_relative * config.batch_size / BASE_BATCH_SIZE

    # Fetch the constructor of the appropriate optimizer from torch.optim
    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Scheduler ---------------------------------------------------------------
    # Set up the learning rate scheduler
    if config.scheduler.lower() == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            [p["lr"] for p in optimizer.param_groups],
            epochs=config.epochs,
            steps_per_epoch=len(dataloader_train),
        )
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler} not supported.")

    # Loss function -----------------------------------------------------------
    # Set up loss function
    criterion = nn.CrossEntropyLoss()

    distance_table = None
    if config.pretrain_levenshtein and not config.levenshtein_vectorized:
        distance_table = levenshtein.build_lookup_table(config.k_mer).to(device)

    # Mask schedule -----------------------------------------------------------
    if False:
        # Linearly increase to 0.75% masking ratio (disabled)
        # mask_ratios = [(0.75 - 0.25) / config.epochs * x + 0.1 for x in range(config.epochs + 1)]
        raise ValueError("Mask schedule not implemented")
    else:
        # Constant masking rate
        mask_ratios = [0.50 for x in range(config.epochs + 1)]

    # LOGGING =================================================================
    # Setup logging and saving

    # If we're using wandb, initialize the run, or resume it if the job was preempted.
    if config.log_wandb and config.global_rank == 0:
        wandb_run_name = config.run_name
        if wandb_run_name is not None and config.run_id is not None:
            wandb_run_name = f"{wandb_run_name}__{config.run_id}"
        EXCLUDED_WANDB_CONFIG_KEYS = [
            "log_wandb",
            "wandb_entity",
            "wandb_project",
            "global_rank",
            "local_rank",
            "run_name",
            "run_id",
            "model_output_dir",
        ]
        wandb.init(
            name=wandb_run_name,
            id=config.run_id,
            resume="allow",
            group=config.run_id,
            entity=config.wandb_entity,
            project=config.wandb_project,
            config=wandb.helper.parse_config(config, exclude=EXCLUDED_WANDB_CONFIG_KEYS),
            job_type="pretrain",
            tags=["pretrain"],
        )
        # If a run_id was not supplied at the command prompt, wandb will
        # generate a name. Let's use that as the run_name.
        if config.run_name is None:
            config.run_name = wandb.run.name
        if config.run_id is None:
            config.run_id = wandb.run.id

    # If we still don't have a run name, generate one from the current time.
    if config.run_name is None:
        config.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.run_id is None:
        config.run_id = utils.generate_id()

    # If no checkpoint path was supplied, automatically determine the path to
    # which we will save the model checkpoint.
    if not config.checkpoint_path and config.models_dir:
        config.model_output_dir = os.path.join(
            config.models_dir,
            config.dataset_name,
            f"{config.run_name}__{config.run_id}",
        )
        config.checkpoint_path = os.path.join(config.model_output_dir, "checkpoint_pretraining.pt")
        if config.log_wandb and config.global_rank == 0:
            wandb.config.update({"checkpoint_path": config.checkpoint_path}, allow_val_change=True)

    # For consistency with finetune, linearprobe and kNN jobs, record the
    # pretrained_run_name and pretrained_run_id.
    if config.log_wandb and config.global_rank == 0:
        wandb.config.update({"pretrained_run_name": config.run_name, "pretrained_run_id": config.run_id})

    if config.checkpoint_path is None:
        print("Model will not be saved.")
    else:
        os.makedirs(config.model_output_dir, exist_ok=True)
        print(f"Model will be saved to '{config.checkpoint_path}'")

    # RESUME ==================================================================
    # Now that everything is set up, we can load the state of the model,
    # optimizer, and scheduler from a checkpoint, if supplied.

    # Initialize step related variables as if we're starting from scratch.
    # Their values will be overridden by the checkpoint if we're resuming.
    total_step = 0
    n_samples_seen = 0

    best_stats = {"max_accuracy": 0, "best_epoch": 0}

    if checkpoint is not None:
        print(f"Loading state from checkpoint (epoch {checkpoint['epoch']})")
        total_step = checkpoint["total_step"]
        n_samples_seen = checkpoint["n_samples_seen"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint["config"].epochs != config.epochs:
            print(
                f"Warning: the number of epochs in the checkpoint ({checkpoint['config'].epochs}) "
                f"does not match the number of epochs in the current configuration ({config.epochs}).",
                flush=True,
            )
            scheduler.total_steps = len(dataloader_train) * config.epochs
        best_stats["max_accuracy"] = checkpoint.get("max_accuracy", 0)
        best_stats["best_epoch"] = checkpoint.get("best_epoch", 0)

    # TRAIN ===================================================================
    print()
    print("Configuration:")
    print()
    print(config, flush=True)
    print()

    # Ensure modules are on the correct device
    model = model.to(device)

    timing_stats = {}
    t_end_epoch = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        t_start_epoch = time.time()
        if config.seed is not None:
            # If the job is resumed from preemption, our RNG state is currently set the
            # same as it was at the start of the first epoch, not where it was when we
            # stopped training. This is not good as it means jobs which are resumed
            # don't do the same thing as they would be if they'd run uninterrupted
            # (making preempted jobs non-reproducible).
            # To address this, we reset the seed at the start of every epoch. Since jobs
            # can only save at the end of and resume at the start of an epoch, this
            # makes the training process reproducible. But we shouldn't use the same
            # RNG state for each epoch - instead we use the original seed to define the
            # series of seeds that we will use at the start of each epoch.
            epoch_seed = utils.determine_epoch_seed(config.seed, epoch=epoch)
            # We want each GPU to have a different seed to the others to avoid
            # correlated randomness between the workers on the same batch.
            # We offset the seed for this epoch by the GPU rank, so every GPU will get a
            # unique seed for the epoch. This means the job is only precisely
            # reproducible if it is rerun with the same number of GPUs (and the same
            # number of CPU workers for the dataloader).
            utils.set_rng_seeds_fixed(epoch_seed + config.global_rank, all_gpu=False)
            if isinstance(getattr(dataloader_train, "generator", None), torch.Generator):
                # Finesse the dataloader's RNG state, if it is not using the global state.
                dataloader_train.generator.manual_seed(epoch_seed + config.global_rank)
            if isinstance(getattr(dataloader_train.sampler, "generator", None), torch.Generator):
                # Finesse the sampler's RNG state, if it is not using the global RNG state.
                dataloader_train.sampler.generator.manual_seed(config.seed + epoch + 10000 * config.global_rank)

        if hasattr(dataloader_train.sampler, "set_epoch"):
            # Handling for DistributedSampler.
            # Set the epoch for the sampler so that it can shuffle the data
            # differently for each epoch, but synchronized across all GPUs.
            dataloader_train.sampler.set_epoch(epoch)

        # Train ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Note the number of samples seen before this epoch started, so we can
        # calculate the number of samples seen in this epoch.
        n_samples_seen_before = n_samples_seen
        # Run one epoch of training
        train_stats, total_step, n_samples_seen = train_one_epoch(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            dataloader=dataloader_train,
            num_labels=n_output_tokens,
            mask_ratio=mask_ratios[epoch - 1],
            device=device,
            epoch=epoch,
            n_epoch=config.epochs,
            total_step=total_step,
            n_samples_seen=n_samples_seen,
            distance_table=distance_table,
            n_special_tokens=n_special_tokens,
            n_all_tokens=n_all_tokens,
            biological_masker=biological_masker,
        )
        t_end_train = time.time()

        timing_stats["train"] = t_end_train - t_start_epoch
        n_epoch_samples = n_samples_seen - n_samples_seen_before
        train_stats["throughput"] = n_epoch_samples / timing_stats["train"]

        print(f"Pretraining epoch {epoch}/{config.epochs} summary:")
        print(f"  Steps ..............{len(dataloader_train):8d}")
        print(f"  Samples ............{n_epoch_samples:8d}")
        if timing_stats["train"] > 172800:
            print(f"  Duration ...........{timing_stats['train']/86400:11.2f} days")
        elif timing_stats["train"] > 5400:
            print(f"  Duration ...........{timing_stats['train']/3600:11.2f} hours")
        elif timing_stats["train"] > 120:
            print(f"  Duration ...........{timing_stats['train']/60:11.2f} minutes")
        else:
            print(f"  Duration ...........{timing_stats['train']:11.2f} seconds")
        print(f"  Throughput .........{train_stats['throughput']:11.2f} samples/sec")
        print(f"  Loss ...............{train_stats['loss']:14.5f}")
        print(f"  Accuracy ...........{train_stats['accuracy']:11.2f} %")
        print(flush=True)

        # Validate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate on validation set
        t_start_val = time.time()

        eval_stats = evaluate(
            config=config,
            model=model,
            criterion=criterion,
            dataloader=dataloader_val,
            num_labels=n_output_tokens,
            mask_ratio=mask_ratios[epoch - 1],
            device=device,
            distance_table=distance_table,
            n_special_tokens=n_special_tokens,
            n_all_tokens=n_all_tokens,
            biological_masker=biological_masker
        )
        t_end_val = time.time()
        timing_stats["val"] = t_end_val - t_start_val
        eval_stats["throughput"] = len(dataloader_val.dataset) / timing_stats["val"]

        # Check if this is the new best model
        if eval_stats["accuracy"] >= best_stats["max_accuracy"]:
            best_stats["max_accuracy"] = eval_stats["accuracy"]
            best_stats["best_epoch"] = epoch

        print(f"Evaluating epoch {epoch}/{config.epochs} summary:")
        if timing_stats["val"] > 172800:
            print(f"  Duration ...........{timing_stats['val']/86400:11.2f} days")
        elif timing_stats["val"] > 5400:
            print(f"  Duration ...........{timing_stats['val']/3600:11.2f} hours")
        elif timing_stats["val"] > 120:
            print(f"  Duration ...........{timing_stats['val']/60:11.2f} minutes")
        else:
            print(f"  Duration ...........{timing_stats['val']:11.2f} seconds")
        print(f"  Throughput .........{eval_stats['throughput']:11.2f} samples/sec")
        print(f"  Loss ...............{eval_stats['loss']:14.5f}")
        print(f"  Accuracy ...........{eval_stats['accuracy']:11.2f} %")

        # Save model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        t_start_save = time.time()

        if config.model_output_dir and (not config.distributed or config.global_rank == 0):
            if config.arch == "maelm":
                safe_save_model(
                    {
                        "model": model.module.encoder,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                    },
                    config.checkpoint_path,
                    config=config,
                    epoch=epoch,
                    total_step=total_step,
                    n_samples_seen=n_samples_seen,
                    bert_config=bert_config.to_dict(),
                    **best_stats,
                )

            elif config.arch == "transformer":
                safe_save_model(
                    {
                        "model": model,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                    },
                    config.checkpoint_path,
                    config=config,
                    epoch=epoch,
                    total_step=total_step,
                    n_samples_seen=n_samples_seen,
                    bert_config=bert_config.to_dict(),
                    **best_stats,
                )

            if config.save_best_model and best_stats["best_epoch"] == epoch:
                ckpt_path_best = os.path.join(config.model_output_dir, "best_pretraining.pt")
                print(f"Copying model to {ckpt_path_best}")
                shutil.copyfile(config.checkpoint_path, ckpt_path_best)

        t_end_save = time.time()
        timing_stats["saving"] = t_end_save - t_start_save

        # Log to wandb ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Overall time won't include uploading to wandb, but there's nothing
        # we can do about that.
        timing_stats["overall"] = time.time() - t_end_epoch
        t_end_epoch = time.time()

        # Send training and eval stats for this epoch to wandb
        if config.log_wandb and config.global_rank == 0:
            wandb.log(
                {
                    "Pretraining/stepwise/epoch": epoch,
                    "Pretraining/stepwise/epoch_progress": epoch,
                    "Pretraining/stepwise/n_samples_seen": n_samples_seen,
                    "Pretraining/epochwise/epoch": epoch,
                    **{f"Pretraining/epochwise/Train/{k}": v for k, v in train_stats.items()},
                    **{f"Pretraining/epochwise/{eval_set}/{k}": v for k, v in eval_stats.items()},
                    **{f"Pretraining/epochwise/duration/{k}": v for k, v in timing_stats.items()},
                },
                step=total_step,
            )
            # Record the wandb time as contributing to the next epoch
            timing_stats = {"wandb": time.time() - t_end_epoch}
        else:
            # Reset timing stats
            timing_stats = {}
        # Print with flush=True forces the output buffer to be printed immediately
        print(flush=True)

    if start_epoch > config.epochs:
        print("Pretraining already completed!")
    else:
        print(f"Pretraining complete! (Trained epochs {start_epoch} to {config.epochs})")
    print(
        f"Best {eval_set} accuracy was {best_stats['max_accuracy']:.2f}%,"
        f" seen at the end of epoch {best_stats['best_epoch']}",
        flush=True,
    )


def train_one_epoch(
    config,
    model,
    optimizer,
    scheduler,
    criterion,
    dataloader,
    num_labels,
    mask_ratio=0.5,
    device="cuda",
    epoch=1,
    n_epoch=None,
    total_step=0,
    n_samples_seen=0,
    distance_table=None,
    n_special_tokens=2,
    n_all_tokens=-1,
    biological_masker=None
):
    r"""
    Train the encoder and classifier for one epoch.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The global config object.
    model : torch.nn.Module
        The encoder/decoder network.
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler.
    criterion : torch.nn.Module
        The loss function.
    dataloader : torch.utils.data.DataLoader
        A dataloader for the training set.
    num_labels : int,
        The number of labels (number of different tokens)
    mask_ratio : float, default=0.5
        The ratio of tokens to mask out.
    device : str or torch.device, default="cuda"
        The device to use.
    epoch : int, default=1
        The current epoch number (indexed from 1).
    n_epoch : int, optional
        The total number of epochs scheduled to train for.
    total_step : int, default=0
        The total number of steps taken so far.
    n_samples_seen : int, default=0
        The total number of samples seen so far.
    distance_table : torch.Tensor, optional
        A pre-computed table of Levenshtein distances between all possible k-mers.
    n_special_tokens: int, default=2
        Number of special (non-kmer) tokens used by the tokenizer.

    Returns
    -------
    results: dict
        A dictionary containing the training performance for this epoch.
    total_step : int
        The total number of steps taken after this epoch.
    n_samples_seen : int
        The total number of samples seen after this epoch.
    """
    # Put the model in train mode
    model.train()

    if config.log_wandb:
        # Lazy import of wandb, since logging to wandb is optional
        import wandb

    loss_epoch = 0
    masked_loss_epoch = 0
    non_masked_loss_epoch = 0
    acc_epoch = 0
    acc_kpt_epoch = 0
    acc_all_epoch = 0

    base_pairs = "ACGT"
    if config.predict_n_nucleotide:
        base_pairs += "N"

    n_output_tokens = num_labels  # len(base_pairs) ** config.k_mer

    if config.print_interval is None:
        # Default to printing to console every time we log to wandb
        config.print_interval = config.log_interval

    t_end_batch = time.time()
    t_start_wandb = t_end_wandb = None
    for batch_idx, (sequences, y_true, att_mask) in enumerate(dataloader):
        t_start_batch = time.time()
        batch_size_this_gpu = sequences.shape[0]

        # Move training inputs and targets to the GPU
        sequences = sequences.to(device)
        att_mask = att_mask.to(device)

        # Build the masking on the fly ----------------------------------------
        # t_start_masking = time.time()

        # Create a mask for allowed tokens i.e. that excludes all special tokens [<MASK>, <UNK>]
        special_tokens_mask = sequences > (n_special_tokens - 1)

        if config.tokenize_n_nucleotide:
            # Either exlude the last token [N..N] if config.predict_n_nucleotide == True
            # Or exclude all tokens containing Ns i.e "bad kamers" whose index in the vocab
            # is greater than 4**k
            special_tokens_mask &= sequences < (n_special_tokens + n_output_tokens - 1)

        special_tokens_mask = special_tokens_mask.to(device)
        masked_input = sequences.clone()

        random_mask = torch.rand(sequences.shape, device=device)

        mask_token_ratio = config.mask_token_ratio

        # out of the mask tokens (50%), mask_token_ratio are replaced with random tokens
        random_token_ratio = config.random_token_ratio + config.mask_token_ratio

        masked_unseen_tokens = (random_mask < mask_token_ratio * mask_ratio) & special_tokens_mask
        masked_random_tokens = (
            (random_mask >= mask_token_ratio * mask_ratio)
            & (random_mask < random_token_ratio * mask_ratio)
            & special_tokens_mask
        )
        masked_original_tokens = (
            (random_mask >= random_token_ratio * mask_ratio) & (random_mask < mask_ratio) & special_tokens_mask
        )

        input_maskout = masked_unseen_tokens | masked_random_tokens | masked_original_tokens
        # Apply the masks
        masked_input[masked_unseen_tokens] = 0  # Masking the token

        # Replace with random token where mask_random_token is True
        # Generate random tokens
        if biological_masker is not None:
            tokens_to_replace = sequences[masked_random_tokens]
            # Get biological replacements (guaranteed different from originals)
            biological_replacements = biological_masker.get_biological_replacements(tokens_to_replace)
            masked_input[masked_random_tokens] = biological_replacements
            if biological_masker is not None:
                tokens_to_replace = sequences[masked_random_tokens]
                biological_replacements = get_biological_replacements(tokens_to_replace, biological_masker)
                masked_input[masked_random_tokens] = biological_replacements

                # SIMPLE DEBUG - only first batch of first epoch
                if batch_idx == 0 and epoch == 1:
                    print(f"\n🔬 Biological masking check:")
                    for i in range(min(5, len(tokens_to_replace))):
                        orig_id = tokens_to_replace[i].item()
                        repl_id = biological_replacements[i].item()
                        orig_kmer = dataloader.dataset.vocab.lookup_token(orig_id)
                        repl_kmer = dataloader.dataset.vocab.lookup_token(repl_id)
                        same = " SAME!" if orig_id == repl_id else "OK"
                        print(f"  {orig_kmer} → {repl_kmer} {same}")


        else:
            min_token_id = n_special_tokens  # 0 is for masking, 1 is for <UNK>
            max_token_id = n_all_tokens  # number of all tokens (including special)
            random_tokens = torch.randint_like(masked_input, low=min_token_id, high=max_token_id)

            # Ensure random tokens are not the same as the original tokens
            while True:
                same_as_original = (random_tokens == masked_input) & masked_random_tokens
                if not same_as_original.any():
                    break
                random_tokens[same_as_original] = torch.randint(
                    size=(same_as_original.sum().item(),), low=min_token_id, high=max_token_id, device=device
                )

            masked_input[masked_random_tokens] = random_tokens[masked_random_tokens]

        # Forward pass --------------------------------------------------------
        t_start_forward = time.time()
        # N.B. To accurately time steps on GPU we need to use torch.cuda.Event
        ct_forward = torch.cuda.Event(enable_timing=True)
        ct_forward.record()
        # Perform the forward pass through the model
        print(config.arch)
        if config.arch == "maelm":
            print("MAELM is implemented")
            out = model(masked_input, att_mask, masked_unseen_tokens, config.maelm_version)
        elif config.arch == "transformer":
            out = model(masked_input, attention_mask=att_mask)

        logits = out.logits.view(-1, n_output_tokens)

        # Measure loss
        if config.pretrain_levenshtein:
            soft_targets = levenshtein.softmax_batch_levenshtein_matrices_vectorized(
                sequences, config.k_mer, dataloader.dataset.vocab
            ).to(device)
            with torch.no_grad():
                targets = torch.argmax(soft_targets, dim=-1)
            soft_targets = soft_targets.view(-1, n_output_tokens)
            if config.separate_loss:

                masked_indices = input_maskout.view(-1)
                non_masked_indices = ~masked_indices & special_tokens_mask.view(-1)  # ignore special tokens

                masked_loss = criterion(logits[masked_indices], soft_targets[masked_indices])
                non_masked_loss = criterion(logits[non_masked_indices], soft_targets[non_masked_indices])

                masked_loss_weight = config.masked_loss_weight
                non_masked_loss_weight = 1 - masked_loss_weight
                loss = masked_loss_weight * masked_loss + non_masked_loss_weight * non_masked_loss
            else:
                loss = criterion(logits[special_tokens_mask.view(-1)], soft_targets[special_tokens_mask.view(-1)])

        else:
            # Need to remove the special token from the index in sequences
            targets = sequences - n_special_tokens * (sequences > (n_special_tokens - 1))

            if config.separate_loss:
                logits = out.logits.view(-1, n_output_tokens)
                targets_flat = targets.view(-1)

                masked_indices = input_maskout.view(-1)
                non_masked_indices = ~masked_indices & special_tokens_mask.view(-1)  # ignore special tokens

                masked_loss = criterion(logits[masked_indices], targets_flat[masked_indices])
                non_masked_loss = criterion(logits[non_masked_indices], targets_flat[non_masked_indices])

                masked_loss_weight = config.masked_loss_weight
                non_masked_loss_weight = 1 - masked_loss_weight
                loss = masked_loss_weight * masked_loss + non_masked_loss_weight * non_masked_loss

            # Need to remove the <UNK> and <CLS> tokens from the index in sequences
            else:
                loss = criterion(
                    out.logits.view(-1, n_output_tokens)[special_tokens_mask.view(-1)],
                    targets.view(-1)[special_tokens_mask.view(-1)],
                )

        # Backward pass -------------------------------------------------------
        # Reset gradients
        optimizer.zero_grad()
        # Now the backward pass
        ct_backward = torch.cuda.Event(enable_timing=True)
        ct_backward.record()
        loss.backward()

        # Update --------------------------------------------------------------
        # Use our optimizer to update the model parameters
        ct_optimizer = torch.cuda.Event(enable_timing=True)
        ct_optimizer.record()
        optimizer.step()

        # Step the scheduler each batch
        scheduler.step()

        # Increment training progress counters
        total_step += 1
        batch_size_all = batch_size_this_gpu * config.world_size
        n_samples_seen += batch_size_all

        # Logging -------------------------------------------------------------
        # Log details about training progress
        t_start_logging = time.time()
        ct_logging = torch.cuda.Event(enable_timing=True)
        ct_logging.record()

        # Update the total loss for the epoch
        loss_batch = loss.clone()
        if config.distributed:
            # Fetch results from other GPUs
            dist.reduce(loss_batch, 0, op=dist.ReduceOp.AVG)
        loss_batch = loss_batch.item()
        loss_epoch += loss_batch

        if config.separate_loss:
            masked_loss_batch = masked_loss.clone()
            non_masked_loss_batch = non_masked_loss.clone()

            if config.distributed:
                # Fetch results from other GPUs
                dist.reduce(masked_loss_batch, 0, op=dist.ReduceOp.AVG)
                dist.reduce(non_masked_loss_batch, 0, op=dist.ReduceOp.AVG)

            masked_loss_batch = masked_loss_batch.item()
            non_masked_loss_batch = non_masked_loss_batch.item()

            masked_loss_epoch += masked_loss_batch
            non_masked_loss_epoch += non_masked_loss_batch

        with torch.no_grad():
            x_pred = torch.argmax(out.logits, dim=-1)

        if epoch <= 1 and batch_idx == 0:
            # Debugging
            print("sequences.shape     =", sequences.shape)
            print("y_true.shape        =", y_true.shape)
            print("input_maskout.shape =", input_maskout.shape)
            print("masked_input.shape  =", masked_input.shape)
            print("targets.shape       =", targets.shape)
            print("x_pred.shape        =", x_pred.shape)
            print("logits.shape        =", out.logits.shape)
            print("loss.shape          =", loss.shape)
            # Debugging intensifies
            print("sequences[0]     =", sequences[0])
            print("attention_mask[0]=", att_mask[0])
            print("input_maskout[0] =", input_maskout[0])
            print("y_true[0]        =", y_true[0])
            print("masked_input[0]  =", masked_input[0])
            print("targets[0]       =", targets[0])
            print("x_pred[0]        =", x_pred[0])
            print("logits[0]        =", out.logits[0])
            print("loss =", loss.detach().item())

        # Compute accuracy
        with torch.no_grad():
            is_correct = x_pred == targets
            # Overall accuracy, including tokens which weren't masked out
            acc_all = is_correct[special_tokens_mask].sum() / is_correct[special_tokens_mask].numel()
            # Accuracy only on the masked tokens
            acc_msk = (
                is_correct[input_maskout & special_tokens_mask].sum() / (input_maskout & special_tokens_mask).sum()
            )
            # Accuracy only on the non-masked tokens
            acc_kpt = (
                is_correct[~input_maskout & special_tokens_mask].sum() / (~input_maskout & special_tokens_mask).sum()
            )
            if config.distributed:
                # Fetch results from other GPUs
                dist.reduce(acc_all, 0, op=dist.ReduceOp.AVG)
                dist.reduce(acc_msk, 0, op=dist.ReduceOp.AVG)
                dist.reduce(acc_kpt, 0, op=dist.ReduceOp.AVG)
            acc_all = 100.0 * acc_all.item()
            acc_msk = 100.0 * acc_msk.item()
            acc_kpt = 100.0 * acc_kpt.item()
            acc_epoch += acc_msk
            acc_kpt_epoch += acc_kpt
            acc_all_epoch += acc_all

        # Log to console
        if batch_idx <= 2 or batch_idx % config.print_interval == 0 or batch_idx >= len(dataloader) - 1:
            if config.separate_loss:
                print(
                    f"Train Epoch:{epoch:3d}" + (f"/{n_epoch}" if n_epoch is not None else ""),
                    " Step:{:6d}/{}".format(batch_idx + 1, len(dataloader)),
                    " LossMask:{:8.5f}".format(masked_loss_batch),
                    " LossSeen:{:8.5f}".format(non_masked_loss_batch),
                    " Loss:{:8.5f}".format(loss_batch),
                    " AccMask:{:6.2f}%".format(acc_msk),
                    " AccSeen:{:6.2f}%".format(acc_kpt),
                    " LR: {}".format(scheduler.get_last_lr()),
                    flush=True,
                )
            else:
                print(
                    f"Train Epoch:{epoch:3d}" + (f"/{n_epoch}" if n_epoch is not None else ""),
                    " Step:{:6d}/{}".format(batch_idx + 1, len(dataloader)),
                    " Loss:{:8.5f}".format(loss_batch),
                    " AccMask:{:6.2f}%".format(acc_msk),
                    " AccSeen:{:6.2f}%".format(acc_kpt),
                    " LR: {}".format(scheduler.get_last_lr()),
                    flush=True,
                )

        # Compute mask ratio actually used
        actual_masking = masked_unseen_tokens.sum() / masked_unseen_tokens.nelement()
        actual_random_token = masked_random_tokens.sum() / masked_random_tokens.nelement()
        actual_original_token = masked_original_tokens.sum() / masked_original_tokens.nelement()

        mask_ratio_actual = input_maskout.sum() / input_maskout.nelement()
        if config.distributed:
            dist.reduce(mask_ratio_actual, 0, op=dist.ReduceOp.AVG)

        # Log to wandb
        if config.log_wandb and config.global_rank == 0 and batch_idx % config.log_interval == 0:
            # Create a log dictionary to send to wandb
            # Epoch progress interpolates smoothly between epochs
            epoch_progress = epoch - 1 + (batch_idx + 1) / len(dataloader)
            # Throughput is the number of samples processed per second
            throughput = batch_size_all / (t_start_logging - t_end_batch)
            log_dict = {
                "Pretraining/stepwise/epoch": epoch,
                "Pretraining/stepwise/epoch_progress": epoch_progress,
                "Pretraining/stepwise/n_samples_seen": n_samples_seen,
                "Pretraining/stepwise/Train/throughput": throughput,
                "Pretraining/stepwise/Train/loss": loss_batch,
                "Pretraining/stepwise/Train/accuracy": acc_msk,
                "Pretraining/stepwise/Train/accuracy_unmasked": acc_kpt,
                "Pretraining/stepwise/Train/accuracy_overall": acc_all,
                "Pretraining/stepwise/Train/mask_ratio": mask_ratio,
                "Pretraining/stepwise/Train/mask_ratio_actual": mask_ratio_actual,
                "Pretraining/stepwise/Train/mask_ratio_masking": actual_masking,
                "Pretraining/stepwise/Train/mask_ratio_random_token": actual_random_token,
                "Pretraining/stepwise/Train/mask_ratio_original_token": actual_original_token,
            }
            if config.separate_loss:
                log_dict = {
                    "Pretraining/stepwise/epoch": epoch,
                    "Pretraining/stepwise/epoch_progress": epoch_progress,
                    "Pretraining/stepwise/n_samples_seen": n_samples_seen,
                    "Pretraining/stepwise/Train/throughput": throughput,
                    "Pretraining/stepwise/Train/loss": loss_batch,
                    "Pretraining/stepwise/Train/loss_masked": masked_loss_batch,
                    "Pretraining/stepwise/Train/loss_unmasked": non_masked_loss_batch,
                    "Pretraining/stepwise/Train/accuracy": acc_msk,
                    "Pretraining/stepwise/Train/accuracy_unmasked": acc_kpt,
                    "Pretraining/stepwise/Train/accuracy_overall": acc_all,
                    "Pretraining/stepwise/Train/mask_ratio": mask_ratio,
                    "Pretraining/stepwise/Train/mask_ratio_actual": mask_ratio_actual,
                    "Pretraining/stepwise/Train/mask_ratio_masking": actual_masking,
                    "Pretraining/stepwise/Train/mask_ratio_random_token": actual_random_token,
                    "Pretraining/stepwise/Train/mask_ratio_original_token": actual_original_token,
                }
            # Track the learning rate of each parameter group
            for lr_idx in range(len(optimizer.param_groups)):
                if "name" in optimizer.param_groups[lr_idx]:
                    grp_name = optimizer.param_groups[lr_idx]["name"]
                elif len(optimizer.param_groups) == 1:
                    grp_name = ""
                else:
                    grp_name = f"grp{lr_idx}"
                if grp_name != "":
                    grp_name = f"-{grp_name}"
                grp_lr = optimizer.param_groups[lr_idx]["lr"]
                log_dict[f"Pretraining/stepwise/lr{grp_name}"] = grp_lr
            # Synchronize ensures everything has finished running on each GPU
            torch.cuda.synchronize()
            # Record how long it took to do each step in the pipeline
            if t_start_wandb is not None:
                # Record how long it took to send to wandb last time
                log_dict["Pretraining/stepwise/duration/wandb"] = t_end_wandb - t_start_wandb
            log_dict["Pretraining/stepwise/duration/dataloader"] = t_start_batch - t_end_batch
            log_dict["Pretraining/stepwise/duration/preamble"] = t_start_forward - t_start_batch
            log_dict["Pretraining/stepwise/duration/forward"] = ct_forward.elapsed_time(ct_backward) / 1000
            log_dict["Pretraining/stepwise/duration/backward"] = ct_backward.elapsed_time(ct_optimizer) / 1000
            log_dict["Pretraining/stepwise/duration/optimizer"] = ct_optimizer.elapsed_time(ct_logging) / 1000
            log_dict["Pretraining/stepwise/duration/overall"] = time.time() - t_end_batch
            t_start_wandb = time.time()
            log_dict["Pretraining/stepwise/duration/logging"] = t_start_wandb - t_start_logging
            # Send to wandb
            wandb.log(log_dict, step=total_step)
            t_end_wandb = time.time()

        # Record the time when we finished this batch
        t_end_batch = time.time()

    if config.separate_loss:
        results = {
            "loss": loss_epoch / (batch_idx + 1),
            "loss_masked": masked_loss_epoch / (batch_idx + 1),
            "loss_non_masked": non_masked_loss_epoch / (batch_idx + 1),
            "accuracy": acc_epoch / (batch_idx + 1),
            "accuracy_unmasked": acc_kpt_epoch / (batch_idx + 1),
            "accuracy_overall": acc_all_epoch / (batch_idx + 1),
        }
    else:
        results = {
            "loss": loss_epoch / (batch_idx + 1),
            "accuracy": acc_epoch / (batch_idx + 1),
            "accuracy_unmasked": acc_kpt_epoch / (batch_idx + 1),
            "accuracy_overall": acc_all_epoch / (batch_idx + 1),
        }

    if biological_masker is not None:
        tokens_to_replace = sequences[masked_random_tokens]
        biological_replacements = get_biological_replacements(tokens_to_replace, biological_masker)
        masked_input[masked_random_tokens] = biological_replacements

        # SIMPLE DEBUG - only first batch of first epoch
        if batch_idx == 0 and epoch == 1:
            print(f"\n🔬 Biological masking check:")
            for i in range(min(5, len(tokens_to_replace))):
                orig_id = tokens_to_replace[i].item()
                repl_id = biological_replacements[i].item()
                orig_kmer = dataloader.dataset.vocab.lookup_token(orig_id)
                repl_kmer = dataloader.dataset.vocab.lookup_token(repl_id)
                same = "❌ SAME!" if orig_id == repl_id else "✅"
                print(f"  {orig_kmer} → {repl_kmer} {same}")

    return results, total_step, n_samples_seen


def evaluate(
    config,
    model,
    criterion,
    dataloader,
    num_labels,
    mask_ratio=0.5,
    device="cuda",
    distance_table=None,
    n_special_tokens=2,
    n_all_tokens=-1,
    biological_masker=None
):
    r"""
    Evaluate the encoder on the validation data.

    Parameters
    ----------
    config : argparse.Namespace or OmegaConf
        The global config object.
    model : torch.nn.Module
        The encoder/decoder network.
    criterion : torch.nn.Module
        The loss function.
    dataloader : torch.utils.data.DataLoader
        A dataloader for the training set.
    num_labels : int,
        The number of labels (number of different tokens)
    mask_ratio : float, default=0.5
        The ratio of tokens to mask out.
    device : str or torch.device, default="cuda"
        The device to use.
    distance_table : torch.Tensor, optional
        A pre-computed table of Levenshtein distances between all possible k-mers.

    Returns
    -------
    results: dict
        A dictionary containing the evaluation performance.
    """
    # Put the model in train mode
    model.eval()

    loss_epoch = 0
    masked_loss_epoch = 0
    non_masked_loss_epoch = 0
    acc_epoch = 0
    acc_kpt_epoch = 0
    acc_all_epoch = 0
    n_samples = 0

    base_pairs = "ACGT"
    if config.predict_n_nucleotide:
        base_pairs += "N"

    n_output_tokens = num_labels  # len(base_pairs) ** config.k_mer

    if config.print_interval is None:
        # Default to printing to console every time we log to wandb
        config.print_interval = config.log_interval

    # Set the random seed to be stable for the evaluation
    # (This is stable if you change the batch size, but not if you change the number of GPU workers)
    rng = torch.Generator(device=device).manual_seed(config.global_rank)

    with torch.no_grad():
        for batch_idx, (sequences, _y_true, att_mask) in enumerate(dataloader):
            batch_size_this_gpu = sequences.shape[0]

            # Move training inputs and targets to the GPU
            sequences = sequences.to(device)
            att_mask = att_mask.to(device)

            # Build the masking on the fly ----------------------------------------
            # t_start_masking = time.time()

            # Create a mask for allowed tokens i.e. that excludes all special tokens [<MASK>, <UNK>]
            special_tokens_mask = sequences > (n_special_tokens - 1)

            if config.tokenize_n_nucleotide:
                # Either exlude the last token [N..N] if config.predict_n_nucleotide == True
                # Or exclude all tokens containing Ns i.e "bad kamers" whose index in the vocab
                # is greater than 4**k
                special_tokens_mask &= sequences < (n_special_tokens + n_output_tokens - 1)

            special_tokens_mask = special_tokens_mask.to(device)
            masked_input = sequences.clone()
            random_mask = torch.rand(masked_input.shape, generator=rng, device=device)

            mask_token_ratio = config.mask_token_ratio
            random_token_ratio = config.random_token_ratio + config.mask_token_ratio

            masked_unseen_tokens = (random_mask < mask_token_ratio * mask_ratio) & special_tokens_mask
            masked_random_tokens = (
                (random_mask >= mask_token_ratio * mask_ratio)
                & (random_mask < random_token_ratio * mask_ratio)
                & special_tokens_mask
            )
            masked_original_tokens = (
                (random_mask >= random_token_ratio * mask_ratio) & (random_mask < mask_ratio) & special_tokens_mask
            )

            input_maskout = masked_unseen_tokens | masked_random_tokens | masked_original_tokens
            # Apply the masks
            masked_input[masked_unseen_tokens] = 0  # Masking the token
            # Keep original tokens where mask_keep_original is True (no action needed)
            # Replace with random token where mask_random_token is True
            # Generate random tokens
            if biological_masker is not None:
                tokens_to_replace = sequences[masked_random_tokens]
                # Get biological replacements (guaranteed different from originals)
                biological_replacements = biological_masker.get_biological_replacements(tokens_to_replace)
                masked_input[masked_random_tokens] = biological_replacements

            else:
                min_token_id = n_special_tokens  # 0 is for masking, 1 is for <UNK>
                max_token_id = n_all_tokens  # number of all tokens (including special)
                random_tokens = torch.randint_like(masked_input, low=min_token_id, high=max_token_id)

                # Ensure random tokens are not the same as the original tokens
                while True:
                    same_as_original = (random_tokens == masked_input) & masked_random_tokens
                    if not same_as_original.any():
                        break
                    random_tokens[same_as_original] = torch.randint(
                        size=(same_as_original.sum().item(),), low=min_token_id, high=max_token_id, device=device
                    )

                masked_input[masked_random_tokens] = random_tokens[masked_random_tokens]

            # Forward pass ----------------------------------------------------
            if config.arch == "maelm":
                out = model(masked_input, att_mask, masked_unseen_tokens, config.maelm_version)
            elif config.arch == "transformer":
                out = model(masked_input, attention_mask=att_mask)

            logits = out.logits.view(-1, n_output_tokens)

            # Measure loss
            if config.pretrain_levenshtein:
                soft_targets = levenshtein.softmax_batch_levenshtein_matrices_vectorized(
                    sequences, config.k_mer, dataloader.dataset.vocab
                ).to(device)
                with torch.no_grad():
                    targets = torch.argmax(soft_targets, dim=-1)
                soft_targets = soft_targets.view(-1, n_output_tokens)
                if config.separate_loss:

                    masked_indices = input_maskout.view(-1)
                    non_masked_indices = ~masked_indices & special_tokens_mask.view(-1)  # ignore special tokens

                    masked_loss = criterion(logits[masked_indices], soft_targets[masked_indices])
                    non_masked_loss = criterion(logits[non_masked_indices], soft_targets[non_masked_indices])

                    masked_loss_weight = config.masked_loss_weight
                    non_masked_loss_weight = 1 - masked_loss_weight
                    loss = masked_loss_weight * masked_loss + non_masked_loss_weight * non_masked_loss
                else:
                    loss = criterion(logits[special_tokens_mask.view(-1)], soft_targets[special_tokens_mask.view(-1)])

            else:
                # Need to remove the special token from the index in sequences
                targets = sequences - n_special_tokens * (sequences > (n_special_tokens - 1))

                if config.separate_loss:
                    targets_flat = targets.view(-1)

                    masked_indices = input_maskout.view(-1)
                    non_masked_indices = ~masked_indices & special_tokens_mask.view(-1)  # ignore special tokens

                    masked_loss = criterion(logits[masked_indices], targets_flat[masked_indices])
                    non_masked_loss = criterion(logits[non_masked_indices], targets_flat[non_masked_indices])

                    masked_loss_weight = config.masked_loss_weight
                    non_masked_loss_weight = 1 - masked_loss_weight
                    loss = masked_loss_weight * masked_loss + non_masked_loss_weight * non_masked_loss

                # Need to remove the <UNK> and <CLS> tokens from the index in sequences
                else:
                    loss = criterion(
                        out.logits.view(-1, n_output_tokens)[special_tokens_mask.view(-1)],
                        targets.view(-1)[special_tokens_mask.view(-1)],
                    )

            # Metrics ---------------------------------------------------------
            # Update the total loss for the epoch
            loss_batch = loss * batch_size_this_gpu
            if config.distributed:
                # Fetch results from other GPUs
                dist.reduce(loss_batch, 0, op=dist.ReduceOp.SUM)
            loss_batch = loss_batch.item()
            loss_epoch += loss_batch

            if config.separate_loss:
                masked_loss_batch = masked_loss * batch_size_this_gpu
                non_masked_loss_batch = non_masked_loss * batch_size_this_gpu

                if config.distributed:
                    # Fetch results from other GPUs
                    dist.reduce(masked_loss_batch, 0, op=dist.ReduceOp.SUM)
                    dist.reduce(non_masked_loss_batch, 0, op=dist.ReduceOp.SUM)

                masked_loss_batch = masked_loss_batch.item()
                non_masked_loss_batch = non_masked_loss_batch.item()

                masked_loss_epoch += masked_loss_batch
                non_masked_loss_epoch += non_masked_loss_batch

            # Compute accuracy
            with torch.no_grad():
                x_pred = torch.argmax(out.logits, dim=-1)
            # Create a mask to ignore all special tokens [<MASK>, <UNK>]
            special_tokens_mask = targets > (n_special_tokens - 1)

            is_correct = x_pred == targets
            # Overall accuracy, including tokens which weren't masked out
            acc_all = (
                batch_size_this_gpu * is_correct[special_tokens_mask].sum() / is_correct[special_tokens_mask].numel()
            )
            # Accuracy only on the masked tokens
            acc_msk = (
                batch_size_this_gpu
                * is_correct[input_maskout & special_tokens_mask].sum()
                / (input_maskout & special_tokens_mask).sum()
            )
            # Accuracy only on the non-masked tokens
            acc_kpt = (
                batch_size_this_gpu
                * is_correct[~input_maskout & special_tokens_mask].sum()
                / (~input_maskout & special_tokens_mask).sum()
            )
            if config.distributed:
                # Fetch results from other GPUs
                dist.reduce(acc_all, 0, op=dist.ReduceOp.SUM)
                dist.reduce(acc_msk, 0, op=dist.ReduceOp.SUM)
                dist.reduce(acc_kpt, 0, op=dist.ReduceOp.SUM)
            acc_epoch += acc_msk
            acc_kpt_epoch += acc_kpt
            acc_all_epoch += acc_all

            # Total batch size
            if config.distributed:
                batch_size = torch.tensor(batch_size_this_gpu, device=device)
                dist.reduce(batch_size, 0, op=dist.ReduceOp.SUM)
                batch_size = batch_size.item()
            else:
                batch_size = batch_size_this_gpu
            n_samples += batch_size

            if batch_idx % (config.print_interval * 10) == 0:
                if config.separate_loss:
                    print(
                        "Eval",
                        " Step:{:6d}/{}".format(batch_idx + 1, len(dataloader)),
                        " Loss:{:8.5f}".format(loss_batch / batch_size),
                        " LossMask:{:8.5f}".format(masked_loss_batch / batch_size),
                        " LossSeen:{:8.5f}".format(non_masked_loss_batch / batch_size),
                        " AccMask:{:6.2f}%".format(100.0 * acc_msk / batch_size),
                        " AccSeen:{:6.2f}%".format(100.0 * acc_kpt / batch_size),
                        flush=True,
                    )
                else:
                    print(
                        "Eval",
                        " Step:{:6d}/{}".format(batch_idx + 1, len(dataloader)),
                        " Loss:{:8.5f}".format(loss_batch / batch_size),
                        " AccMask:{:6.2f}%".format(100.0 * acc_msk / batch_size),
                        " AccSeen:{:6.2f}%".format(100.0 * acc_kpt / batch_size),
                        flush=True,
                    )

    if config.separate_loss:
        results = {
            "loss": loss_epoch / n_samples,
            "loss_masked": masked_loss_epoch / n_samples,
            "loss_non_masked": non_masked_loss_epoch / n_samples,
            "accuracy": 100.0 * acc_epoch / n_samples,
            "accuracy_unmasked": 100.0 * acc_kpt_epoch / n_samples,
            "accuracy_overall": 100.0 * acc_all_epoch / n_samples,
            "n_samples": n_samples,
        }
    else:
        results = {
            "loss": loss_epoch / n_samples,
            "accuracy": 100.0 * acc_epoch / n_samples,
            "accuracy_unmasked": 100.0 * acc_kpt_epoch / n_samples,
            "accuracy_overall": 100.0 * acc_all_epoch / n_samples,
            "n_samples": n_samples,
        }

    return results


def get_parser():
    r"""
    Build argument parser for the command line interface.

    Returns
    -------
    parser : argparse.ArgumentParser
        CLI argument parser.
    """
    import argparse
    import sys

    # Use the name of the file called to determine the name of the program
    prog = os.path.split(sys.argv[0])[1]
    if prog == "__main__.py" or prog == "__main__":
        # If the file is called __main__.py, go up a level to the module name
        prog = os.path.split(__file__)[1]
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Pretrain BarcodeBERT.",
        add_help=False,
    )
    # Help arg ----------------------------------------------------------------
    group = parser.add_argument_group("Help")
    group.add_argument(
        "--help",
        "-h",
        action="help",
        help="Show this help message and exit.",
    )
    # Dataset args ------------------------------------------------------------
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory within which the dataset can be found. Default: %(default)s",
    )
    group.add_argument(
        "--dataset",
        dest="dataset_name",
        type=str,
        default="CANADA-1.5M",
        required=True,
        help="This specifies the format of the .csv files in data-dir. \
              In particular if the dataframe contains an index for each taxon. \
              Default: %(default)s",
    )
    group.add_argument(
        "--tokenizer",
        type=str,
        default="kmer",
        help="Name of tokenizer to use for DNA sequences. Default: %(default)s",
    )
    group.add_argument(
        "--bpe-path",
        "--bpe_path",
        type=str,
        default="./",
        help="Path of the bpe tokenizer to use for DNA sequences. Default: %(default)s",
    )
    group.add_argument(
        "--k-mer",
        "--k_mer",
        type=int,
        default=6,
        help="Size of k-mer to use for DNA tokenization. Default: %(default)s",
    )
    group.add_argument(
        "--stride",
        type=int,
        help="Stride to use for DNA tokenization. Default: Same as k-mer size.",
    )
    group.add_argument(
        "--max-len",
        "--max_len",
        type=int,
        default=660,
        help="Maximum length of input sequences. Default: %(default)s",
    )
    # Architecture args -------------------------------------------------------
    group = parser.add_argument_group("Architecture")
    group.add_argument(
        "--model",
        "--encoder",
        "--arch",
        "--architecture",
        dest="arch",
        type=str,
        default="transformer",
        help="Name of model architecture. Default: %(default)s",
    )

    group.add_argument(
        "--n-layers",
        "--n_layers",
        type=int,
        default=6,
        help="Number of layers in the transformer. Default: %(default)s",
    )

    group.add_argument(
        "--n-heads",
        "--n_heads",
        type=int,
        default=6,
        help="Number of attention heads in the transformer. Default: %(default)s",
    )

    group.add_argument(
        "--decoder-n-layers",
        "--decoder_n_layers",
        type=int,
        default=6,
        help="Number of attention heads in the decoder of MAELM. Default: %(default)s",
    )

    group.add_argument(
        "--decoder-n-heads",
        "--decoder_n_heads",
        type=int,
        default=6,
        help="Number of attention heads in the decoder of MAELM. Default: %(default)s",
    )

    group.add_argument(
        "--decoder-embed-dim",
        "--decoder_embed_dim",
        type=int,
        default=768,
        help="Size of the decoder embedding in the decoder of MAE-LM. Default: %(default)s",
    )

    group.add_argument(
        "--maelm-version",
        "--maelm_version",
        type=str,
        default="maelm_v2",
        help="which implementation we want to use. Default: %(default)s",
    )

    # Optimization args -------------------------------------------------------
    group = parser.add_argument_group("Optimization routine")
    group.add_argument(
        "--epochs",
        type=int,
        default=45,
        help="Number of epochs to train for. Default: %(default)s",
    )
    group.add_argument(
        "--lr",
        dest="lr_relative",
        type=float,
        default=0.0001,
        help=(
            f"Maximum learning rate, set per {BASE_BATCH_SIZE} batch size."
            " The actual learning rate used will be scaled up by the total"
            " batch size (across all GPUs). Default: %(default)s"
        ),
    )
    group.add_argument(
        "--weight-decay",
        "--weight_decay",
        "--wd",
        dest="weight_decay",
        type=float,
        default=0.00001,
        help="Weight decay. Default: %(default)s",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Name of optimizer (case-sensitive). Default: %(default)s",
    )
    group.add_argument(
        "--scheduler",
        type=str,
        default="OneCycle",
        help="Learning rate scheduler. Default: %(default)s",
    )
    parser.add_argument(
        "--tokenize-n-nucleotide",
        "--tokenize_n_nucleotide",
        "--tokenize-n",
        "--tokenize_n",
        dest="tokenize_n_nucleotide",
        action="store_true",
        help="Include N as a valid character in tokenization. If false, all kmers including Ns will be mappend to [UNK]",
    )
    parser.add_argument(
        "--predict-n-nucleotide",
        "--predict_n_nucleotide",
        "--predict-n",
        "--predict_n",
        dest="predict_n_nucleotide",
        action="store_true",
        help="Predict tokens containing Ns. This will increase the size of the output space and will force tokenize_n_nucleotide to True",
    )
    parser.add_argument(
        "--pretrain-levenshtein",
        "--pretrain_levenshtein",
        action="store_true",
        help="Use Levenshtein distance for the pretraining cross-entropy loss.",
    )
    parser.add_argument(
        "--levenshtein-vectorized",
        "--levenshtein_vectorized",
        action="store_true",
        help="Use the vectorized implementation of Levenshtein distance computation.",
    )
    parser.add_argument(
        "--separate-loss",
        "--separate_loss",
        type=bool,
        default=True,
        help="Set to True to use the weighted loss for masked and non_masked components (default: True).",
    )
    parser.add_argument(
        "--masked-loss-weight",
        "--masked_loss_weight",
        type=float,
        default=1.0,
        help="masked loss component weight (non_masked_weight = 1 - masked_loss_weight)",
    )
    parser.add_argument(
        "--mask-token-ratio",
        "--mask_token_ratio",
        type=float,
        default=1.0,
        help="actual masking ratio",
    )
    parser.add_argument(
        "--random-token-ratio",
        "--random_token_ratio",
        type=float,
        default=0,
        help="random token ratio",
    )

    # Output checkpoint args --------------------------------------------------
    group = parser.add_argument_group("Output checkpoint")
    group.add_argument(
        "--models-dir",
        type=str,
        default="model_checkpoints",
        metavar="PATH",
        help="Output directory for all models. Ignored if --checkpoint is set. Default: %(default)s",
    )
    group.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        default="",
        type=str,
        metavar="PATH",
        help=(
            "Save and resume partially trained model and optimizer state from this checkpoint."
            " Overrides --models-dir."
        ),
    )
    group.add_argument(
        "--checkpoint-resume",
        "--checkpoint_resume",
        dest="checkpoint_path_resume",
        default="",
        type=str,
        metavar="PATH",
        help=(
            "Resume partially trained model and optimizer state from this checkpoint,"
            " if CHECKPOINT_PATH does not exist."
        ),
    )
    group.add_argument(
        "--save-best-model",
        action="store_true",
        help="Save a copy of the model with best validation performance.",
    )
    # Reproducibility args ----------------------------------------------------
    group = parser.add_argument_group("Reproducibility")
    group.add_argument(
        "--seed",
        type=int,
        help="Random number generator (RNG) seed. Default: not controlled",
    )
    group.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable non-deterministic features of cuDNN.",
    )
    # Hardware configuration args ---------------------------------------------
    group = parser.add_argument_group("Hardware configuration")
    group.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size_per_gpu",
        type=int,
        default=16,
        help=(
            "Batch size per GPU. The total batch size will be this value times"
            " the total number of GPUs used. Default: %(default)s"
        ),
    )
    group.add_argument(
        "--cpu-workers",
        "--cpu_workers",
        "--workers",
        dest="cpu_workers",
        type=int,
        help="Number of CPU workers per node. Default: number of CPUs available on device.",
    )
    group.add_argument(
        "--no-cuda",
        action="store_true",
        help="Use CPU only, no GPUs.",
    )
    group.add_argument(
        "--gpu",
        "--local-rank",
        dest="local_rank",
        default=None,
        type=int,
        help="Index of GPU to use when training a single process. (Ignored for distributed training.)",
    )
    # Logging args ------------------------------------------------------------
    group = parser.add_argument_group("Debugging and logging")
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Number of batches between each log to wandb (if enabled). Default: %(default)s",
    )
    group.add_argument(
        "--print-interval",
        type=int,
        default=None,
        help="Number of batches between each print to STDOUT. Default: same as LOG_INTERVAL.",
    )
    group.add_argument(
        "--log-wandb",
        action="store_true",
        help="Log results with Weights & Biases https://wandb.ai",
    )
    group.add_argument(
        "--disable-wandb",
        "--disable_wandb",
        "--no-wandb",
        dest="disable_wandb",
        action="store_true",
        help="Overrides --log-wandb and ensures wandb is always disabled.",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        default="uoguelph_mlrg",
        help="The entity (organization) within which your wandb project is located. Default: %(default)s",
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="BarcodeBERT",
        help="Name of project on wandb, where these runs will be saved. Default: %(default)s",
    )
    group.add_argument(
        "--run-name",
        type=str,
        help="Human-readable identifier for the model run or job. Used to name the run on wandb.",
    )
    group.add_argument(
        "--run-id",
        type=str,
        help="Unique identifier for the model run or job. Used as the run ID on wandb.",
    )

    return parser


def cli():
    r"""Command-line interface for model training."""
    parser = get_parser()
    config = parser.parse_args()
    # Handle disable_wandb overriding log_wandb and forcing it to be disabled.
    if config.disable_wandb:
        config.log_wandb = False
    del config.disable_wandb
    if config.levenshtein_vectorized:
        # If the vectorized implementation of Levenshtein distances was requested,
        # we must be using Levenshtein distances with soft labels.
        config.pretrain_levenshtein = True
    if config.pretrain_levenshtein:
        # If we are pretraining with Levenshtein distances, we can't use the <UNK> token.
        config.tokenize_n_nucleotide = True

    if config.predict_n_nucleotide:
        # If we ask to predict Ns, then the tokenizer must accept Ns
        if not config.tokenize_n_nucleotide:
            print(
                "Predict_n_nucleotide is set to true, \
                  setting tokenize_n_nucleotide to True"
            )
            config.tokenize_n_nucleotide = True
    return run(config)


if __name__ == "__main__":
    cli()
