"""
Evaluation routines.
"""

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F

from . import utils


def evaluate(
    dataloader,
    model,
    device,
    partition_name="Val",
    verbosity=1,
    is_distributed=False,
    eval_level=None,  # NEW: Specify which level to evaluate at (default: model's max_level)
):
    r"""
    Evaluate model performance on a dataset.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset to evaluate on.
    model : torch.nn.Module
        Model to evaluate.
    device : torch.device
        Device to run the model on.
    partition_name : str, default="Val"
        Name of the partition being evaluated.
    verbosity : int, default=1
        Verbosity level.
    is_distributed : bool, default=False
        Whether the model is distributed across multiple GPUs.
    eval_level : int, optional
        Which taxonomic level to evaluate at (0=phylum, 5=species).
        Default: uses model's max_level if InferSum, otherwise None.

    Returns
    -------
    results : dict
        Dictionary of evaluation results.
    """
    model.eval()

    # NEW: Determine if model is InferSum and get max_level
    model_deref = model.module if is_distributed else model
    is_infersum = hasattr(model_deref, 'max_level')
    if is_infersum and eval_level is None:
        eval_level = model_deref.max_level

    y_true_all = []
    y_pred_all = []
    xent_all = []
    valid_mask_all = []  # NEW: Track which samples have valid labels

    for sequences, y_true, attn_mask in dataloader:
        sequences = sequences.to(device)
        y_true = y_true.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            logits = model(sequences, mask=attn_mask).logits  # Don't pass labels to avoid loss computation
            y_pred = torch.argmax(logits, dim=-1)

            # NEW: Handle hierarchical labels (batch, 6) vs single labels (batch,)
            if y_true.dim() > 1:
                # Hierarchical labels - extract the evaluation level
                y_true_single = y_true[:, eval_level]
                valid_mask = y_true_single != -1

                # Compute cross-entropy only for valid samples
                if valid_mask.sum() > 0:
                    xent = torch.full((y_true_single.shape[0],), float('nan'), device=device)
                    xent[valid_mask] = F.cross_entropy(
                        logits[valid_mask],
                        y_true_single[valid_mask],
                        reduction="none"
                    )
                else:
                    xent = torch.full((y_true_single.shape[0],), float('nan'), device=device)

                y_true_for_eval = y_true_single
            else:
                # Single level labels - original behavior
                xent = F.cross_entropy(logits, y_true, reduction="none")
                y_true_for_eval = y_true
                valid_mask = torch.ones(y_true.shape[0], dtype=torch.bool, device=device)

        if is_distributed:
            # Fetch results from other GPUs
            xent = utils.concat_all_gather(xent)
            y_true_for_eval = utils.concat_all_gather(y_true_for_eval)
            y_pred = utils.concat_all_gather(y_pred)
            valid_mask = utils.concat_all_gather(valid_mask)

        xent_all.append(xent.cpu().numpy())
        y_true_all.append(y_true_for_eval.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())
        valid_mask_all.append(valid_mask.cpu().numpy())

    # Concatenate the targets and predictions from each batch
    xent = np.concatenate(xent_all)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    valid_mask = np.concatenate(valid_mask_all)

    # If the dataset size was not evenly divisible by the world size,
    # DistributedSampler will pad the end of the list of samples
    # with some repetitions. We need to trim these off.
    n_samples = len(dataloader.dataset)
    xent = xent[:n_samples]
    y_true = y_true[:n_samples]
    y_pred = y_pred[:n_samples]
    valid_mask = valid_mask[:n_samples]

    # NEW: Filter to only valid samples for metrics
    valid_indices = valid_mask.astype(bool)
    n_valid = valid_indices.sum()

    if n_valid == 0:
        if verbosity >= 1:
            print(f"\n{partition_name} evaluation: No valid samples at this level!")
        return {
            "count": 0,
            "count_total": len(y_true),
            "cross-entropy": float('nan'),
            "accuracy": 0.0,
            "accuracy-balanced": 0.0,
            "f1-micro": 0.0,
            "f1-macro": 0.0,
            "f1-support": 0.0,
        }

    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    xent_valid = xent[valid_indices]

    # Create results dictionary
    results = {}
    results["count"] = int(n_valid)
    results["count_total"] = len(y_true)  # NEW: Total samples including invalid
    results["cross-entropy"] = np.nanmean(xent_valid)

    # Note that these evaluation metrics have all been converted to percentages
    results["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_true_valid, y_pred_valid)
    results["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_true_valid, y_pred_valid)
    results["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_true_valid, y_pred_valid, average="micro", zero_division=0)
    results["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    results["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)

    if verbosity >= 1:
        print(f"\n{partition_name} evaluation results:")
        if is_infersum:
            level_names = ['phylum', 'class', 'order', 'family', 'genus', 'species']
            print(f"  {'Evaluation level ':.<24s} {level_names[eval_level]}")
        for k, v in results.items():
            if k == "count":
                print(f"  {k + ' ':.<21s}{v:7d}")
            elif k == "count_total":
                print(f"  {k + ' ':.<21s}{v:7d}")
            elif "entropy" in k:
                print(f"  {k + ' ':.<24s} {v:9.5f} nat")
            else:
                print(f"  {k + ' ':.<24s} {v:6.2f} %")

    return results


def evaluate_all_levels(
    dataloader,
    model,
    device,
    partition_name="Val",
    verbosity=1,
    is_distributed=False,
):
    """
    NEW: Evaluate model at ALL taxonomic levels (for InferSum models).

    Returns accuracy at each level where samples have valid labels.
    """
    model.eval()

    model_deref = model.module if is_distributed else model
    if not hasattr(model_deref, 'predict_all_levels'):
        raise ValueError("evaluate_all_levels requires an InferSum model with predict_all_levels method")

    level_names = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    max_level = model_deref.max_level

    # Collect predictions at all levels
    all_preds = [[] for _ in range(6)]
    all_labels = [[] for _ in range(6)]

    for sequences, y_true, attn_mask in dataloader:
        sequences = sequences.to(device)
        y_true = y_true.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            # Get predictions at all levels
            all_level_probs = model_deref.predict_all_levels(sequences, attn_mask)

            for lvl in range(6):
                if lvl <= max_level:
                    preds = torch.argmax(all_level_probs[lvl], dim=-1)
                else:
                    preds = torch.zeros(sequences.shape[0], dtype=torch.long, device=device)

                # Handle hierarchical vs single labels
                if y_true.dim() > 1:
                    labels = y_true[:, lvl]
                else:
                    # Single label - only valid for the specified level
                    if lvl == max_level:
                        labels = y_true
                    else:
                        labels = torch.full((y_true.shape[0],), -1, device=device)

                if is_distributed:
                    preds = utils.concat_all_gather(preds)
                    labels = utils.concat_all_gather(labels)

                all_preds[lvl].append(preds.cpu().numpy())
                all_labels[lvl].append(labels.cpu().numpy())

    # Compute metrics for each level
    results = {"partition": partition_name}
    n_samples = len(dataloader.dataset)

    for lvl in range(6):
        preds = np.concatenate(all_preds[lvl])[:n_samples]
        labels = np.concatenate(all_labels[lvl])[:n_samples]

        valid_mask = labels != -1
        n_valid = valid_mask.sum()

        if n_valid > 0:
            preds_valid = preds[valid_mask]
            labels_valid = labels[valid_mask]
            acc = 100.0 * sklearn.metrics.accuracy_score(labels_valid, preds_valid)
        else:
            acc = float('nan')

        results[f"{level_names[lvl]}_accuracy"] = acc
        results[f"{level_names[lvl]}_count"] = int(n_valid)

    if verbosity >= 1:
        print(f"\n{partition_name} multi-level evaluation results:")
        for lvl in range(6):
            acc = results[f"{level_names[lvl]}_accuracy"]
            count = results[f"{level_names[lvl]}_count"]
            if not np.isnan(acc):
                print(f"  {level_names[lvl] + ' ':.<15s} {acc:6.2f} % ({count} samples)")
            else:
                print(f"  {level_names[lvl] + ' ':.<15s} N/A (no valid samples)")

    return results