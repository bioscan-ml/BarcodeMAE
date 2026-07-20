"""
Evaluation routines.
"""

import os

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F

from . import utils


def knn_results_path(results_file, weights):
    r"""
    Route distance-weighted ("soft") kNN results to a separate file from
    uniform-vote results, so the two never collide/overwrite each other in
    the same results file under an identical run-name tag.

    weights="uniform" (the default): returns results_file unchanged.
    weights="distance": inserts "distance_" into the filename, right after
    a leading "KNN_" if present (e.g. "KNN_RESULTS.txt" ->
    "KNN_distance_RESULTS.txt"), otherwise prefixes "distance_".
    """
    if weights != "distance":
        return results_file
    dirname, basename = os.path.split(results_file)
    if basename.startswith("KNN_"):
        basename = "KNN_distance_" + basename[len("KNN_") :]
    else:
        basename = "distance_" + basename
    return os.path.join(dirname, basename)


def knn_vote(neighbor_labels, neighbor_dists=None, weights="uniform"):
    r"""
    Majority-vote (or distance-weighted vote) over each query's k nearest
    neighbors' (already class-index-encoded) labels.

    Parameters
    ----------
    neighbor_labels : np.ndarray of shape (n_queries, k)
        Encoded class index of each of the k nearest neighbors, per query
        (e.g. ``clf._y[neigh_ind]`` from a fitted sklearn KNeighborsClassifier).
    neighbor_dists : np.ndarray of shape (n_queries, k), optional
        Distance to each neighbor, in the same order as neighbor_labels.
        Required if weights="distance".
    weights : {"uniform", "distance"}, default="uniform"
        "uniform": every neighbor gets one vote (plain majority vote).
        "distance": each neighbor's vote is weighted by 1/distance, so
        closer neighbors count more (matches sklearn's own
        KNeighborsClassifier(weights="distance") convention, including its
        handling of exact matches: if any neighbor is at distance 0, only
        those zero-distance neighbors vote).

    Returns
    -------
    np.ndarray of shape (n_queries,)
        Encoded class index predicted for each query.
    """
    if weights == "uniform":
        return np.array([np.bincount(row).argmax() for row in neighbor_labels])
    if weights != "distance":
        raise ValueError(f"Unknown weights mode: {weights!r} (expected 'uniform' or 'distance')")
    if neighbor_dists is None:
        raise ValueError("neighbor_dists is required when weights='distance'")

    preds = np.empty(len(neighbor_labels), dtype=neighbor_labels.dtype)
    for i, (labels_row, dists_row) in enumerate(zip(neighbor_labels, neighbor_dists)):
        zero_mask = dists_row == 0
        if zero_mask.any():
            # Exact matches: only they get a vote (infinite weight in the limit).
            labels_row = labels_row[zero_mask]
            w = np.ones(zero_mask.sum())
        else:
            w = 1.0 / dists_row
        class_weights = {}
        for lbl, wt in zip(labels_row, w):
            class_weights[lbl] = class_weights.get(lbl, 0.0) + wt
        preds[i] = max(class_weights, key=class_weights.get)
    return preds


def evaluate(
    dataloader,
    model,
    device,
    partition_name="Val",
    verbosity=1,
    is_distributed=False,
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

    Returns
    -------
    results : dict
        Dictionary of evaluation results.
    """
    model.eval()

    y_true_all = []
    y_pred_all = []
    xent_all = []

    for sequences, y_true, attn_mask in dataloader:
        sequences = sequences.to(device)
        y_true = y_true.to(device)
        attn_mask = attn_mask.to(device)
        with torch.no_grad():
            logits = model(sequences, mask=attn_mask, labels=y_true).logits
            xent = F.cross_entropy(logits, y_true, reduction="none")
            y_pred = torch.argmax(logits, dim=-1)

        if is_distributed:
            # Fetch results from other GPUs
            xent = utils.concat_all_gather(xent)
            y_true = utils.concat_all_gather(y_true)
            y_pred = utils.concat_all_gather(y_pred)

        xent_all.append(xent.cpu().numpy())
        y_true_all.append(y_true.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())

    # Concatenate the targets and predictions from each batch
    xent = np.concatenate(xent_all)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    # If the dataset size was not evenly divisible by the world size,
    # DistributedSampler will pad the end of the list of samples
    # with some repetitions. We need to trim these off.
    n_samples = len(dataloader.dataset)
    xent = xent[:n_samples]
    y_true = y_true[:n_samples]
    y_pred = y_pred[:n_samples]
    # Create results dictionary
    results = {}
    results["count"] = len(y_true)
    results["cross-entropy"] = np.mean(xent)
    # Note that these evaluation metrics have all been converted to percentages
    results["accuracy"] = 100.0 * sklearn.metrics.accuracy_score(y_true, y_pred)
    results["accuracy-balanced"] = 100.0 * sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
    results["f1-micro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    results["f1-macro"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    results["f1-support"] = 100.0 * sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    # Could expand to other metrics too

    if verbosity >= 1:
        print(f"\n{partition_name} evaluation results:")
        for k, v in results.items():
            if k == "count":
                print(f"  {k + ' ':.<21s}{v:7d}")
            elif "entropy" in k:
                print(f"  {k + ' ':.<24s} {v:9.5f} nat")
            else:
                print(f"  {k + ' ':.<24s} {v:6.2f} %")

    return results
