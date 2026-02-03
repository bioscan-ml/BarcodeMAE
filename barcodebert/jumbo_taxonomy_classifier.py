"""
Binary taxonomy classification head for Jumbo CLS tokens.

This module adds a binary classification task during pretraining to predict
whether pairs of sequences share the same taxonomic label at a specified level
(e.g., phylum, class, order, family, genus, or species).
"""

import torch
import torch.nn as nn


class JumboTaxonomyClassifier(nn.Module):
    """
    Binary classifier that takes two jumbo representations and predicts
    if they share the same taxonomic label at a specified level.
    """

    def __init__(self, jumbo_dim, hidden_dim=256, dropout=0.1):
        """
        Args:
            jumbo_dim: Dimension of flattened jumbo representation (J * D)
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.classifier = nn.Sequential(
            # Input is concatenation of two jumbo representations
            nn.Linear(jumbo_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, jumbo_rep1, jumbo_rep2):
        """
        Args:
            jumbo_rep1: First jumbo representation (B, jumbo_dim)
            jumbo_rep2: Second jumbo representation (B, jumbo_dim)

        Returns:
            logits: Binary classification logits (B, 1)
        """
        # Concatenate the two representations
        combined = torch.cat([jumbo_rep1, jumbo_rep2], dim=-1)  # (B, jumbo_dim * 2)
        logits = self.classifier(combined)  # (B, 1)
        return logits


def create_taxonomy_pairs(taxonomy_labels, same_ratio=0.5, max_pairs=64, debug_print=True):
    """
    Create pairs of indices for taxonomy classification.

    Enumerates all possible positive (same taxon) and negative (different taxon)
    pairs, then randomly samples up to max_pairs while preserving same_ratio.
    If one pool has fewer pairs than its target, the remaining budget is given
    to the other pool.

    Args:
        taxonomy_labels: Tensor of taxonomic labels (B,). Negative values are invalid.
        same_ratio: Target ratio of positive (same-taxon) pairs. Default: 0.5.
        max_pairs: Maximum total number of pairs to return. Default: 64.
        debug_print: If True, print debug info.

    Returns:
        idx1: First indices in pairs (num_pairs,) or None if no pairs possible
        idx2: Second indices in pairs (num_pairs,) or None if no pairs possible
        labels: Binary labels - 1 if same taxon, 0 if different (num_pairs,) or None
        num_same: Number of positive pairs
        num_diff: Number of negative pairs
    """
    device = taxonomy_labels.device

    # Filter out invalid taxonomy labels (< 0)
    valid_mask = taxonomy_labels >= 0
    valid_indices = torch.where(valid_mask)[0]
    valid_labels = taxonomy_labels[valid_indices]
    valid_batch_size = len(valid_indices)

    if debug_print:
        num_invalid = (taxonomy_labels < 0).sum().item()
        print("\n[Taxonomy Pairs Debug]")
        print(f"  Total sequences in batch: {taxonomy_labels.size(0)}")
        print(f"  Valid genus labels: {valid_batch_size}")
        print(f"  Invalid genus labels (skipped): {num_invalid}")
        if valid_batch_size > 0:
            print(f"  First 5 valid genus IDs: {valid_labels[:5].tolist()}")

    if valid_batch_size < 2:
        if debug_print:
            print(f"  WARNING: Only {valid_batch_size} sequence(s) with valid labels. Need at least 2.")
        return None, None, None, 0, 0

    # Enumerate all possible pairs (i < j) among valid sequences
    all_pairs = torch.combinations(torch.arange(valid_batch_size, device=device), r=2)  # (num_all_pairs, 2)

    # Split into positive (same taxon) and negative (different taxon) pools
    same_mask = valid_labels[all_pairs[:, 0]] == valid_labels[all_pairs[:, 1]]
    pos_pairs = all_pairs[same_mask]   # (num_pos_possible, 2)
    neg_pairs = all_pairs[~same_mask]  # (num_neg_possible, 2)

    if debug_print:
        unique_taxa = torch.unique(valid_labels)
        print(f"  Unique genera: {len(unique_taxa)}")
        print(f"  Possible positive pairs: {len(pos_pairs)}, negative pairs: {len(neg_pairs)}")

    if len(pos_pairs) == 0 and len(neg_pairs) == 0:
        if debug_print:
            print("  WARNING: No pairs possible.")
        return None, None, None, 0, 0

    # Compute target counts from same_ratio and max_pairs
    num_pos_target = int(max_pairs * same_ratio)
    num_neg_target = max_pairs - num_pos_target

    # Cap each pool at its target, then redistribute leftover budget
    num_pos = min(num_pos_target, len(pos_pairs))
    num_neg = min(num_neg_target, len(neg_pairs))

    if num_pos < num_pos_target:
        # Positive pool exhausted — give remaining budget to negative
        num_neg = min(len(neg_pairs), max_pairs - num_pos)
    elif num_neg < num_neg_target:
        # Negative pool exhausted — give remaining budget to positive
        num_pos = min(len(pos_pairs), max_pairs - num_neg)

    # Randomly sample from each pool
    if num_pos > 0:
        sampled_pos = pos_pairs[torch.randperm(len(pos_pairs), device=device)[:num_pos]]
    else:
        sampled_pos = torch.empty(0, 2, dtype=torch.long, device=device)

    if num_neg > 0:
        sampled_neg = neg_pairs[torch.randperm(len(neg_pairs), device=device)[:num_neg]]
    else:
        sampled_neg = torch.empty(0, 2, dtype=torch.long, device=device)

    # Combine and map local indices back to original batch indices
    sampled_local = torch.cat([sampled_pos, sampled_neg], dim=0)
    idx1 = valid_indices[sampled_local[:, 0]]
    idx2 = valid_indices[sampled_local[:, 1]]
    labels = torch.cat([
        torch.ones(num_pos, dtype=torch.long, device=device),
        torch.zeros(num_neg, dtype=torch.long, device=device),
    ])

    if debug_print:
        print(f"  Sampled {num_pos} positive pairs, {num_neg} negative pairs (total: {num_pos + num_neg})")

    return idx1, idx2, labels, num_pos, num_neg


def compute_taxonomy_classification_loss(
    jumbo_tokens, taxonomy_labels, classifier, same_ratio=0.5, max_pairs=64, debug_print=False
):
    """
    Compute binary taxonomy classification loss for a batch.

    Args:
        jumbo_tokens: Jumbo token representations from encoder (B, J, D)
        taxonomy_labels: Taxonomic labels for each sequence at specified level (B,)
        classifier: JumboTaxonomyClassifier instance
        same_ratio: Target ratio of positive (same-taxon) pairs
        max_pairs: Maximum total number of pairs to sample
        debug_print: If True, print debugging information about pair creation

    Returns:
        loss: Binary cross-entropy loss (None if no pairs could be created)
        accuracy: Accuracy of predictions (None if no pairs could be created)
        num_pairs: Number of pairs created
        num_same: Number of same-taxonomy pairs
        num_diff: Number of different-taxonomy pairs
    """
    batch_size = jumbo_tokens.size(0)

    # Flatten jumbo tokens to (B, J*D)
    jumbo_flat = jumbo_tokens.reshape(batch_size, -1)

    # Create pairs (filtering out invalid genus labels)
    idx1, idx2, labels, num_same, num_diff = create_taxonomy_pairs(
        taxonomy_labels, same_ratio=same_ratio, max_pairs=max_pairs, debug_print=debug_print
    )

    # Handle case where no pairs could be created
    if idx1 is None:
        return None, None, 0, 0, 0

    # Get representations for pairs
    jumbo_rep1 = jumbo_flat[idx1]
    jumbo_rep2 = jumbo_flat[idx2]

    # Compute logits
    logits = classifier(jumbo_rep1, jumbo_rep2).squeeze(-1)  # (num_pairs,)

    # Compute loss
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

    # Compute accuracy
    predictions = (torch.sigmoid(logits) > 0.5).long()
    accuracy = (predictions == labels).float().mean()

    num_pairs = len(labels)

    return loss, accuracy, num_pairs, num_same, num_diff


# Backward compatibility aliases
JumboGenusClassifier = JumboTaxonomyClassifier
create_genus_pairs = create_taxonomy_pairs
compute_genus_classification_loss = compute_taxonomy_classification_loss
