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

    Supports two modes:
    1. Concatenation mode (default): Takes flattened jumbo reps (B, J*D)
    2. Pooling mode: Takes un-flattened jumbo tokens (B, J, D) and pools them
    """

    def __init__(self, jumbo_dim, hidden_dim=256, dropout=0.1, pool_jumbo=False, pool_type='mean'):
        """
        Args:
            jumbo_dim: Dimension of flattened jumbo representation (J * D)
                      OR single token dimension (D) if pool_jumbo=True
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            pool_jumbo: If True, expects (B, J, D) and pools across J dimension
                       If False, expects (B, J*D) flattened (backward compatible)
            pool_type: Pooling type ('mean' or 'max'), only used if pool_jumbo=True
        """
        super().__init__()
        self.pool_jumbo = pool_jumbo
        self.pool_type = pool_type

        # Determine input dimension to classifier
        if pool_jumbo:
            # Input will be pooled: (B, D) per sequence
            classifier_input_dim = jumbo_dim * 2  # Two pooled representations
        else:
            # Input is concatenated flattened: (B, J*D) per sequence
            classifier_input_dim = jumbo_dim * 2

        self.classifier = nn.Sequential(
            # Input is concatenation of two representations
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary classification
        )

    def forward(self, jumbo_rep1, jumbo_rep2):
        """
        Args:
            jumbo_rep1: First jumbo representation
                       - If pool_jumbo=False: (B, jumbo_dim) flattened
                       - If pool_jumbo=True: (B, J, D) un-flattened tokens
            jumbo_rep2: Second jumbo representation (same format as jumbo_rep1)

        Returns:
            logits: Binary classification logits (B, 1)
        """
        # Pool if needed
        if self.pool_jumbo:
            # Input: (B, J, D) -> Pool across J dimension -> (B, D)
            if self.pool_type == 'mean':
                rep1 = jumbo_rep1.mean(dim=1)
                rep2 = jumbo_rep2.mean(dim=1)
            elif self.pool_type == 'max':
                rep1 = jumbo_rep1.max(dim=1)[0]
                rep2 = jumbo_rep2.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pool_type: {self.pool_type}")
        else:
            # Input is already flattened: (B, J*D)
            rep1 = jumbo_rep1
            rep2 = jumbo_rep2

        # Concatenate the two representations
        combined = torch.cat([rep1, rep2], dim=-1)  # (B, dim * 2)
        logits = self.classifier(combined)  # (B, 1)
        return logits


def create_taxonomy_pairs(
    taxonomy_labels,
    same_ratio=0.5,
    max_pairs=64,
    debug_print=True,
    bin_labels=None,
    family_labels=None,
):
    """
    Create pairs of indices for taxonomy classification.

    Enumerates all possible positive (same taxon) and negative (different taxon)
    pairs from genus labels, then augments with BIN and family information for
    sequences without genus labels (BIOSCAN-5M only).

    For BIOSCAN-5M with BIN and family data:
    - Positive pairs from genus_index (same genus)
    - Additional positive pairs from dna_bin_index for sequences without genus labels
      (same BIN → same genus)
    - Negative pairs from genus_index (different genus)
    - Additional negative pairs from family_index for sequences without genus labels
      (different family → different genus)

    Args:
        taxonomy_labels: Tensor of genus labels (B,). Negative values are invalid.
        same_ratio: Target ratio of positive (same-taxon) pairs. Default: 0.5.
        max_pairs: Maximum total number of pairs to return. Default: 64.
        debug_print: If True, print debug info.
        bin_labels: Optional tensor of BIN labels (B,). Used to infer additional
                   positive pairs for sequences without genus labels (BIOSCAN-5M).
        family_labels: Optional tensor of family labels (B,). Used to infer additional
                      negative pairs for sequences without genus labels (BIOSCAN-5M).

    Returns:
        idx1: First indices in pairs (num_pairs,) or None if no pairs possible
        idx2: Second indices in pairs (num_pairs,) or None if no pairs possible
        labels: Binary labels - 1 if same taxon, 0 if different (num_pairs,) or None
        num_same: Number of positive pairs
        num_diff: Number of negative pairs
    """
    device = taxonomy_labels.device

    # Filter out invalid taxonomy labels (< 0) - these are sequences without genus labels
    valid_mask = taxonomy_labels >= 0
    valid_indices = torch.where(valid_mask)[0]
    valid_labels = taxonomy_labels[valid_indices]
    valid_batch_size = len(valid_indices)

    # Track sequences without genus labels for BIN/family augmentation
    invalid_mask = ~valid_mask
    invalid_indices = torch.where(invalid_mask)[0]
    invalid_batch_size = len(invalid_indices)

    if debug_print:
        print("\n[Taxonomy Pairs Debug]")
        print(f"  Total sequences in batch: {taxonomy_labels.size(0)}")
        print(f"  Valid genus labels: {valid_batch_size}")
        print(f"  Invalid genus labels (will try BIN/family): {invalid_batch_size}")
        if valid_batch_size > 0:
            print(f"  First 5 valid genus IDs: {valid_labels[:5].tolist()}")

    # === Step 1: Create pairs from genus labels ===
    pos_pairs = torch.empty(0, 2, dtype=torch.long, device=device)
    neg_pairs = torch.empty(0, 2, dtype=torch.long, device=device)

    if valid_batch_size >= 2:
        # Enumerate all possible pairs (i < j) among valid sequences
        all_pairs = torch.combinations(torch.arange(valid_batch_size, device=device), r=2)

        # Split into positive (same taxon) and negative (different taxon) pools
        same_mask = valid_labels[all_pairs[:, 0]] == valid_labels[all_pairs[:, 1]]
        pos_pairs = all_pairs[same_mask]
        neg_pairs = all_pairs[~same_mask]

        if debug_print:
            unique_taxa = torch.unique(valid_labels)
            print(f"  Unique genera (from genus labels): {len(unique_taxa)}")
            print(f"  Genus-based positive pairs: {len(pos_pairs)}, negative pairs: {len(neg_pairs)}")

    # === Step 2: Augment with BIN-based positive pairs (for sequences without genus labels) ===
    bin_pos_pairs = torch.empty(0, 2, dtype=torch.long, device=device)
    if bin_labels is not None and invalid_batch_size >= 2:
        # Get BIN labels for sequences without genus labels
        invalid_bin_labels = bin_labels[invalid_indices]
        valid_bin_mask = invalid_bin_labels >= 0
        valid_bin_indices = invalid_indices[valid_bin_mask]
        valid_bin_labels = invalid_bin_labels[valid_bin_mask]

        if len(valid_bin_indices) >= 2:
            # Create all pairs among sequences with valid BIN labels
            local_indices = torch.arange(len(valid_bin_indices), device=device)
            bin_all_pairs = torch.combinations(local_indices, r=2)

            # Keep only pairs with same BIN (positive pairs)
            bin_same_mask = valid_bin_labels[bin_all_pairs[:, 0]] == valid_bin_labels[bin_all_pairs[:, 1]]
            bin_pos_pairs_local = bin_all_pairs[bin_same_mask]

            # Map back to original batch indices
            bin_pos_pairs = valid_bin_indices[bin_pos_pairs_local]

            if debug_print:
                unique_bins = torch.unique(valid_bin_labels)
                print(f"  Sequences without genus but with valid BIN: {len(valid_bin_indices)}")
                print(f"  Unique BINs (in sequences without genus): {len(unique_bins)}")
                print(f"  BIN-based positive pairs (no genus label): {len(bin_pos_pairs)}")

    # === Step 3: Augment with family-based negative pairs (for sequences without genus labels) ===
    family_neg_pairs = torch.empty(0, 2, dtype=torch.long, device=device)
    if family_labels is not None and invalid_batch_size >= 2:
        # Get family labels for sequences without genus labels
        invalid_family_labels = family_labels[invalid_indices]
        valid_family_mask = invalid_family_labels >= 0
        valid_family_indices = invalid_indices[valid_family_mask]
        valid_family_labels = invalid_family_labels[valid_family_mask]

        if len(valid_family_indices) >= 2:
            # Create all pairs among sequences with valid family labels
            local_indices = torch.arange(len(valid_family_indices), device=device)
            family_all_pairs = torch.combinations(local_indices, r=2)

            # Keep only pairs with different family (negative pairs)
            family_diff_mask = (
                valid_family_labels[family_all_pairs[:, 0]] != valid_family_labels[family_all_pairs[:, 1]]
            )
            family_neg_pairs_local = family_all_pairs[family_diff_mask]

            # Map back to original batch indices
            family_neg_pairs = valid_family_indices[family_neg_pairs_local]

            if debug_print:
                print(f"  Family-based negative pairs (no genus label): {len(family_neg_pairs)}")

    # === Step 4: Combine all pairs ===
    # Map genus-based pairs back to original batch indices (they're currently in valid_indices space)
    if len(pos_pairs) > 0:
        pos_pairs = valid_indices[pos_pairs]
    if len(neg_pairs) > 0:
        neg_pairs = valid_indices[neg_pairs]

    # Combine positive pairs: genus-based + BIN-based
    all_pos_pairs = torch.cat([pos_pairs, bin_pos_pairs], dim=0)
    # Combine negative pairs: genus-based + family-based
    all_neg_pairs = torch.cat([neg_pairs, family_neg_pairs], dim=0)

    if debug_print:
        print(f"  Total positive pairs available: {len(all_pos_pairs)}")
        print(f"  Total negative pairs available: {len(all_neg_pairs)}")

    if len(all_pos_pairs) == 0 and len(all_neg_pairs) == 0:
        if debug_print:
            print("  WARNING: No pairs possible.")
        return None, None, None, 0, 0

    # === Step 5: Sample pairs respecting same_ratio and max_pairs ===
    num_pos_target = int(max_pairs * same_ratio)
    num_neg_target = max_pairs - num_pos_target

    # Cap each pool at its target, then redistribute leftover budget
    num_pos = min(num_pos_target, len(all_pos_pairs))
    num_neg = min(num_neg_target, len(all_neg_pairs))

    if num_pos < num_pos_target:
        # Positive pool exhausted — give remaining budget to negative
        num_neg = min(len(all_neg_pairs), max_pairs - num_pos)
    elif num_neg < num_neg_target:
        # Negative pool exhausted — give remaining budget to positive
        num_pos = min(len(all_pos_pairs), max_pairs - num_neg)

    # Randomly sample from each pool
    if num_pos > 0:
        sampled_pos = all_pos_pairs[torch.randperm(len(all_pos_pairs), device=device)[:num_pos]]
    else:
        sampled_pos = torch.empty(0, 2, dtype=torch.long, device=device)

    if num_neg > 0:
        sampled_neg = all_neg_pairs[torch.randperm(len(all_neg_pairs), device=device)[:num_neg]]
    else:
        sampled_neg = torch.empty(0, 2, dtype=torch.long, device=device)

    # Extract idx1 and idx2
    sampled_pairs = torch.cat([sampled_pos, sampled_neg], dim=0)
    idx1 = sampled_pairs[:, 0]
    idx2 = sampled_pairs[:, 1]
    labels = torch.cat(
        [
            torch.ones(num_pos, dtype=torch.long, device=device),
            torch.zeros(num_neg, dtype=torch.long, device=device),
        ]
    )

    if debug_print:
        print(f"  Sampled {num_pos} positive pairs, {num_neg} negative pairs (total: {num_pos + num_neg})")

    return idx1, idx2, labels, num_pos, num_neg


def compute_taxonomy_classification_loss(
    jumbo_tokens,
    taxonomy_labels,
    classifier,
    same_ratio=0.5,
    max_pairs=32,
    debug_print=False,
    bin_labels=None,
    family_labels=None,
):
    """
    Compute binary taxonomy classification loss for a batch.

    Args:
        jumbo_tokens: Jumbo token representations from encoder (B, J, D)
                     Will be flattened to (B, J*D) if classifier doesn't use pooling
        taxonomy_labels: Taxonomic labels for each sequence at specified level (B,)
        classifier: JumboTaxonomyClassifier instance
        same_ratio: Target ratio of positive (same-taxon) pairs
        max_pairs: Maximum total number of pairs to sample
        debug_print: If True, print debugging information about pair creation
        bin_labels: Optional BIN labels (B,) for augmenting pairs (BIOSCAN-5M)
        family_labels: Optional family labels (B,) for augmenting pairs (BIOSCAN-5M)

    Returns:
        loss: Binary cross-entropy loss (None if no pairs could be created)
        accuracy: Accuracy of predictions (None if no pairs could be created)
        num_pairs: Number of pairs created
        num_same: Number of same-taxonomy pairs
        num_diff: Number of different-taxonomy pairs
    """
    batch_size = jumbo_tokens.size(0)

    # Prepare jumbo representations based on classifier's pooling mode
    if classifier.pool_jumbo:
        # Classifier will pool internally, pass un-flattened tokens (B, J, D)
        jumbo_rep = jumbo_tokens
    else:
        # Classifier expects flattened, so flatten jumbo tokens to (B, J*D)
        jumbo_rep = jumbo_tokens.reshape(batch_size, -1)

    # Create pairs (filtering out invalid genus labels, augmenting with BIN/family if available)
    idx1, idx2, labels, num_same, num_diff = create_taxonomy_pairs(
        taxonomy_labels,
        same_ratio=same_ratio,
        max_pairs=max_pairs,
        debug_print=debug_print,
        bin_labels=bin_labels,
        family_labels=family_labels,
    )

    # Handle case where no pairs could be created
    if idx1 is None:
        return None, None, 0, 0, 0

    # Get representations for pairs
    jumbo_rep1 = jumbo_rep[idx1]
    jumbo_rep2 = jumbo_rep[idx2]

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
