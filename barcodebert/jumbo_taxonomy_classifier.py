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


def create_taxonomy_pairs(taxonomy_labels, same_ratio=0.5):
    """
    Create pairs of indices and their labels (same taxonomic group or not).

    Handles edge cases:
    - When batch has only one taxonomic group (all samples are same)
    - When some groups have only single samples
    - Returns None values if pairing is impossible

    Args:
        taxonomy_labels: Tensor of taxonomic labels at specified level (B,)
        same_ratio: Ratio of same-taxonomy pairs to create

    Returns:
        idx1: First indices in pairs (num_pairs,) or None if no pairs possible
        idx2: Second indices in pairs (num_pairs,) or None if no pairs possible
        labels: Binary labels (num_pairs,) - 1 if same taxon, 0 if different, or None
        num_same: Actual number of same-taxonomy pairs created
        num_diff: Actual number of different-taxonomy pairs created
    """
    batch_size = taxonomy_labels.size(0)
    device = taxonomy_labels.device

    # Edge case: batch size < 2
    if batch_size < 2:
        return None, None, None, 0, 0

    # Count samples per taxonomic group
    unique_taxa, counts = torch.unique(taxonomy_labels, return_counts=True)
    num_taxa = len(unique_taxa)

    # Edge case: only one taxonomic group in batch (all same)
    if num_taxa == 1:
        # Can only create same-taxonomy pairs
        if counts[0] < 2:
            # Single sample, can't make any pairs
            return None, None, None, 0, 0
        # Make all pairs from same taxonomic group
        num_pairs = batch_size // 2
        idx1 = torch.arange(0, num_pairs * 2, 2, device=device)
        idx2 = torch.arange(1, num_pairs * 2, 2, device=device)
        labels = torch.ones(num_pairs, dtype=torch.long, device=device)
        return idx1, idx2, labels, num_pairs, 0

    # Calculate how many pairs to create
    num_same_target = int(batch_size * same_ratio)
    num_diff_target = batch_size - num_same_target

    idx1_list = []
    idx2_list = []
    labels_list = []

    # Create same-taxonomy pairs
    # Identify which taxonomic groups have 2+ samples
    taxa_with_pairs = unique_taxa[counts >= 2]
    num_same_actual = 0

    if len(taxa_with_pairs) > 0:
        for _ in range(num_same_target * 2):  # Try more times to hit target
            if num_same_actual >= num_same_target:
                break
            # Sample a taxonomic group that has pairs
            taxon = taxa_with_pairs[torch.randint(len(taxa_with_pairs), (1,))].item()
            same_taxon_indices = torch.where(taxonomy_labels == taxon)[0]

            # Sample two different indices from same taxonomic group
            perm = torch.randperm(len(same_taxon_indices))[:2]
            idx1_list.append(same_taxon_indices[perm[0]])
            idx2_list.append(same_taxon_indices[perm[1]])
            labels_list.append(1)
            num_same_actual += 1

    # Create different-taxonomy pairs
    num_diff_actual = 0
    max_attempts = num_diff_target * 3  # Allow multiple attempts

    for _ in range(max_attempts):
        if num_diff_actual >= num_diff_target:
            break
        # Sample two random indices
        idx1 = torch.randint(batch_size, (1,), device=device).item()
        idx2 = torch.randint(batch_size, (1,), device=device).item()

        # Only add if they're from different taxonomic groups and not same index
        if idx1 != idx2 and taxonomy_labels[idx1] != taxonomy_labels[idx2]:
            idx1_list.append(torch.tensor(idx1, device=device))
            idx2_list.append(torch.tensor(idx2, device=device))
            labels_list.append(0)
            num_diff_actual += 1

    # If we couldn't create any pairs, return None
    if len(idx1_list) == 0:
        return None, None, None, 0, 0

    idx1 = torch.stack(idx1_list).to(device)
    idx2 = torch.stack(idx2_list).to(device)
    labels = torch.tensor(labels_list, dtype=torch.long, device=device)

    return idx1, idx2, labels, num_same_actual, num_diff_actual


def compute_taxonomy_classification_loss(jumbo_tokens, taxonomy_labels, classifier, same_ratio=0.5):
    """
    Compute binary taxonomy classification loss for a batch.

    Args:
        jumbo_tokens: Jumbo token representations from encoder (B, J, D)
        taxonomy_labels: Taxonomic labels for each sequence at specified level (B,)
        classifier: JumboTaxonomyClassifier instance
        same_ratio: Ratio of same-taxonomy pairs

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

    # Create pairs
    idx1, idx2, labels, num_same, num_diff = create_taxonomy_pairs(taxonomy_labels, same_ratio=same_ratio)

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
