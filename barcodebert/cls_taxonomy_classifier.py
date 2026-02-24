"""
Binary taxonomy classification head for CLS tokens.

This module adds a binary classification task during pretraining to predict
whether pairs of sequences share the same taxonomic label at a specified level
(e.g., phylum, class, order, family, genus, or species) using the CLS token
representation.

Unlike jumbo tokens which are processed through a wide MLP, CLS tokens are
processed through the standard transformer FFN layers along with other tokens.
"""

import torch
import torch.nn as nn


class CLSTaxonomyClassifier(nn.Module):
    """
    Binary classifier that takes two CLS token representations and predicts
    if they share the same taxonomic label at a specified level.
    """

    def __init__(self, hidden_dim, classifier_hidden_dim=256, dropout=0.1):
        """
        Args:
            hidden_dim: Dimension of CLS token representation (D)
            classifier_hidden_dim: Hidden layer dimension for the classifier
            dropout: Dropout probability
        """
        super().__init__()

        self.classifier = nn.Sequential(
            # Input is concatenation of two CLS representations
            nn.Linear(hidden_dim * 2, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),  # Binary classification
        )

    def forward(self, cls_rep1, cls_rep2):
        """
        Args:
            cls_rep1: First CLS representation (B, hidden_dim)
            cls_rep2: Second CLS representation (B, hidden_dim)

        Returns:
            logits: Binary classification logits (B, 1)
        """
        # Concatenate the two representations
        combined = torch.cat([cls_rep1, cls_rep2], dim=-1)  # (B, hidden_dim * 2)
        logits = self.classifier(combined)  # (B, 1)
        return logits


def compute_cls_taxonomy_classification_loss(
    hidden_states,
    taxonomy_labels,
    classifier,
    same_ratio=0.5,
    max_pairs=32,
    debug_print=False,
    bin_labels=None,
    family_labels=None,
):
    """
    Compute binary taxonomy classification loss for a batch using CLS token.

    Args:
        hidden_states: Hidden states from transformer (B, seq_len, D)
                      CLS token is assumed to be at position 0
        taxonomy_labels: Taxonomic labels for each sequence at specified level (B,)
        classifier: CLSTaxonomyClassifier instance
        same_ratio: Ratio of same-taxonomy pairs
        max_pairs: Maximum number of pairs to sample
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
    # batch_size = hidden_states.size(0)

    # Extract CLS token representation (position 0)
    cls_tokens = hidden_states[:, 0, :]  # (B, D)

    # Import create_taxonomy_pairs from jumbo_taxonomy_classifier
    from barcodebert.jumbo_taxonomy_classifier import create_taxonomy_pairs

    # Create pairs (filtering out invalid taxonomy labels, augmenting with BIN/family if available)
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

    # Get CLS representations for pairs
    cls_rep1 = cls_tokens[idx1]
    cls_rep2 = cls_tokens[idx2]

    # Compute logits
    logits = classifier(cls_rep1, cls_rep2).squeeze(-1)  # (num_pairs,)

    # Compute loss
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

    # Compute accuracy
    predictions = (torch.sigmoid(logits) > 0.5).long()
    accuracy = (predictions == labels).float().mean()

    num_pairs = len(labels)

    return loss, accuracy, num_pairs, num_same, num_diff
