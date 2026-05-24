"""
Auxiliary taxonomy-aware losses for pretraining.

Three losses, each a standalone function with a unified signature:
    loss, metrics = <loss_fn>(embeddings, labels, ...)

Embeddings are obtained via representations_from_df() from barcodebert.datasets,
which already handles all model/token configurations (CLS, CLS+registers, jumbo,
tokens, etc.) through its representation_type argument.

All loss functions:
  - Accept (N, D) float tensors — convert numpy output of representations_from_df
    with torch.from_numpy(X).float() before passing in.
  - Skip sequences with label < 0 (unlabelled in BIOSCAN-5M).
  - Return (None, None) when there is not enough labelled data.
  - Work on any taxonomic level — just pass the right label column.

Usage sketch
------------
from barcodebert.datasets import representations_from_df
from barcodebert.taxonomy_aux_losses import (
    triplet_loss_batch_hard,
    supcon_loss,
    TaxonomyClassificationHead,
    crossentropy_taxonomy_loss,
)

# Get embeddings — works for transformer, maelm, jumbo, CLS+registers, etc.
# Just set representation_type to match how your model was trained.
X, y, _ = representations_from_df(
    df, "genus_index", model, tokenizer, "BIOSCAN-5M",
    representation_type="cls",       # or "tokens", "jumbo_avg", "tokens_with_registers", …
    use_cls_token=True,
)

emb    = torch.from_numpy(X).float()
labels = torch.from_numpy(y).long()

loss_t, m_t = triplet_loss_batch_hard(emb, labels, margin=0.3)
loss_s, m_s = supcon_loss(emb, labels, temperature=0.07)

# Cross-entropy needs a head built once with the right num_classes:
head   = TaxonomyClassificationHead(hidden_dim=emb.shape[1], num_classes=n_genera)
loss_c, m_c = crossentropy_taxonomy_loss(emb, labels, head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from barcodebert.datasets import representations_from_df  # noqa: F401  — re-exported for callers


# ──────────────────────────────────────────────────────────────────────────────
# 1. Triplet loss — batch-hard mining
# ──────────────────────────────────────────────────────────────────────────────

def triplet_loss_batch_hard(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    distance: str = "cosine",
) -> tuple:
    """
    Batch-hard triplet loss (Hermans et al., 2017).

    For every valid anchor:
      - hardest positive  = same-label sample with the LARGEST distance
      - hardest negative  = different-label sample with the SMALLEST distance

    Parameters
    ----------
    embeddings : (B, D)
    labels     : (B,)  integer labels; values < 0 are treated as unlabelled and skipped
    margin     : additive margin (soft-margin variant uses softplus instead)
    distance   : "cosine" or "euclidean"

    Returns
    -------
    loss    : scalar tensor, or None if fewer than 2 valid classes in the batch
    metrics : dict with keys "loss", "frac_active" (triplets with loss > 0),
              "mean_pos_dist", "mean_neg_dist"  — or None
    """
    valid = labels >= 0
    if valid.sum() < 2:
        return None, None

    emb = embeddings[valid]
    lab = labels[valid]
    n = emb.size(0)

    unique_classes = lab.unique()
    if unique_classes.numel() < 2:
        return None, None

    # Pairwise distance matrix (N, N)
    if distance == "cosine":
        emb_norm = F.normalize(emb, dim=1)
        dist = 1.0 - emb_norm @ emb_norm.t()          # cosine distance ∈ [0, 2]
    else:
        # Euclidean: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        sq = (emb ** 2).sum(1, keepdim=True)
        dist = (sq + sq.t() - 2.0 * emb @ emb.t()).clamp(min=0).sqrt()

    # Boolean masks: same_mask[i, j] = True  ↔  lab[i] == lab[j]  (i ≠ j)
    same_mask = (lab.unsqueeze(0) == lab.unsqueeze(1))
    diff_mask = ~same_mask
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    same_mask = same_mask & ~eye

    # Keep only anchors that have at least one positive AND one negative
    has_pos = same_mask.any(dim=1)
    has_neg = diff_mask.any(dim=1)
    valid_anchor = has_pos & has_neg
    if valid_anchor.sum() == 0:
        return None, None

    # Hardest positive: max distance among same-label pairs
    # Replace invalid positions with -inf so max ignores them
    pos_dist_mat = dist.masked_fill(~same_mask, float("-inf"))
    hardest_pos_dist = pos_dist_mat[valid_anchor].max(dim=1).values   # (A,)

    # Hardest negative: min distance among different-label pairs
    # Replace invalid positions with +inf so min ignores them
    neg_dist_mat = dist.masked_fill(~diff_mask, float("inf"))
    hardest_neg_dist = neg_dist_mat[valid_anchor].min(dim=1).values   # (A,)

    # Triplet loss with soft margin (smoother gradients than hard hinge)
    raw = hardest_pos_dist - hardest_neg_dist + margin
    loss = F.softplus(raw).mean()

    with torch.no_grad():
        frac_active = (raw > 0).float().mean().item()
        metrics = {
            "loss": loss.item(),
            "frac_active": frac_active,
            "mean_pos_dist": hardest_pos_dist.mean().item(),
            "mean_neg_dist": hardest_neg_dist.mean().item(),
        }

    return loss, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 2. Supervised Contrastive loss (SupCon)
# ──────────────────────────────────────────────────────────────────────────────

def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> tuple:
    """
    Supervised Contrastive loss (Khosla et al., 2020).

    For each anchor all other same-label samples are positives; everything
    else is a negative. Embeddings are L2-normalised internally.

    Parameters
    ----------
    embeddings  : (B, D)
    labels      : (B,)  values < 0 are skipped
    temperature : τ — lower = sharper distribution

    Returns
    -------
    loss    : scalar tensor, or None if fewer than 2 valid classes
    metrics : dict with keys "loss", "mean_n_positives"  — or None
    """
    valid = labels >= 0
    if valid.sum() < 2:
        return None, None

    emb = F.normalize(embeddings[valid], dim=1)   # (N, D)
    lab = labels[valid]
    n = emb.size(0)

    unique_classes = lab.unique()
    if unique_classes.numel() < 2:
        return None, None

    # Similarity matrix scaled by temperature
    sim = emb @ emb.t() / temperature             # (N, N)

    # Mask out self-comparisons from both numerator and denominator
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(eye, float("-inf"))

    # For numerical stability: subtract row-wise max before exp
    sim_max, _ = sim.clone().masked_fill(eye, float("-inf")).max(dim=1, keepdim=True)
    exp_sim = torch.exp(sim - sim_max.detach())   # (N, N)
    exp_sim = exp_sim.masked_fill(eye, 0.0)

    # Denominator: sum over all non-self pairs
    denom = exp_sim.sum(dim=1, keepdim=True)      # (N, 1)

    # Positive mask: same label, not self
    pos_mask = (lab.unsqueeze(0) == lab.unsqueeze(1)) & ~eye  # (N, N)
    n_positives = pos_mask.sum(dim=1).float()                  # (N,)

    # Anchors that have at least one positive in the batch
    has_pos = n_positives > 0
    if has_pos.sum() == 0:
        return None, None

    # Log-probability of each positive pair, averaged per anchor
    log_prob = sim - sim_max - torch.log(denom + 1e-8)  # (N, N) — subtract sim_max to invert stability trick
    log_prob = log_prob.masked_fill(eye, 0.0)            # avoid 0 * (-inf) = nan at diagonal
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / n_positives.clamp(min=1)

    loss = -mean_log_prob_pos[has_pos].mean()

    with torch.no_grad():
        metrics = {
            "loss": loss.item(),
            "mean_n_positives": n_positives[has_pos].mean().item(),
        }

    return loss, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cross-entropy over taxonomy labels
# ──────────────────────────────────────────────────────────────────────────────

class TaxonomyClassificationHead(nn.Module):
    """
    Linear classification head for direct taxonomic label prediction.

    Instantiate once per taxonomic level you want to predict:
        genus_head  = TaxonomyClassificationHead(768, n_genera)
        family_head = TaxonomyClassificationHead(768, n_families)

    The head is separate from the encoder — attach its parameters to the
    optimizer alongside the encoder's parameters.
    """

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def crossentropy_taxonomy_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    classifier: TaxonomyClassificationHead,
) -> tuple:
    """
    Direct cross-entropy loss over any taxonomic level.

    The taxonomic level is determined entirely by which label tensor you pass
    in (genus_index, family_index, order_index, …).  Sequences with label < 0
    are skipped.

    Parameters
    ----------
    embeddings : (B, D)  — typically the CLS token
    labels     : (B,)    — integer class indices; < 0 means unlabelled
    classifier : TaxonomyClassificationHead instance

    Returns
    -------
    loss    : scalar tensor, or None if no valid labels in the batch
    metrics : dict with keys "loss", "accuracy", "n_valid"  — or None
    """
    valid = labels >= 0
    n_valid = valid.sum().item()
    if n_valid == 0:
        return None, None

    emb_valid = embeddings[valid]
    lab_valid = labels[valid]

    logits = classifier(emb_valid)                # (N, num_classes)
    loss = F.cross_entropy(logits, lab_valid)

    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == lab_valid).float().mean().item()
        metrics = {
            "loss": loss.item(),
            "accuracy": acc,
            "n_valid": n_valid,
        }

    return loss, metrics