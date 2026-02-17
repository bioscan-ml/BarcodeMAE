"""
Jumbo BERT Transformer with Taxonomy Classification

Wraps JumboBertForTokenClassification and adds a taxonomy classification head
that operates on the jumbo tokens for non-MAELM (single encoder) architectures.
"""

import torch
import torch.nn as nn

from barcodebert.jumbo_taxonomy_classifier import JumboTaxonomyClassifier
from barcodebert.jumbo_transformer import create_jumbo_transformer_model


class JumboTransformerWithTaxonomy(nn.Module):
    """
    Jumbo BERT transformer with taxonomy classification head.

    This is for non-MAELM architectures (standard transformer/BERT pretraining).
    The model uses jumbo CLS tokens for taxonomy classification.
    """

    def __init__(
        self,
        bert_config,
        jumbo_multiplier=6,
        share_jumbo_mlp_across_layers=False,
        enable_taxonomy_classification=True,
        mlp_expansion_factor=2,
    ):
        super().__init__()

        self.jumbo_multiplier = jumbo_multiplier
        self.enable_taxonomy_classification = enable_taxonomy_classification

        # Create the jumbo transformer model
        self.transformer = create_jumbo_transformer_model(
            bert_config,
            jumbo_multiplier=jumbo_multiplier,
            share_jumbo_mlp_across_layers=share_jumbo_mlp_across_layers,
            mlp_expansion_factor=mlp_expansion_factor,
        )

        # Add taxonomy classifier if enabled
        if self.enable_taxonomy_classification:
            jumbo_dim = bert_config.hidden_size * jumbo_multiplier
            self.taxonomy_classifier = JumboTaxonomyClassifier(jumbo_dim=jumbo_dim)
        else:
            self.taxonomy_classifier = None

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, **kwargs
    ):
        """
        Forward pass through jumbo transformer.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            token_type_ids: Token type IDs (B, seq_len)
            position_ids: Position IDs (B, seq_len)
            inputs_embeds: Input embeddings (B, seq_len, hidden_size)

        Returns:
            outputs object with:
                - logits: Token classification logits (B, seq_len, num_labels)
                - hidden_states: Final hidden states (B, seq_len, hidden_size)
                - jumbo_representation: Flattened jumbo representation (B, J*D)
                - jumbo_tokens: Jumbo tokens (B, J, D)
        """
        # Forward through jumbo transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Attach jumbo_tokens to outputs for consistency with MAELM interface
        # Note: jumbo_tokens are already available in outputs.jumbo_tokens from the transformer

        return outputs

    @property
    def config(self):
        """Return the config from the underlying transformer."""
        return self.transformer.config

    @property
    def bert(self):
        """Return the BERT model from the underlying transformer for compatibility."""
        return self.transformer.bert

    @property
    def genus_classifier(self):
        """Backward compatibility alias for taxonomy_classifier."""
        return self.taxonomy_classifier


def create_jumbo_transformer_with_taxonomy(
    bert_config,
    jumbo_multiplier=6,
    share_jumbo_mlp_across_layers=False,
    enable_taxonomy_classification=True,
    mlp_expansion_factor=2,
):
    """
    Factory function to create a jumbo transformer with taxonomy classification.

    Args:
        bert_config: BertConfig for the model
        jumbo_multiplier: Number of jumbo CLS tokens (default: 6)
        share_jumbo_mlp_across_layers: Share jumbo MLP across layers (default: False)
        enable_taxonomy_classification: Add taxonomy classifier (default: True)
        mlp_expansion_factor: Hidden layer expansion ratio in Jumbo MLP (default: 2)

    Returns:
        JumboTransformerWithTaxonomy model
    """
    return JumboTransformerWithTaxonomy(
        bert_config,
        jumbo_multiplier=jumbo_multiplier,
        share_jumbo_mlp_across_layers=share_jumbo_mlp_across_layers,
        enable_taxonomy_classification=enable_taxonomy_classification,
        mlp_expansion_factor=mlp_expansion_factor,
    )


if __name__ == "__main__":
    from transformers import BertConfig

    # Test the model
    print("Testing JumboTransformerWithTaxonomy...")

    config = BertConfig(
        vocab_size=256,
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
        intermediate_size=384 * 4,
        max_position_embeddings=512,
        num_labels=256,
    )

    model = create_jumbo_transformer_with_taxonomy(config, jumbo_multiplier=6, enable_taxonomy_classification=True)

    # Test input
    batch_size = 4
    seq_len = 100
    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    print("\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    print("\nOutput shapes:")
    print(f"  logits: {outputs.logits.shape}")
    print(f"  hidden_states: {outputs.hidden_states.shape}")
    print(f"  jumbo_representation: {outputs.jumbo_representation.shape}")
    print(f"  jumbo_tokens: {outputs.jumbo_tokens.shape}")

    # Test taxonomy classifier
    if model.taxonomy_classifier is not None:
        print("\nTesting taxonomy classifier...")
        jumbo_flat = outputs.jumbo_representation

        # Create pairs
        jumbo_rep1 = jumbo_flat[0::2]
        jumbo_rep2 = jumbo_flat[1::2]
        num_pairs = min(jumbo_rep1.shape[0], jumbo_rep2.shape[0])
        jumbo_rep1 = jumbo_rep1[:num_pairs]
        jumbo_rep2 = jumbo_rep2[:num_pairs]

        with torch.no_grad():
            taxonomy_logits = model.taxonomy_classifier(jumbo_rep1, jumbo_rep2)

        print(f"  Taxonomy logits shape: {taxonomy_logits.shape}")
        print(f"  Expected: ({num_pairs}, 1)")

    print("\n✓ Test passed!")
