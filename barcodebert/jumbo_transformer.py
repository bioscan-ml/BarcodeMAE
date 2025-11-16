"""
Jumbo BERT for Token Classification (DNA Transformer)

Simple implementation of Jumbo CLS tokens for a BERT-style transformer.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertLayer


class JumboTokenHandler(nn.Module):
    """
    Handles Jumbo CLS token operations.

    Takes J CLS tokens of dim D: (B, J, D)
    Flattens to (B, J*D), applies a wide MLP, then reshapes back to (B, J, D)
    with a residual connection.
    """

    def __init__(self, embed_dim: int, jumbo_multiplier: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim               # D
        self.jumbo_multiplier = jumbo_multiplier # J
        self.jumbo_width = embed_dim * jumbo_multiplier  # J * D

        self.jumbo_mlp = nn.Sequential(
            nn.LayerNorm(self.jumbo_width),
            nn.Linear(self.jumbo_width, self.jumbo_width * 2), # Wide hidden layer X4
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.jumbo_width * 2, self.jumbo_width),
            nn.Dropout(dropout),
        )

    def apply_jumbo_mlp(self, jumbo_tokens: torch.Tensor) -> torch.Tensor:
        """
        jumbo_tokens: (B, J, D)
        returns: (B, J, D) after wide MLP + residual
        """
        bsz, actual_J, dim = jumbo_tokens.shape
        expected_width = actual_J * dim

        if expected_width != self.jumbo_width:
            raise ValueError(
                f"JumboTokenHandler: width mismatch. "
                f"Expected {self.jumbo_width} (J*D), got {expected_width}."
            )

        # (B, J, D) -> (B, J*D)
        flat = jumbo_tokens.reshape(bsz, expected_width)

        # wide MLP
        mlp_out = self.jumbo_mlp(flat)
        flat_with_residual = flat + mlp_out  # residual in flat space

        # (B, J*D) -> (B, J, D)
        reshaped = flat_with_residual.reshape(bsz, actual_J, dim)

        #Jumbo tokens
        return reshaped


class JumboBertLayer(nn.Module):
    """
    BERT layer modified to follow Jumbo pattern:

    1. Run standard self-attention over ALL tokens (Jumbo + patches).
    2. Split output into:
       - Jumbo tokens (first J)
       - Patch tokens (rest)
    3. Apply Jumbo MLP only to Jumbo tokens.
    4. Apply standard FFN only to patch tokens.
    5. Concatenate back Jumbo + patch tokens.
    """

    def __init__(self, original_bert_layer: BertLayer, jumbo_handler: JumboTokenHandler):
        super().__init__()
        # Keep original components
        self.attention = original_bert_layer.attention
        self.intermediate = original_bert_layer.intermediate
        self.output = original_bert_layer.output

        self.jumbo_handler = jumbo_handler
        self.j = jumbo_handler.jumbo_multiplier

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        hidden_states: (B, J + N, D)
        attention_mask: extended additive mask, shape (B, 1, 1, J+N)
        returns: (B, J + N, D)
        """
        # 1. Self-attention (BERT-style)
        #    attention(...) returns (attn_output, maybe_attentions)
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=False,
        )
        attn_output = attn_outputs[0]  # (B, J+N, D); already includes residual + LayerNorm inside BERT

        # 2. Split into jumbo and patch tokens
        jumbo_tokens = attn_output[:, :self.j, :]   # (B, J, D)
        patch_tokens = attn_output[:, self.j:, :]   # (B, N, D)

        # 3. Jumbo path
        jumbo_out = self.jumbo_handler.apply_jumbo_mlp(jumbo_tokens)  # (B, J, D)

        # 4. Patch path: standard FFN
        patch_intermediate = self.intermediate(patch_tokens)
        patch_out = self.output(patch_intermediate, patch_tokens)     # (B, N, D) includes residual + LN

        # 5. Concatenate
        final_output = torch.cat([jumbo_out, patch_out], dim=1)       # (B, J+N, D)
        return final_output


class JumboBertForTokenClassification(nn.Module):
    """
    Jumbo version of BERT for token-level classification on DNA sequences.

    - Adds J Jumbo CLS tokens at the beginning of the sequence.
    - All layers share one JumboTokenHandler.
    - Patch tokens (original sequence positions) get per-token logits.
    - Final Jumbo representation is returned as a flattened vector (B, J*D).
    """

    def __init__(self, config: BertConfig, jumbo_multiplier: int = 6, share_jumbo_mlp_across_layers: bool = False):
        super().__init__()
        self.config = config
        self.jumbo_multiplier = jumbo_multiplier
        self.num_labels = config.num_labels

        # Base BERT model (we'll only use its embeddings + encoder blocks)
        self.bert = BertModel(config, add_pooling_layer=False)

        if share_jumbo_mlp_across_layers:
            # Single shared handler
            shared_jumbo_handler = JumboTokenHandler(
                embed_dim=config.hidden_size,
                jumbo_multiplier=jumbo_multiplier,
                dropout=config.hidden_dropout_prob,
            )
            jumbo_handlers = [shared_jumbo_handler] * len(self.bert.encoder.layer)
        else:
            # Separate handler per layer
            print("Using separate JumboTokenHandler per layer.")
            jumbo_handlers = [
                JumboTokenHandler(config.hidden_size, jumbo_multiplier, config.hidden_dropout_prob)
                for _ in range(len(self.bert.encoder.layer))
            ]

        # Shared Jumbo handler for ALL layers


        # J separate Jumbo CLS tokens, each of dim D
        self.jumbo_cls_tokens = nn.Parameter(
            torch.zeros(1, jumbo_multiplier, config.hidden_size)
        )

        # Replace encoder layers with JumboBertLayer that use the shared handler
        new_layers = []
        for original_layer, jumbo_handler in zip(self.bert.encoder.layer, jumbo_handlers):
            new_layers.append(JumboBertLayer(original_layer, jumbo_handler))
        self.bert.encoder.layer = nn.ModuleList(new_layers)

        # Token classification head (applied only to patch tokens)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Final norm for Jumbo representation (flattened J*D)
        self.jumbo_norm = nn.LayerNorm(config.hidden_size * jumbo_multiplier)

        # Init Jumbo CLS tokens
        nn.init.trunc_normal_(self.jumbo_cls_tokens, mean=0.0, std=0.02)
        self.device = None

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        # 1. Standard BERT embeddings for the original sequence (no Jumbo yet)
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )  # (B, N, D)

        batch_size, seq_len, hidden_size = embedding_output.size()

        device = embedding_output.device
        self.device = device
        # 2. Prepend J Jumbo CLS tokens
        jumbo_cls = self.jumbo_cls_tokens.expand(batch_size, -1, -1)  # (B, J, D)
        jumbo_cls = jumbo_cls.to(device)
        hidden_states = torch.cat([jumbo_cls, embedding_output], dim=1)  # (B, J+N, D)

        # 3. Build attention mask to match new length (J + N)
        if attention_mask is None:
            base_mask = torch.ones(batch_size, seq_len, device=device)
        else:
            base_mask = attention_mask.to(device)

        jumbo_mask = torch.ones(
            batch_size,
            self.jumbo_multiplier,
            device=device,
            dtype=base_mask.dtype,
        )
        full_mask = torch.cat([jumbo_mask, base_mask], dim=1)  # (B, J+N)

        extended_attention_mask = self.bert.get_extended_attention_mask(
            full_mask,
            (batch_size, full_mask.size(1)),
            device=device,
        )

        # 4. Encoder
        for layer in self.bert.encoder.layer:
            hidden_states = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
            )

        # 5. Split final output back into jumbo + patch tokens
        final_jumbo_tokens = hidden_states[:, :self.jumbo_multiplier, :]  # (B, J, D)
        final_patch_tokens = hidden_states[:, self.jumbo_multiplier:, :]  # (B, N, D)

        # 6. Token classification on patch tokens
        patch_outputs = self.dropout(final_patch_tokens)
        logits = self.classifier(patch_outputs)  # (B, N, num_labels)

        # 7. Jumbo representation: flattened + LayerNorm
        flattened_jumbo = final_jumbo_tokens.reshape(batch_size, -1)  # (B, J*D)
        jumbo_representation = self.jumbo_norm(flattened_jumbo)       # (B, J*D)

        return type("JumboTokenClassificationOutput", (), {
            "logits": logits,
            "hidden_states": final_patch_tokens,
            "jumbo_representation": jumbo_representation,
        })()


def create_jumbo_transformer_model(original_config: BertConfig, jumbo_multiplier: int = 6, share_jumbo_mlp_across_layers: bool = False):
    """
    Factory function to create Jumbo transformer model.
    """
    return JumboBertForTokenClassification(original_config, jumbo_multiplier, share_jumbo_mlp_across_layers)


# Quick test you can use this to check the implementation
# if __name__ == "__main__":
#     config = BertConfig(
#         vocab_size=1000,
#         hidden_size=768,
#         num_hidden_layers=6,
#         num_attention_heads=12,
#         intermediate_size=3072,
#         max_position_embeddings=512,
#         num_labels=256,  # DNA token labels
#     )
#
#     model = create_jumbo_transformer_model(config, jumbo_multiplier=6)
#     print(model)
#     batch_size = 2
#     seq_len = 10
#     input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
#     attention_mask = torch.ones(batch_size, seq_len)
#
#     out = model(input_ids=input_ids, attention_mask=attention_mask)
#
#     print("Input shape:", input_ids.shape)                     # (2, 10)
#     print("Logits shape:", out.logits.shape)                   # (2, 10, 256)
#     print("Hidden states shape:", out.hidden_states.shape)     # (2, 10, 768)
#     print("Jumbo rep shape:", out.jumbo_representation.shape)  # (2, 6*768) = (2, 4608)
#     print("Jumbo BERT test passed!")
