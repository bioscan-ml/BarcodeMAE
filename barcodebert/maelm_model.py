import torch
import torch.nn as nn
from transformers import BertForTokenClassification, BertModel

from barcodebert.jumbo_taxonomy_classifier import JumboTaxonomyClassifier
from barcodebert.jumbo_transformer import create_jumbo_transformer_model


class MAELMModel(nn.Module):

    def __init__(
        self,
        encoder_config,
        decoder_config,
        jumbo,
        jumbo_multiplier,
        share_jumbo_layers,
        enable_genus_classification=False,
        jumbo_source="encoder",
    ):
        super(MAELMModel, self).__init__()
        # Encoder BERT model
        # model = BertForMaskedLM(encoder_config)
        self.jumbo = jumbo
        self.jumbo_multiplier = jumbo_multiplier
        self.share_jumbo_layers = share_jumbo_layers
        self.enable_genus_classification = enable_genus_classification
        self.jumbo_source = jumbo_source  # "encoder" or "decoder"

        # Validate jumbo_source configuration
        if self.jumbo_source not in ["encoder", "decoder"]:
            raise ValueError(f"jumbo_source must be 'encoder' or 'decoder', got '{self.jumbo_source}'")

        if self.enable_genus_classification and not self.jumbo:
            raise ValueError("Taxonomy classification requires jumbo=True")

        if self.jumbo:
            self.encoder = create_jumbo_transformer_model(
                encoder_config, jumbo_multiplier=jumbo_multiplier, share_jumbo_mlp_across_layers=share_jumbo_layers
            )
        else:
            self.encoder = BertModel(encoder_config)

        # Decoder BERT model with token classification head (always standard, never jumbo)
        self.decoder = BertForTokenClassification(decoder_config)

        # Encoder Embeddings (word and positional)
        # self.encoder_embedding = self.encoder.embeddings.word_embeddings
        # self.encoder_position_embeddings = self.encoder.embeddings.position_embeddings

        # Projection layer to map encoder hidden states to decoder input
        self.projection_layer = nn.Linear(encoder_config.hidden_size, decoder_config.hidden_size)

        # Decoder Embeddings (word and positional)
        self.decoder_embedding = self.decoder.bert.embeddings.word_embeddings

        # Taxonomy classification head (only if jumbo is enabled and taxonomy classification is requested)
        if self.enable_genus_classification:
            # Determine the hidden size based on jumbo source
            # Both use the same dimension since decoder processes encoder's jumbo tokens
            if self.jumbo_source == "encoder":
                jumbo_dim = encoder_config.hidden_size * jumbo_multiplier
            else:  # decoder - uses decoder's hidden size since jumbo tokens were projected
                jumbo_dim = decoder_config.hidden_size * jumbo_multiplier

            self.taxonomy_classifier = JumboTaxonomyClassifier(jumbo_dim=jumbo_dim)
        else:
            self.taxonomy_classifier = None

    def forward(self, input_ids, attention_mask, mask_positions, model_type="maelm_v2"):
        if model_type == "maelm_v2":
            return self.forward_maelm(input_ids, attention_mask, mask_positions)
        elif model_type == "baseline":
            return self.forward_baseline(input_ids, attention_mask)

    def forward_maelm(self, input_ids, attention_mask, mask_positions):
        """
        This version is removing the masked token from the encoder input by padding the sequence with the UNK token.
        The positional ids are based on the input sequence.

        """
        batch_size, seq_len = input_ids.size()

        # Create mask for unmasked positions (True where tokens are unmasked)
        seen_token_positions = ~mask_positions  # Shape: [batch_size, seq_len]
        # Get the number of unmasked tokens per sequence
        seen_lengths = seen_token_positions.sum(dim=1)  # Shape: [batch_size]
        # Get the maximum number of unmasked tokens in the batch
        max_seen_len = seen_lengths.max()

        # Create position IDs for the input sequences
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        # the sequences are padded by 1 which is the [UNK] token
        padded_encoder_input_ids = torch.ones(batch_size, max_seen_len, device=input_ids.device).long()
        # the position id is not reset / this can be implemented in various ways
        padded_encoder_position_ids = torch.zeros(batch_size, max_seen_len, device=input_ids.device).long()
        # the attention mask of the padded tokens should be zero
        padded_encoder_attention_mask = torch.zeros(batch_size, max_seen_len, device=input_ids.device).int()

        # this part is compeleting the padded input_ids and attention mask and position ids without loop on the batches
        indices = torch.arange(max_seen_len, device=input_ids.device).unsqueeze(0).expand(batch_size, max_seen_len)
        seen_indices = indices < seen_lengths.unsqueeze(1)
        # print(seen_indices)
        padded_encoder_input_ids[seen_indices] = input_ids[seen_token_positions]
        padded_encoder_position_ids[seen_indices] = position_ids[seen_token_positions]
        padded_encoder_attention_mask[seen_indices] = attention_mask[seen_token_positions].int()
        # The position id of the pad tokens set to be 0
        padded_encoder_position_ids[~seen_indices] = 0

        # Pass the encoder inputs through the encoder model
        encoder_outputs = self.encoder(
            input_ids=padded_encoder_input_ids,
            attention_mask=padded_encoder_attention_mask,
            position_ids=padded_encoder_position_ids,
        )

        # Handle Jumbo vs standard output
        if self.jumbo:
            encoder_sequence_output = encoder_outputs.hidden_states
            jumbo_tokens = encoder_outputs.jumbo_tokens  # (B, J, D)
        else:
            encoder_sequence_output = encoder_outputs.last_hidden_state
            jumbo_tokens = None

        # If the encoder and decoder have different hidden states, project the encoder hidden states
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size:
            encoder_sequence_output = self.projection_layer(encoder_sequence_output)
            if jumbo_tokens is not None:
                jumbo_tokens = self.projection_layer(jumbo_tokens)

        # Map encoder outputs back to the original sequence positions
        # Use decoder hidden size since we may have projected encoder outputs
        decoder_input_embeddings = torch.zeros(
            batch_size,
            seq_len,
            self.decoder.config.hidden_size,
            device=input_ids.device,
            dtype=encoder_sequence_output.dtype,
        )

        decoder_input_embeddings[seen_token_positions] = encoder_sequence_output[seen_indices].to(
            decoder_input_embeddings.dtype
        )

        # this should not be hard coded
        mask_token_id = 0
        # here I am using the embedding of mask tokens from the decoder embeddings not the encoder
        decoder_input_embeddings[mask_positions] = self.decoder_embedding.weight[mask_token_id]

        # The attention mask of decoder is the same as the input
        decoder_attention_mask = attention_mask
        # The positions ids of the decoder is the same as the input
        decoder_position_ids = position_ids

        # Prepend Jumbo tokens to decoder input (if encoder has jumbo tokens)
        if jumbo_tokens is not None:
            # print(f"jumbo_tokens shape: {jumbo_tokens.shape}")
            # print(f"decoder_input_embeddings before cat: {decoder_input_embeddings.shape}")

            decoder_input_embeddings = torch.cat([jumbo_tokens, decoder_input_embeddings], dim=1)
            jumbo_mask = torch.ones(
                batch_size, self.jumbo_multiplier, device=input_ids.device, dtype=attention_mask.dtype
            )
            decoder_attention_mask = torch.cat([jumbo_mask, decoder_attention_mask], dim=1)
            jumbo_pos = torch.zeros(
                batch_size, self.jumbo_multiplier, device=input_ids.device, dtype=position_ids.dtype
            )
            decoder_position_ids = torch.cat([jumbo_pos, position_ids], dim=1)

            # print(f"decoder_input_embeddings after cat: {decoder_input_embeddings.shape}")
            # print(f"decoder_position_ids after cat: {decoder_position_ids.shape}")

        # Pass through the decoder (standard BertForTokenClassification)
        # Request hidden states if we need decoder-processed jumbo tokens for classification
        need_hidden_states = jumbo_tokens is not None and self.jumbo_source == "decoder"

        outputs = self.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            return_dict=True,
            output_hidden_states=need_hidden_states,
        )

        # Handle jumbo tokens
        decoder_processed_jumbo_tokens = None
        if jumbo_tokens is not None:
            # Extract the first J positions from decoder's hidden states (the processed jumbo tokens)
            # when using decoder source
            if self.jumbo_source == "decoder":
                # BertForTokenClassification returns hidden_states as a tuple of all layer outputs
                # The last element is the final layer's output: (B, J+seq_len, hidden_dim)
                decoder_processed_jumbo_tokens = outputs.hidden_states[-1][:, : self.jumbo_multiplier, :]

            # Remove jumbo positions from logits (they shouldn't be used for token prediction)
            outputs.logits = outputs.logits[:, self.jumbo_multiplier :, :].contiguous()

        # Attach jumbo_tokens to outputs based on jumbo_source
        if self.jumbo_source == "encoder":
            outputs.jumbo_tokens = jumbo_tokens  # Direct from encoder
        elif self.jumbo_source == "decoder":
            outputs.jumbo_tokens = decoder_processed_jumbo_tokens  # After decoder processing
        else:
            outputs.jumbo_tokens = None

        # print(outputs.logits.shape)
        return outputs

    def forward_baseline(self, input_ids, attention_mask):
        """
        baseline: the mask tokens will be given to the encoder and decoder as input

        """
        batch_size, seq_len = input_ids.size()
        # Pass the encoder inputs through the encoder model
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        encoder_sequence_output = encoder_outputs.last_hidden_state
        # If the encoder and decoder have different hidden states, project the encoder hidden states
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size:
            encoder_sequence_output = self.projection_layer(encoder_sequence_output)

        decoder_input_embeddings = encoder_sequence_output
        # The attention mask of decoder is the same as the input
        decoder_attention_mask = attention_mask
        # The positions ids of the decoder is the same as the input
        decoder_position_ids = position_ids

        # Pass through the decoder (BertForMaskedLM)
        outputs = self.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            return_dict=True,
        )

        return outputs

    @property
    def genus_classifier(self):
        """Backward compatibility alias for taxonomy_classifier."""
        return self.taxonomy_classifier