import torch
import torch.nn as nn
from transformers import BertForTokenClassification, BertModel
from barcodebert.jumbo_bert import create_jumbo_transformer_model

class MAELMModel(nn.Module):

    def __init__(self, encoder_config, decoder_config, jumbo, jumbo_multiplier):
        super(MAELMModel, self).__init__()
        # Encoder BERT model
        # model = BertForMaskedLM(encoder_config)
        self.jumbo = jumbo
        self.jumbo_multiplier = jumbo_multiplier
        if self.jumbo:
            self.encoder = create_jumbo_transformer_model(encoder_config, jumbo_multiplier=jumbo_multiplier)
        else:
            self.encoder = BertModel(encoder_config)

        # Decoder BERT model with token classification head
        # self.decoder = BertForTokenClassification(decoder_config)
        self.decoder = BertForTokenClassification(decoder_config)

        # Encoder Embeddings (word and positional)
        # self.encoder_embedding = self.encoder.embeddings.word_embeddings
        # self.encoder_position_embeddings = self.encoder.embeddings.position_embeddings

        # Projection layer to map encoder hidden states to decoder input
        self.projection_layer = nn.Linear(encoder_config.hidden_size, decoder_config.hidden_size)

        # Decoder Embeddings (word and positional)
        self.decoder_embedding = self.decoder.bert.embeddings.word_embeddings

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
        padded_encoder_attention_mask[seen_indices] = attention_mask[seen_token_positions]
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

        # Map encoder outputs back to the original sequence positions
        decoder_input_embeddings = torch.zeros(
            batch_size, seq_len, encoder_sequence_output.size(-1), device=input_ids.device, dtype=encoder_sequence_output.dtype
        )

        # If the encoder and decoder have different hidden states, project the encoder hidden states
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size:
            encoder_sequence_output = self.projection_layer(encoder_sequence_output)
            # if jumbo_tokens is not None:
            #     jumbo_tokens = self.projection_layer(jumbo_tokens)

        decoder_input_embeddings[seen_token_positions] = encoder_sequence_output[seen_indices].to(decoder_input_embeddings.dtype)

        # this should not be hard coded
        mask_token_id = 0
        # here I am using the embedding of mask tokens from the decoder embeddings not the encoder
        decoder_input_embeddings[mask_positions] = self.decoder_embedding.weight[mask_token_id]

        # The attention mask of decoder is the same as the input
        decoder_attention_mask = attention_mask
        # The positions ids of the decoder is the same as the input
        decoder_position_ids = position_ids

        # Prepend Jumbo tokens to decoder input
        if jumbo_tokens is not None:
            decoder_input_embeddings = torch.cat([jumbo_tokens, decoder_input_embeddings], dim=1)
            jumbo_mask = torch.ones(batch_size, self.jumbo_multiplier, device=input_ids.device,
                                    dtype=attention_mask.dtype)
            decoder_attention_mask = torch.cat([jumbo_mask, decoder_attention_mask], dim=1)
            # Jumbo tokens all get position 0 (they're context, not sequence)
            jumbo_pos = torch.zeros(batch_size, self.jumbo_multiplier, device=input_ids.device,
                                    dtype=position_ids.dtype)
            decoder_position_ids = torch.cat([jumbo_pos, position_ids], dim=1)  # No shift

        # Pass through the decoder (BertForMaskedLM)
        outputs = self.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            return_dict=True,
        )

        if jumbo_tokens is not None:
            outputs.logits = outputs.logits[:, self.jumbo_multiplier:, :]

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
            return_dict=True
        )

        return outputs