import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import BertForTokenClassification, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MAELMModel(nn.Module):

    def __init__(self, encoder_config, decoder_config):
        super(MAELMModel, self).__init__()
        # Encoder BERT model
        self.encoder = BertModel(encoder_config)

        # Decoder BERT model with token classification head
        self.decoder = BertForTokenClassification(decoder_config)

        # Encoder Embeddings (word and positional)
        self.encoder_embedding = self.encoder.embeddings.word_embeddings
        self.encoder_position_embeddings = self.encoder.embeddings.position_embeddings

        # Projection layer to map encoder hidden states to decoder input
        self.projection_layer = nn.Linear(encoder_config.hidden_size, decoder_config.hidden_size)

        # Decoder Embeddings (word and positional)
        self.decoder_embedding = self.decoder.bert.embeddings.word_embeddings

    def forward(self, input_ids, attention_mask, mask_positions=None, labels=None, model_type="maelm_v2"):
        # Auto-detect mask positions if not provided
        if mask_positions is None:
            # Assuming MASK token ID is 4 (DNABERT-2 default)
            mask_positions = input_ids == 4

        if model_type == "maelm_v1":
            return self.forward_v1(input_ids, attention_mask, mask_positions, labels)
        elif model_type == "maelm_v2":
            return self.forward_v2(input_ids, attention_mask, mask_positions, labels)
        elif model_type == "baseline":
            return self.forward_baseline(input_ids, attention_mask, labels)

    def forward_v1(self, input_ids, attention_mask, mask_positions, labels=None):
        """
        This version is removing the masked token from the encoder inout by zeroing out
        the embeddings/attention_masked of the masked tokens.
        """
        batch_size, seq_len = input_ids.size()
        # Generate position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        # Create mask for unmasked positions
        seen_token_indices = ~mask_positions

        # Get embeddings for all tokens
        token_embeddings = self.encoder_embedding(input_ids)
        position_embeddings = self.encoder_position_embeddings(position_ids)
        all_embeddings = token_embeddings + position_embeddings

        # Zero out embeddings of masked tokens for the encoder
        encoder_embeddings = all_embeddings * seen_token_indices.unsqueeze(-1).float()

        # Adjust the attention mask to ignore masked tokens
        encoder_attention_mask = attention_mask * seen_token_indices.int()

        # Pass through the encoder
        encoder_outputs = self.encoder(
            inputs_embeds=encoder_embeddings,
            attention_mask=encoder_attention_mask,
        ).last_hidden_state

        # Prepare decoder input embeddings
        # not sure about this part
        mask_token_id = 0
        mask_token_embedding = self.decoder_embedding.weight[mask_token_id]
        mask_token_embedding_expanded = mask_token_embedding.unsqueeze(0).unsqueeze(0)
        mask_token_embeddings = mask_token_embedding_expanded + position_embeddings

        decoder_input_embeddings = torch.where(
            mask_positions.unsqueeze(-1),
            mask_token_embeddings,
            encoder_outputs + position_embeddings,
        )

        # Decoder attention mask
        decoder_attention_mask = attention_mask

        # Pass through the decoder
        outputs = self.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True,
        )

        return outputs

    def forward_v2(self, input_ids, attention_mask, mask_positions, labels=None):
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
        padded_encoder_attention_mask = torch.zeros(batch_size, max_seen_len, device=input_ids.device).to(attention_mask.dtype)

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

        encoder_sequence_output = encoder_outputs.last_hidden_state

        # Map encoder outputs back to the original sequence positions
        decoder_input_embeddings = torch.zeros(
            batch_size, seq_len, encoder_sequence_output.size(-1), device=input_ids.device
        )

        # If the encoder and decoder have different hidden states, project the encoder hidden states
        if self.encoder.config.output_hidden_states != self.decoder.config.output_hidden_states:
            encoder_sequence_output = self.projection_layer(encoder_sequence_output)

        decoder_input_embeddings[seen_token_positions] = encoder_sequence_output[seen_indices]

        # this should not be hard coded
        mask_token_id = 0
        # here I am using the embedding of mask tokens from the decoder embeddings not the encoder
        decoder_input_embeddings[mask_positions] = self.decoder_embedding.weight[mask_token_id]

        # The attention mask of decoder is the same as the input
        decoder_attention_mask = attention_mask
        # The positions ids of the decoder is the same as the input
        decoder_position_ids = position_ids

        # Pass through the decoder (BertForMaskedLM)
        outputs = self.decoder(
            inputs_embeds=decoder_input_embeddings,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            labels=labels,
            return_dict=True,
        )

        return outputs

    def forward_baseline(self, input_ids, attention_mask, labels=None):
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
        if self.encoder.config.output_hidden_states != self.decoder.config.output_hidden_states:
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
            return_dict=True,
            labels=labels,
            position_ids=decoder_position_ids,
        )

        return outputs

class MAELMForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.encoder = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get last hidden state
        # outputs[0] is last_hidden_state (batch_size, sequence_length, hidden_size)
        sequence_output = outputs[0]

        # Global Average Pooling with Attention Mask
        if attention_mask is not None:
            # Expand mask to match hidden state dimensions
            # attention_mask: (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
            # sequence_output: (batch_size, sequence_length, hidden_size)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            
            # Sum embeddings of non-padding tokens
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            
            # Count non-padding tokens to avoid division by zero
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Average
            GAP_embeddings = sum_embeddings / sum_mask
        else:
            GAP_embeddings = sequence_output.mean(1)

        GAP_embeddings = self.dropout(GAP_embeddings)
        logits = self.classifier(GAP_embeddings)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
