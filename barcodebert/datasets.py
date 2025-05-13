"""
Datasets.
"""

import os
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from torchtext.vocab import vocab as build_vocab_from_dict
from transformers import AutoTokenizer
import torch.distributed as dist
from torch.utils.data import get_worker_info
import math
import random


class KmerTokenizer(object):
    def __init__(self, k, vocabulary_mapper, stride=1, padding=False, max_len=660):
        self.k = k
        self.stride = stride
        self.padding = padding
        self.max_len = max_len
        self.vocabulary_mapper = vocabulary_mapper

    def __call__(self, dna_sequence, offset=0) -> tuple[list, list]:
        tokens = []
        att_mask = [1] * (self.max_len // self.stride)
        x = dna_sequence[offset:]
        if self.padding:
            if len(x) > self.max_len:
                x = x[: self.max_len]
            else:
                att_mask[len(x) // self.stride :] = [0] * (len(att_mask) - len(x) // self.stride)
                x = x + "N" * (self.max_len - len(x))
        for i in range(0, len(x) - self.k + 1, self.stride):
            k_mer = x[i : i + self.k]
            tokens.append(k_mer)

        tokens = torch.tensor(self.vocabulary_mapper(tokens), dtype=torch.int64)
        att_mask = torch.tensor(att_mask, dtype=torch.int32)

        return tokens, att_mask


class BPETokenizer(object):
    def __init__(self, padding=False, max_tokenized_len=128, bpe_path=None):
        self.padding = padding
        self.max_tokenized_len = max_tokenized_len

        assert os.path.isdir(bpe_path), f"The bpe path does not exist: {bpe_path}"

        self.bpe = AutoTokenizer.from_pretrained(bpe_path)

        # root_folder = os.path.dirname(__file__)
        # if bpe_type == "dnabert":
        #     # self.bpe = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        #     bpe_folder = os.path.join(root_folder, "bpe_tokenizers", "bpe_dnabert2")
        #     assert os.path.isdir(bpe_folder), f"Directory does not exist: {bpe_folder}"
        #     self.bpe = AutoTokenizer.from_pretrained(f"{bpe_folder}/")
        # elif bpe_type.__contains__("barcode"):
        #     length = bpe_type.split("_")[-1]
        #     bpe_folder = os.path.join(root_folder, "bpe_tokenizers", f"bpe_barcode_{length}")
        #     assert os.path.isdir(bpe_folder), f"Directory does not exist: {bpe_folder}"
        #     self.bpe = AutoTokenizer.from_pretrained(bpe_folder)
        # else:
        #     raise NotImplementedError(f"bpe_type {bpe_type} is  not supported.")

    def __call__(self, dna_sequence, offset=0) -> tuple[list, list]:
        x = dna_sequence[offset:]
        tokens = self.bpe(x, padding=True, return_tensors="pt")["input_ids"]
        tokens[tokens == 2] = 3
        tokens[tokens == 1] = 2
        tokens[tokens == 0] = 1  # all the UNK + CLS have token of 1

        tokens = tokens[0].tolist()

        if len(tokens) > self.max_tokenized_len:
            att_mask = [1] * self.max_tokenized_len
            tokens = tokens[: self.max_tokenized_len]
        else:
            att_mask = [1] * (len(tokens)) + [0] * (self.max_tokenized_len - len(tokens))
            tokens = tokens + [1] * (self.max_tokenized_len - len(tokens))

        att_mask = torch.tensor(att_mask, dtype=torch.int32)
        tokens = torch.tensor(tokens, dtype=torch.int64)
        return tokens, att_mask


class StreamingDNADataset(Dataset):
    def __init__(
            self,
            file_path,
            k_mer=4,
            stride=None,
            max_len=256,
            randomize_offset=False,
            tokenizer="kmer",
            bpe_path=None,
            tokenize_n_nucleotide=False,
            dataset_format="CANADA-1.5M",
            chunk_size=10000,  # Number of rows to load at once
    ):
        self.file_path = file_path
        self.k_mer = k_mer
        self.stride = k_mer if stride is None else stride
        self.max_len = max_len
        self.randomize_offset = randomize_offset
        self.chunk_size = chunk_size

        # Check that the dataframe format is valid
        if dataset_format not in ["CANADA-1.5M", "BIOSCAN-5M", "DNABERT-2"]:
            raise NotImplementedError(f"Dataset {dataset_format} not supported.")
        self.dataset_format = dataset_format

        # Set up tokenizer
        if tokenizer == "kmer":
            # Vocabulary
            base_pairs = "ACGT"
            self.special_tokens = ["[MASK]", "[UNK]"]
            UNK_TOKEN = "[UNK]"

            if tokenize_n_nucleotide:
                base_pairs += "N"
            kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=self.k_mer)]

            # Separate between good (idx < 4**k) and bad k-mers (idx > 4**k) for prediction
            if tokenize_n_nucleotide:
                prediction_kmers = []
                other_kmers = []
                for kmer in kmers:
                    if "N" in kmer:
                        other_kmers.append(kmer)
                    else:
                        prediction_kmers.append(kmer)

                kmers = prediction_kmers + other_kmers

            kmer_dict = dict.fromkeys(kmers, 1)
            self.vocab = build_vocab_from_dict(kmer_dict, specials=self.special_tokens)
            self.vocab.set_default_index(self.vocab[UNK_TOKEN])
            self.vocab_size = len(self.vocab)
            self.tokenizer = KmerTokenizer(
                self.k_mer, self.vocab, stride=self.stride, padding=True, max_len=self.max_len
            )
        elif tokenizer == "bpe":
            self.tokenizer = BPETokenizer(padding=True, max_tokenized_len=self.max_len, bpe_path=bpe_path)
            self.vocab_size = self.tokenizer.bpe.vocab_size
        else:
            raise ValueError(f'Tokenizer "{tokenizer}" not recognized.')

        # Get dataset length and labels without loading all data
        self._calculate_length_and_labels()

    def set_epoch(self, epoch):
        """Update the dataset state for a new epoch"""
        # Reset cache to ensure fresh data loading
        if hasattr(self, '_current_chunk'):
            del self._current_chunk
            del self._current_chunk_key
        # You can add epoch-specific randomization here if needed
        if self.randomize_offset:
            # Set a deterministic but different seed for each epoch
            random.seed(epoch)
    def _calculate_length_and_labels(self):
        """Calculate dataset length and prepare label mapping without loading full dataset"""
        # Count number of lines in the file (subtract 1 for header)
        with open(self.file_path, 'r') as f:
            self.length = sum(1 for _ in f) - 1

        # For label information, we can either:
        # 1. Read just the label column with usecols
        # 2. Process in chunks to get unique labels

        if self.dataset_format == "CANADA-1.5M":
            # Read just the species_name column to get unique values
            label_df = pd.read_csv(
                self.file_path,
                sep="\t" if self.file_path.endswith(".tsv") else ",",
                usecols=["species_name"]
            )
            self.label_set, _ = pd.factorize(label_df["species_name"], sort=True)
            self.num_labels = len(self.label_set)
        elif self.dataset_format == "BIOSCAN-5M":
            # Just need the number of labels
            self.num_labels = 22_622
        elif self.dataset_format == "DNABERT-2":
            # Dummy labels for DNABERT-2
            self.num_labels = 1

    def _get_chunk(self, idx):
        """Get the chunk containing the index"""
        chunk_idx = idx // self.chunk_size
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.length)

        # Calculate skip and nrows for efficient reading
        skiprows = list(range(1, start_idx + 1))  # +1 to account for header
        nrows = end_idx - start_idx

        # Load the specific chunk
        chunk = pd.read_csv(
            self.file_path,
            sep="\t" if self.file_path.endswith(".tsv") else ",",
            skiprows=skiprows,
            nrows=nrows,
            header=0 if not skiprows else None,  # Use header if reading from start
            names=None if not skiprows else self._get_column_names(),
            keep_default_na=False
        )

        return chunk

    def _get_column_names(self):
        """Get column names from file"""
        return pd.read_csv(
            self.file_path,
            sep="\t" if self.file_path.endswith(".tsv") else ",",
            nrows=0
        ).columns.tolist()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Calculate which chunk and position within chunk
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        # Get current cache key
        current_chunk_key = chunk_idx

        # Check if we need to load a new chunk
        if not hasattr(self, '_current_chunk') or self._current_chunk_key != current_chunk_key:
            self._current_chunk = self._get_chunk(idx)
            self._current_chunk_key = current_chunk_key

        # Get data from current chunk
        barcode = self._current_chunk.iloc[local_idx]["nucleotides"]

        # Process labels based on dataset format
        if self.dataset_format == "CANADA-1.5M":
            # Need to factorize on the fly
            if not hasattr(self, '_label_mapping'):
                # Lazily create label mapping
                all_labels = pd.read_csv(
                    self.file_path,
                    sep="\t" if self.file_path.endswith(".tsv") else ",",
                    usecols=["species_name"]
                )
                unique_labels, _ = pd.factorize(all_labels["species_name"], sort=True)
                self._label_mapping = {name: i for i, name in enumerate(unique_labels)}

            label = self._label_mapping.get(self._current_chunk.iloc[local_idx]["species_name"], 0)
        else:  # BIOSCAN-5M format
            label = self._current_chunk.iloc[local_idx]["species_index"]

        # Process barcode
        if self.randomize_offset:
            offset = torch.randint(self.k_mer, (1,)).item()
        else:
            offset = 0

        processed_barcode, att_mask = self.tokenizer(barcode, offset=offset)
        label = torch.tensor(label, dtype=torch.int64)

        return processed_barcode, label, att_mask

class DNADataset(Dataset):
    def __init__(
        self,
        file_path,
        k_mer=4,
        stride=None,
        max_len=256,
        randomize_offset=False,
        tokenizer="kmer",
        bpe_path=None,
        tokenize_n_nucleotide=False,
        dataset_format="CANADA-1.5M",
    ):
        self.k_mer = k_mer
        self.stride = k_mer if stride is None else stride
        self.max_len = max_len
        self.randomize_offset = randomize_offset

        # Check that the dataframe contains a valid format
        if dataset_format not in ["CANADA-1.5M", "BIOSCAN-5M", "DNABERT-2"]:
            raise NotImplementedError(f"Dataset {dataset_format} not supported.")

        if tokenizer == "kmer":
            # Vocabulary
            base_pairs = "ACGT"
            self.special_tokens = ["[MASK]", "[UNK]"]  # ["[MASK]", "[CLS]", "[SEP]", "[PAD]", "[EOS]", "[UNK]"]
            UNK_TOKEN = "[UNK]"

            if tokenize_n_nucleotide:
                # Encode kmers which contain N differently depending on where it is
                base_pairs += "N"
            kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=self.k_mer)]

            # Separate between good (idx < 4**k) and bad k-mers (idx > 4**k) for prediction
            if tokenize_n_nucleotide:
                prediction_kmers = []
                other_kmers = []
                for kmer in kmers:
                    if "N" in kmer:
                        other_kmers.append(kmer)
                    else:
                        prediction_kmers.append(kmer)

                kmers = prediction_kmers + other_kmers

            kmer_dict = dict.fromkeys(kmers, 1)
            self.vocab = build_vocab_from_dict(kmer_dict, specials=self.special_tokens)
            self.vocab.set_default_index(self.vocab[UNK_TOKEN])
            self.vocab_size = len(self.vocab)
            self.tokenizer = KmerTokenizer(
                self.k_mer, self.vocab, stride=self.stride, padding=True, max_len=self.max_len
            )
        elif tokenizer == "bpe":
            self.tokenizer = BPETokenizer(padding=True, max_tokenized_len=self.max_len, bpe_path=bpe_path)
            self.vocab_size = self.tokenizer.bpe.vocab_size
        else:
            raise ValueError(f'Tokenizer "{tokenizer}" not recognized.')
        df = pd.read_csv(file_path, sep="\t" if file_path.endswith(".tsv") else ",", keep_default_na=False)
        self.barcodes = df["nucleotides"].to_list()

        if dataset_format == "CANADA-1.5M":
            self.labels, self.label_set = pd.factorize(df["species_name"], sort=True)
            self.num_labels = len(self.label_set)
        elif dataset_format == "BIOSCAN-5M":
            self.label_names = df["species_name"].to_list()
            self.labels = df["species_index"].to_list()
            self.num_labels = 22_622
        elif dataset_format == "DNABERT-2":
            # this is just dummy labels for the DNABERT-S_2M dataset
            self.labels = np.zeros(len(self.barcodes), dtype=np.int64)
            self.num_labels = 1

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        if self.randomize_offset:
            offset = torch.randint(self.k_mer, (1,)).item()
        else:
            offset = 0
        processed_barcode, att_mask = self.tokenizer(self.barcodes[idx], offset=offset)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)
        return processed_barcode, label, att_mask


def representations_from_df(df, target_level, model, tokenizer, dataset_name, mode=None, mask_rate=None):

    orders = df["order_name"].to_numpy()
    if dataset_name == "CANADA-1.5M":
        _label_set, y = np.unique(df[target_level], return_inverse=True)
    elif dataset_name == "BIOSCAN-5M":
        # _label_set = np.unique(df[target_level])
        y = df[target_level]
    else:
        raise NotImplementedError("Dataset format is not supported. Must be one of CANADA-1.5M or BIOSCAN-5M")

    dna_embeddings = []

    with torch.no_grad():
        for barcode in df["nucleotides"]:
            x, att_mask = tokenizer(barcode)

            x = x.unsqueeze(0).to(model.device)
            att_mask = att_mask.unsqueeze(0).to(model.device)
            x = model(x, att_mask).hidden_states[-1]
            # previous mean pooling
            # x = x.mean(1)
            # dna_embeddings.append(x.cpu().numpy())

            # updated mean pooling to account for the attention mask and padding tokens
            # sum the embeddings of the tokens (excluding padding tokens)
            sum_embeddings = (x * att_mask.unsqueeze(-1)).sum(1)  # (batch_size, hidden_size)
            # sum the attention mask (number of tokens in the sequence without considering the padding tokens)
            sum_mask = att_mask.sum(1, keepdim=True)
            # calculate the mean embeddings
            mean_embeddings = sum_embeddings / sum_mask  # (batch_size, hidden_size)

            dna_embeddings.append(mean_embeddings.cpu().numpy())

    print(f"There are {len(df)} points in the dataset")
    latent = np.array(dna_embeddings)
    latent = np.squeeze(latent, 1)
    print(latent.shape)
    return latent, y, orders


# def representations_from_df(df, target_level, model, tokenizer, dataset_name, mode="nonmask", mask_rate=0.5):
#
#     orders = df["order_name"].to_numpy()
#     if dataset_name == "CANADA-1.5M":
#         _label_set, y = np.unique(df[target_level], return_inverse=True)
#     elif dataset_name == "BIOSCAN-5M":
#         # _label_set = np.unique(df[target_level])
#         y = df[target_level]
#     else:
#         raise NotImplementedError("Dataset format is not supported. Must be one of CANADA-1.5M or BIOSCAN-5M")
#
#     dna_embeddings = []
#     print("mode", mode)
#     print("mask rate", mask_rate)
#
#     with torch.no_grad():
#         for barcode in df["nucleotides"]:
#             x, att_mask = tokenizer(barcode)
#
#             if mode == "drop":
#                 x, att_mask = tokenizer(barcode)
#                 x = x.unsqueeze(0).to(model.device)
#                 att_mask = att_mask.unsqueeze(0).to(model.device)
#
#                 random_mask = torch.rand(x.size())
#                 mask_token_ratio = mask_rate
#                 mask_ratio = 1
#                 dropped_tokens = random_mask < mask_token_ratio * mask_ratio
#                 att_mask[dropped_tokens] = 0
#
#                 x = model(x, att_mask).hidden_states[-1][~dropped_tokens]
#                 att_mask = att_mask[~dropped_tokens].unsqueeze(-1)
#
#                 sum_embeddings = (x * att_mask.unsqueeze(-1)).sum(1)  # (batch_size, hidden_size)
#                 # sum the attention mask (number of tokens in the sequence without considering the padding tokens)
#                 sum_mask = att_mask.sum(0, keepdim=True)
#                 # calculate the mean embeddings
#                 mean_embeddings = sum_embeddings / sum_mask  # (batch_size, hidden_size)
#
#                 dna_embeddings.append(mean_embeddings.cpu().numpy().reshape(-1))
#
#             elif mode == "combined":
#
#                 n_special_tokens = 2
#                 # print(x.size())
#                 random_mask = torch.rand(x.size())
#                 mask_token_ratio = mask_rate
#                 mask_ratio = 1
#                 masked_unseen_tokens = random_mask < mask_token_ratio * mask_ratio
#
#                 x = x.unsqueeze(0).to(model.device)
#                 att_mask = att_mask.unsqueeze(0).to(model.device)
#                 masked_unseen_tokens = masked_unseen_tokens.to(model.device)
#
#                 special_tokens_mask = x > (n_special_tokens - 1)
#                 masked_unseen_tokens_n = masked_unseen_tokens & special_tokens_mask
#
#                 x[masked_unseen_tokens_n] = 0
#
#                 x = model(x, att_mask).hidden_states[-1]
#
#                 sum_embeddings = (x * att_mask.unsqueeze(-1)).sum(1)  # (batch_size, hidden_size)
#                 # sum the attention mask (number of tokens in the sequence without considering the padding tokens)
#                 sum_mask = att_mask.sum(1, keepdim=True)
#                 # calculate the mean embeddings
#                 mean_embeddings = sum_embeddings / sum_mask  # (batch_size, hidden_size)
#
#                 dna_embeddings.append(mean_embeddings.cpu().numpy().reshape(-1))
#
#             elif mode == "mask":
#                 n_special_tokens = 2
#                 # print(x.size())
#                 random_mask = torch.rand(x.size())
#                 mask_token_ratio = 0.5
#                 mask_ratio = 1
#                 masked_unseen_tokens = random_mask < mask_token_ratio * mask_ratio
#                 # print(masked_unseen_tokens)
#
#                 x = x.unsqueeze(0).to(model.device)
#                 att_mask = att_mask.unsqueeze(0).to(model.device)
#                 masked_unseen_tokens = masked_unseen_tokens.to(model.device)
#
#                 special_tokens_mask = x > (n_special_tokens - 1)
#                 masked_unseen_tokens_n = masked_unseen_tokens & special_tokens_mask
#                 # print(masked_unseen_tokens_n)
#
#                 x[masked_unseen_tokens_n] = 0
#                 # att_mask[~masked_unseen_tokens_n] = 0
#                 x = model(x, att_mask).hidden_states[-1][masked_unseen_tokens_n]
#                 # print(x.shape)
#
#                 mean_embeddings = x.mean(0)
#
#                 # print(mean_embeddings.shape)
#                 dna_embeddings.append(mean_embeddings.cpu().numpy().reshape(-1))
#
#             elif mode == "nonmask":
#
#                 n_special_tokens = 2
#                 # print(x.size())
#                 random_mask = torch.rand(x.size())
#                 mask_token_ratio = 0.5
#                 mask_ratio = 1
#                 masked_unseen_tokens = random_mask < mask_token_ratio * mask_ratio
#                 # print(masked_unseen_tokens)
#
#                 x = x.unsqueeze(0).to(model.device)
#                 att_mask = att_mask.unsqueeze(0).to(model.device)
#                 masked_unseen_tokens = masked_unseen_tokens.to(model.device)
#
#                 special_tokens_mask = x > (n_special_tokens - 1)
#                 masked_unseen_tokens_n = masked_unseen_tokens & special_tokens_mask
#                 # print(masked_unseen_tokens_n)
#
#                 x[masked_unseen_tokens_n] = 0
#                 # att_mask[~masked_unseen_tokens_n] = 0
#                 x = model(x, att_mask).hidden_states[-1][~masked_unseen_tokens_n]
#
#                 mean_embeddings = x.mean(0)
#
#                 dna_embeddings.append(mean_embeddings.cpu().numpy().reshape(-1))
#             else:
#                 raise ValueError(f"Mode {mode} not recognized.")
#
#     print(f"There are {len(df)} points in the dataset")
#     latent = np.array(dna_embeddings)
#     # latent = np.squeeze(latent, 1)
#     print(latent.shape)
#     return latent, y, orders
