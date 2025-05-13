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


class DNAIterableDataset(IterableDataset):
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
            chunk_size=50000,  # Larger chunk size by default
            shuffle_buffer=10000,  # Size of shuffle buffer (0 for no shuffle)
    ):
        self.file_path = file_path
        self.k_mer = k_mer
        self.stride = k_mer if stride is None else stride
        self.max_len = max_len
        self.randomize_offset = randomize_offset
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer
        self.dataset_format = dataset_format

        # Count lines in file for length approximation
        with open(file_path, 'r') as f:
            self.total_samples = sum(1 for _ in f) - 1  # Subtract header

        # Get column names
        self.column_names = pd.read_csv(file_path, nrows=0).columns.tolist()

        # Create tokenizer
        if tokenizer == "kmer":
            # Vocabulary
            base_pairs = "ACGT"
            self.special_tokens = ["[MASK]", "[UNK]"]
            UNK_TOKEN = "[UNK]"

            if tokenize_n_nucleotide:
                base_pairs += "N"
            kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=self.k_mer)]

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

        # Set up label information
        if dataset_format == "CANADA-1.5M":
            # Create a mapping of species names to indices
            # Use chunk loading to avoid memory issues
            all_species = set()
            for df_chunk in pd.read_csv(file_path, chunksize=chunk_size):
                all_species.update(df_chunk["species_name"].unique())
            self.label_set = sorted(list(all_species))
            self.label_map = {name: idx for idx, name in enumerate(self.label_set)}
            self.num_labels = len(self.label_set)
        elif dataset_format == "BIOSCAN-5M":
            self.num_labels = 22_622
        elif dataset_format == "DNABERT-2":
            # Dummy labels for DNABERT-2
            self.num_labels = 1

        # For tracking current epoch
        self.epoch = 0
        self._shuffle_seed = 0

    def _get_start_end_indices(self):
        """Get start and end indices for this worker and process in distributed setting"""
        # Get worker info
        worker_info = get_worker_info()
        num_workers = 1
        worker_id = 0

        # If inside dataloader worker process
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # Get process info for distributed training
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # Calculate per-worker chunk
        # First shard by process (GPU), then by worker within process
        per_worker = int(math.ceil((self.total_samples - 1) / (world_size * num_workers)))

        # Global worker ID
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        # Calculate initial positions
        start_idx = global_worker_id * per_worker + 1  # +1 to skip header
        end_idx = min(start_idx + per_worker, self.total_samples)

        return start_idx, end_idx, global_worker_id, global_num_workers

    def _process_sample(self, row):
        """Process a single sample row into model inputs"""
        barcode = row["nucleotides"]

        # Process labels based on dataset format
        if self.dataset_format == "CANADA-1.5M":
            label = self.label_map.get(row["species_name"], 0)
        elif self.dataset_format == "BIOSCAN-5M":
            label = row["species_index"]
        else:  # DNABERT-2
            label = 0

        # Process barcode with tokenizer
        if self.randomize_offset:
            offset = torch.randint(self.k_mer, (1,)).item()
        else:
            offset = 0

        processed_barcode, att_mask = self.tokenizer(barcode, offset=offset)
        label = torch.tensor(label, dtype=torch.int64)

        return processed_barcode, label, att_mask

    def set_epoch(self, epoch):
        """Update epoch counter for deterministic shuffling"""
        self.epoch = epoch
        self._shuffle_seed = hash((epoch, 0))  # Use hash for more entropy

    def __iter__(self):
        # Get worker-specific start/end indices
        start_idx, end_idx, worker_id, num_workers = self._get_start_end_indices()

        # Create pandas reader for this worker's chunk
        skiprows = list(range(1, start_idx))  # Skip rows before start
        nrows = end_idx - start_idx + 1  # Include end index

        # For better performance, read in chunks
        chunk_size = min(self.chunk_size, nrows)
        chunks_iter = pd.read_csv(
            self.file_path,
            skiprows=skiprows,
            nrows=nrows,
            chunksize=chunk_size,
            names=self.column_names,
            header=None if skiprows else 0  # Only use header if reading from start
        )

        if self.shuffle_buffer > 0:
            # Use a shuffle buffer for better performance
            buffer = []
            buffer_size = min(self.shuffle_buffer, nrows)

            # Set the random seed based on epoch, worker ID for reproducibility
            rng = random.Random(self._shuffle_seed + worker_id)

            # Process chunks with shuffling
            for chunk in chunks_iter:
                # Process each row in the chunk
                for _, row in chunk.iterrows():
                    # Add processed sample to buffer
                    buffer.append(self._process_sample(row))

                    # If buffer is full, yield a random sample
                    if len(buffer) >= buffer_size:
                        idx = rng.randint(0, len(buffer) - 1)
                        yield buffer[idx]
                        buffer[idx] = buffer[-1]
                        buffer.pop()

            # Empty the buffer
            while buffer:
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer[idx]
                buffer[idx] = buffer[-1]
                buffer.pop()
        else:
            # No shuffling - directly yield samples
            for chunk in chunks_iter:
                for _, row in chunk.iterrows():
                    yield self._process_sample(row)

    def __len__(self):
        # Approximate length for progress bars
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        worker_info = get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
        else:
            num_workers = 1

        return math.ceil(self.total_samples / (world_size * num_workers))


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
