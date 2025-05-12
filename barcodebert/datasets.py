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


class LazyDNADataset(IterableDataset):
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
            batch_size=None,
            drop_last=True,
            seed=None,
            buffer_size=10000,  # Size of buffer for shuffling
    ):
        self.file_path = file_path
        self.k_mer = k_mer
        self.stride = k_mer if stride is None else stride
        self.max_len = max_len
        self.randomize_offset = randomize_offset
        self.dataset_format = dataset_format
        self._batch_size_per_gpu = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0  # Track current epoch for shuffling
        self.buffer_size = buffer_size
        self._total_lines = None

        if dataset_format not in ["CANADA-1.5M", "BIOSCAN-5M", "DNABERT-2"]:
            raise NotImplementedError(f"Dataset {dataset_format} not supported.")

        if tokenizer == "kmer":
            base_pairs = "ACGT"
            self.special_tokens = ["[MASK]", "[UNK]"]
            UNK_TOKEN = "[UNK]"

            if tokenize_n_nucleotide:
                base_pairs += "N"
            kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=self.k_mer)]

            if tokenize_n_nucleotide:
                prediction_kmers = [k for k in kmers if "N" not in k]
                other_kmers = [k for k in kmers if "N" in k]
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

    def get_total_lines(self):
        """Count the number of lines in the file"""
        if self._total_lines is not None:
            return self._total_lines

        # First check if we have a cached line count file
        cache_path = f"{self.file_path}.linecount"
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self._total_lines = int(f.read().strip())
        else:
            # Use wc command for faster line counting
            try:
                import subprocess
                result = subprocess.run(['wc', '-l', self.file_path],
                                        capture_output=True, text=True)
                self._total_lines = int(result.stdout.split()[0])

                # Subtract header if csv/tsv
                if self.file_path.endswith(('.csv', '.tsv')):
                    self._total_lines -= 1

                # Cache the result
                with open(cache_path, 'w') as f:
                    f.write(str(self._total_lines))

            except (subprocess.SubprocessError, ValueError):
                # Fallback to manual counting
                print("Fallback to manual line counting (slow)...")
                with open(self.file_path, 'r') as f:
                    self._total_lines = sum(1 for _ in f)
                if self.file_path.endswith(('.csv', '.tsv')):
                    self._total_lines -= 1

                # Cache the result
                with open(cache_path, 'w') as f:
                    f.write(str(self._total_lines))

        return self._total_lines

    def set_epoch(self, epoch):
        """Set the epoch to enable proper shuffling between epochs"""
        self.epoch = epoch

    def __len__(self):
        """Returns the number of batches this dataset will produce per epoch"""
        if self._batch_size_per_gpu is None:
            raise ValueError("batch_size_per_gpu must be set before calling __len__")

        # Calculate total samples
        total_lines = self.get_total_lines()

        # Calculate samples per GPU
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        samples_per_gpu = total_lines // world_size

        # Calculate batches per GPU
        if self.drop_last:
            num_batches = samples_per_gpu // self._batch_size_per_gpu
        else:
            num_batches = math.ceil(samples_per_gpu / self._batch_size_per_gpu)

        # Debug info (only print from rank 0)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"LazyDNADataset stats:")
            print(f"  File: {self.file_path}")
            print(f"  Total samples: {total_lines}")
            print(f"  World size: {world_size}")
            print(f"  Samples per GPU: {samples_per_gpu}")
            print(f"  Batch size per GPU: {self._batch_size_per_gpu}")
            print(f"  Batches per GPU: {num_batches}")

        return num_batches

    def parse_row(self, row):
        """Parse a row from the dataframe"""
        dna_seq = row["nucleotides"]
        if self.dataset_format == "CANADA-1.5M":
            label = row["species_name"]
        elif self.dataset_format == "BIOSCAN-5M":
            label = row["species_index"]
        elif self.dataset_format == "DNABERT-2":
            label = 0  # dummy label
        else:
            raise NotImplementedError

        offset = torch.randint(self.k_mer, (1,)).item() if self.randomize_offset else 0
        tokens, att_mask = self.tokenizer(dna_seq, offset=offset)
        return tokens, torch.tensor(int(label), dtype=torch.int64), att_mask

    def __iter__(self):
        """Iterate through the dataset, shuffling and distributing data appropriately"""
        # Determine global rank & world size for distributed training
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank, world_size = 0, 1

        # Handle DataLoader workers
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            total_workers = worker_info.num_workers
            # flatten ranks+workers into a single shard index
            worker_rank = rank * total_workers + worker_id
            worker_world_size = world_size * total_workers
        else:
            worker_rank = rank
            worker_world_size = world_size

        # Create a deterministic shuffle seed that changes per epoch
        shuffle_seed = None
        if self.seed is not None:
            # Different seed for each epoch, rank, and worker
            shuffle_seed = self.seed + self.epoch * 10000 + worker_rank
            torch.manual_seed(shuffle_seed)

        # Efficient chunked reading with a shuffle buffer
        chunk_size = min(self.buffer_size, 5000)  # Reasonable chunk size
        buffer = []

        # For debugging
        samples_yielded = 0

        # Read in chunks for efficiency
        df_iter = pd.read_csv(
            self.file_path,
            sep="\t" if self.file_path.endswith(".tsv") else ",",
            chunksize=chunk_size,
            keep_default_na=False,
        )

        file_position = 0  # Track position in file
        for chunk in df_iter:
            # Find rows that belong to this worker
            chunk_rows = []
            for i, row in chunk.iterrows():
                row_idx = file_position + i - chunk.index[0]
                if row_idx % worker_world_size == worker_rank:
                    chunk_rows.append(row)

            # Add to buffer
            for row in chunk_rows:
                buffer.append(row)

                # When buffer reaches target size, shuffle and yield
                if len(buffer) >= self.buffer_size:
                    # Shuffle buffer if seed is set
                    if shuffle_seed is not None:
                        random.shuffle(buffer)

                    # Process and yield samples
                    for item in buffer:
                        yield self.parse_row(item)
                        samples_yielded += 1

                    # Clear buffer
                    buffer = []

            # Update file position
            file_position += len(chunk)

        # Process any remaining items in buffer
        if buffer:
            if shuffle_seed is not None:
                random.shuffle(buffer)

            for item in buffer:
                yield self.parse_row(item)
                samples_yielded += 1

        # Debug info
        print(f"Worker {worker_rank}/{worker_world_size}: Yielded {samples_yielded} samples")


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
