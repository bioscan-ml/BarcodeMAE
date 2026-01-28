"""
Datasets.
"""

import json
import os
from itertools import product

import numpy as np
import pandas as pd
import torch
from mycoai.data import Data
from mycoai.data.encoders import TaxonEncoder
from torch.utils.data import Dataset
from torchtext.vocab import vocab as build_vocab_from_dict
from tqdm import tqdm
from transformers import AutoTokenizer


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
        taxonomic_level="species",
        label2id=None,
        return_genus=False,  # Deprecated: use return_taxonomy_level instead
        return_taxonomy_level=None,  # Can be: phylum, class, order, family, genus, species
    ):
        self.k_mer = k_mer
        self.stride = k_mer if stride is None else stride
        self.max_len = max_len
        self.randomize_offset = randomize_offset
        self.barcodes = []
        self.tax_encoder = None
        self.labels = []
        self.taxonomy_labels = []  # For any taxonomic level
        self.label2id = label2id

        # Handle backward compatibility: return_genus -> return_taxonomy_level
        if return_genus and return_taxonomy_level is None:
            return_taxonomy_level = "genus"
        self.return_taxonomy_level = return_taxonomy_level

        # Check that the dataframe contains a valid format
        if dataset_format not in ["CANADA-1.5M", "BIOSCAN-5M", "ITS-5M"]:
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
        if dataset_format == "ITS-5M":
            if "train" in file_path:
                fungi_data = Data(file_path, allow_duplicates=True)
            elif "test" in file_path:
                fungi_data = Data(file_path, allow_duplicates=False)

            self.tax_encoder = TaxonEncoder(data=fungi_data) if self.tax_encoder is None else self.tax_encoder
            for _, row in tqdm(fungi_data.data.iterrows(), total=fungi_data.data.shape[0]):
                self.barcodes.append(row["sequence"])
                # self.labels.append(self.tax_encoder.encode(row))
            # self.tax_encoder.finish_training()
        else:
            self.barcodes = df["nucleotides"].to_list()

        if dataset_format == "CANADA-1.5M":
            self.labels, self.label_set = pd.factorize(df["species_name"], sort=True)
            self.num_labels = len(self.label_set)
            # Load taxonomy labels if requested
            if self.return_taxonomy_level:
                taxonomy_column = f"{self.return_taxonomy_level}_name"
                if taxonomy_column in df.columns:
                    # Replace empty strings and NaN with 'UNKNOWN'
                    taxonomy_col = df[taxonomy_column].replace("", "UNKNOWN").fillna("UNKNOWN")
                    self.taxonomy_labels, self.taxonomy_label_set = pd.factorize(taxonomy_col, sort=True)
                    # Map 'UNKNOWN' samples to -1 so they're excluded from pair creation
                    unknown_mask = taxonomy_col == "UNKNOWN"
                    num_unknown = unknown_mask.sum()
                    self.taxonomy_labels = [
                        -1 if is_unknown else label for label, is_unknown in zip(self.taxonomy_labels, unknown_mask)
                    ]
                    print(f"Taxonomy labels: {len(self.taxonomy_labels)} total, {num_unknown} marked as UNKNOWN (-1)")
                    print(f"Unique taxonomy categories: {self.taxonomy_label_set}")
                    json.dump(
                        self.taxonomy_label_set.tolist(),
                        open("/home/m4safari/projects/def-lila-ab/m4safari/BarcodeMAE/taxonomy_label_set.json", "w"),
                    )

                else:
                    print(f"Warning: Column '{taxonomy_column}' not found. Using dummy labels.")
                    self.taxonomy_labels = [0] * len(self.labels)
            else:
                self.taxonomy_labels = [0] * len(self.labels)  # Dummy labels
        elif dataset_format == "BIOSCAN-5M":
            self.label_names = df["species_name"].to_list()
            self.labels = df["species_index"].to_list()
            self.num_labels = 22_622
            # Load taxonomy labels if requested
            if self.return_taxonomy_level:
                taxonomy_column = f"{self.return_taxonomy_level}_name"
                if taxonomy_column in df.columns:
                    # Replace empty strings and NaN with 'UNKNOWN'
                    taxonomy_col = df[taxonomy_column].replace("", "UNKNOWN").fillna("UNKNOWN")
                    self.taxonomy_labels, self.taxonomy_label_set = pd.factorize(taxonomy_col, sort=True)
                    # Map 'UNKNOWN' samples to -1 so they're excluded from pair creation
                    unknown_mask = taxonomy_col == "UNKNOWN"
                    num_unknown = unknown_mask.sum()
                    self.taxonomy_labels = [
                        -1 if is_unknown else label for label, is_unknown in zip(self.taxonomy_labels, unknown_mask)
                    ]
                    print(f"Taxonomy labels: {len(self.taxonomy_labels)} total, {num_unknown} marked as UNKNOWN (-1)")
                    print(f"Unique taxonomy categories: {self.taxonomy_label_set.tolist()}")

                else:
                    print(f"Warning: Column '{taxonomy_column}' not found. Using dummy labels.")
                    self.taxonomy_labels = [0] * len(self.labels)
            else:
                self.taxonomy_labels = [0] * len(self.labels)  # Dummy labels
        elif dataset_format == "ITS-5M":
            labels_file = file_path.replace(".fasta", "_labels.csv")
            if os.path.isfile(labels_file):
                labels_df = pd.read_csv(labels_file)
                self.labels = labels_df[taxonomic_level].to_list()
            else:
                raise FileNotFoundError("Labels file not found for ITS-5M. Expected: " + labels_file)

            if len(self.labels) != len(self.barcodes):
                raise ValueError(
                    f"Mismatch between barcodes ({len(self.barcodes)}) and labels ({len(self.labels)}). "
                    "Ensure the labels CSV matches the FASTA order."
                )

            labels_np = np.asarray(self.labels)
            valid_mask = labels_np != 9999999  # label for unknown labels
            n_before = len(self.labels)
            if not valid_mask.any():
                raise ValueError("All ITS-5M labels are invalid.")
            # apply mask
            self.barcodes = [b for b, keep in zip(self.barcodes, valid_mask) if keep]
            self.labels = labels_np[valid_mask].tolist()
            print("max labels before change", max(self.labels))
            self.num_labels = len(self.label_set)
            print(f"[DNADataset][ITS-5M] Removed {n_before - len(self.labels)} samples with unknown labels.")
            # Reindex to contiguous [0..C-1] and keep mappings
            # if self.label2id is None:
            #     self.labels, self.label_set = pd.factorize(self.labels, sort=True)  # self.labels now 0..C-1
            #     self.num_labels = len(self.label_set)
            #     print("max labels after change",max(self.labels))
            #     self.id2label = {i: lab for i, lab in enumerate(self.label_set)}
            #     self.label2id = {lab: i for i, lab in enumerate(self.label_set)}
            # else:
            #     mapped = []
            #     kept_barcodes = []
            #     dropped = 0
            #     for b, lab in zip(self.barcodes, self.labels):
            #         if lab in self.label2id:
            #             mapped.append(self.label2id[lab])
            #             kept_barcodes.append(b)
            #         else:
            #             dropped += 1
            #     if dropped:
            #         print(f"[DNADataset][ITS-5M] Dropped {dropped} samples with taxa unseen in train.")
            #     self.barcodes = kept_barcodes
            #     self.labels = mapped
            #     self.num_labels = len(self.label2id)
            #     self.id2label = {i: lab for lab, i in self.label2id.items()}

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        if self.randomize_offset:
            offset = torch.randint(self.k_mer, (1,)).item()
        else:
            offset = 0
        processed_barcode, att_mask = self.tokenizer(self.barcodes[idx], offset=offset)
        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        if self.return_taxonomy_level:
            taxonomy_label = torch.tensor(self.taxonomy_labels[idx], dtype=torch.int64)
            return processed_barcode, label, att_mask, taxonomy_label
        else:
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
