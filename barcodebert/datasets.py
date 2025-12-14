"""
Datasets.
"""

import os
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import vocab as build_vocab_from_dict
from transformers import AutoTokenizer
from mycoai.data import Data
from tqdm import tqdm
from mycoai.data.encoders import TaxonEncoder


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
        tax_encoder=None,
        use_hierarchical=False,  # NEW: If True, load all 6 labels and don't filter
    ):
        self.k_mer = k_mer
        self.stride = k_mer if stride is None else stride
        self.max_len = max_len
        self.randomize_offset = randomize_offset
        self.barcodes = []
        self.tax_encoder = tax_encoder
        self.labels = []
        self.label2id = label2id
        self.classes_per_level = None
        self.use_hierarchical = use_hierarchical  # NEW
        self.all_labels = None  # NEW: Store all 6 levels when hierarchical

        # Check that the dataframe contains a valid format
        if dataset_format not in ["CANADA-1.5M", "BIOSCAN-5M", "ITS-5M"]:
            raise NotImplementedError(f"Dataset {dataset_format} not supported.")

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

        df = pd.read_csv(file_path, sep="\t" if file_path.endswith(".tsv") else ",", keep_default_na=False)

        if dataset_format == "ITS-5M":
            if "train" in file_path:
                fungi_data = Data(file_path, allow_duplicates=True)
            elif "test" in file_path:
                fungi_data = Data(file_path, allow_duplicates=False)

            if self.tax_encoder is None:
                self.tax_encoder = TaxonEncoder(data=fungi_data)

            for index, row in tqdm(fungi_data.data.iterrows(), total=fungi_data.data.shape[0]):
                self.barcodes.append(row["sequence"])

            # Compute class counts per level for InferSum
            self._compute_classes_per_level()

        else:
            self.barcodes = df["nucleotides"].to_list()

        if dataset_format == "CANADA-1.5M":
            self.labels, self.label_set = pd.factorize(df["species_name"], sort=True)
            self.num_labels = len(self.label_set)
        elif dataset_format == "BIOSCAN-5M":
            self.label_names = df["species_name"].to_list()
            self.labels = df["species_index"].to_list()
            self.num_labels = 22_622
        elif dataset_format == "ITS-5M":
            labels_file = file_path.replace(".fasta", "_labels.csv")
            if os.path.isfile(labels_file):
                labels_df = pd.read_csv(labels_file)
            else:
                raise FileNotFoundError("Labels file not found for ITS-5M. Expected: " + labels_file)

            # NEW: Different handling based on use_hierarchical
            if self.use_hierarchical:
                self._load_hierarchical_labels(labels_df, taxonomic_level)
            else:
                self._load_single_level_labels(labels_df, taxonomic_level)

    def _load_single_level_labels(self, labels_df, taxonomic_level):
        """Original behavior: load single level, filter samples without labels."""
        self.labels = labels_df[taxonomic_level].to_list()

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

        # Filter out samples without labels
        self.barcodes = [b for b, keep in zip(self.barcodes, valid_mask) if keep]
        self.labels = labels_np[valid_mask].tolist()
        print("max labels before change", max(self.labels))
        self.num_labels = len(set(self.labels))
        print(f"[DNADataset][ITS-5M] Removed {n_before - len(self.labels)} samples with unknown labels.")

    def _load_hierarchical_labels(self, labels_df, taxonomic_level):
        """NEW: Load all 6 taxonomic levels, don't filter samples."""
        self.tax_levels = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        self.all_labels = []

        n_samples = len(labels_df)
        if n_samples != len(self.barcodes):
            raise ValueError(
                f"Mismatch between barcodes ({len(self.barcodes)}) and labels ({n_samples}). "
                "Ensure the labels CSV matches the FASTA order."
            )

        # Count valid samples at each level for logging
        valid_counts = {lvl: 0 for lvl in self.tax_levels}

        for idx in range(n_samples):
            row_labels = []
            for lvl in self.tax_levels:
                if lvl in labels_df.columns:
                    val = labels_df[lvl].iloc[idx]
                    # Use -1 for missing/unknown labels
                    if val == 9999999 or pd.isna(val):
                        row_labels.append(-1)
                    else:
                        row_labels.append(int(val))
                        valid_counts[lvl] += 1
                else:
                    row_labels.append(-1)
            self.all_labels.append(row_labels)

        # Log statistics
        print(f"[DNADataset][ITS-5M][Hierarchical] Loaded {n_samples} samples with multi-level labels:")
        for lvl in self.tax_levels:
            pct = 100.0 * valid_counts[lvl] / n_samples
            print(f"  {lvl}: {valid_counts[lvl]}/{n_samples} ({pct:.1f}%) valid labels")

        # For compatibility, also store single-level labels (with -1 for missing)
        level_idx = self.tax_levels.index(taxonomic_level)
        self.labels = [row[level_idx] for row in self.all_labels]

        # num_labels at the target taxonomic level (excluding -1)
        valid_labels_at_level = [l for l in self.labels if l != -1]
        if valid_labels_at_level:
            self.num_labels = max(valid_labels_at_level) + 1
        else:
            self.num_labels = self.classes_per_level[level_idx] if self.classes_per_level else 0

        print(f"[DNADataset][ITS-5M][Hierarchical] num_labels at {taxonomic_level}: {self.num_labels}")

    def _compute_classes_per_level(self):
        """Compute number of classes at each taxonomic level from tax_encoder."""
        if self.tax_encoder is not None:
            self.classes_per_level = []
            # Get class counts from inference matrices dimensions
            # inference_matrices[i] maps from level i+1 to level i
            # Shape: (num_children, num_parents)
            for lvl in range(5):  # 0 to 4
                if lvl < len(self.tax_encoder.inference_matrices):
                    # Number of classes at level lvl = number of columns (parents)
                    self.classes_per_level.append(
                        self.tax_encoder.inference_matrices[lvl].shape[1]
                    )
                else:
                    self.classes_per_level.append(0)

            # For species level (lvl=5), get from the last matrix's rows
            if len(self.tax_encoder.inference_matrices) > 0:
                self.classes_per_level.append(
                    self.tax_encoder.inference_matrices[-1].shape[0]
                )
            else:
                self.classes_per_level.append(0)

            print(f"[DNADataset] Classes per level: {self.classes_per_level}")

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        if self.randomize_offset:
            offset = torch.randint(self.k_mer, (1,)).item()
        else:
            offset = 0
        processed_barcode, att_mask = self.tokenizer(self.barcodes[idx], offset=offset)

        # NEW: Return all 6 labels if hierarchical, else single label
        if self.use_hierarchical and self.all_labels is not None:
            label = torch.tensor(self.all_labels[idx], dtype=torch.int64)  # Shape: (6,)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.int64)  # Shape: ()

        return processed_barcode, label, att_mask


def representations_from_df(df, target_level, model, tokenizer, dataset_name, mode=None, mask_rate=None):

    orders = df["order_name"].to_numpy()
    if dataset_name == "CANADA-1.5M":
        _label_set, y = np.unique(df[target_level], return_inverse=True)
    elif dataset_name == "BIOSCAN-5M":
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

            sum_embeddings = (x * att_mask.unsqueeze(-1)).sum(1)
            sum_mask = att_mask.sum(1, keepdim=True)
            mean_embeddings = sum_embeddings / sum_mask

            dna_embeddings.append(mean_embeddings.cpu().numpy())

    print(f"There are {len(df)} points in the dataset")
    latent = np.array(dna_embeddings)
    latent = np.squeeze(latent, 1)
    print(latent.shape)
    return latent, y, orders