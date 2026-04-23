"""
Datasets.
"""

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
        use_cls_token=False,  # Whether to prepend [CLS] token to sequences
        return_bin_labels=False,  # Return BIN labels for BIOSCAN-5M (augments pair creation)
        return_family_labels=False,  # Return family labels for BIOSCAN-5M (augments pair creation)
        filter_unknown_labels=False,  # ITS-5M finetuning: drop samples whose target label is unknown
    ):
        self.k_mer = k_mer
        self.stride = k_mer if stride is None else stride
        self.max_len = max_len
        self.randomize_offset = randomize_offset
        self.use_cls_token = use_cls_token
        self.barcodes = []
        self.tax_encoder = None
        self.labels = []
        self.taxonomy_labels = []  # For any taxonomic level
        self.bin_labels = []  # BIN labels for BIOSCAN-5M
        self.family_labels = []  # Family labels for BIOSCAN-5M
        self.label2id = label2id

        # Handle backward compatibility: return_genus -> return_taxonomy_level
        if return_genus and return_taxonomy_level is None:
            return_taxonomy_level = "genus"
        self.return_taxonomy_level = return_taxonomy_level
        self.return_bin_labels = return_bin_labels
        self.return_family_labels = return_family_labels

        # Check that the dataframe contains a valid format
        if dataset_format not in ["CANADA-1.5M", "BIOSCAN-5M", "ITS-5M"]:
            raise NotImplementedError(f"Dataset {dataset_format} not supported.")

        if tokenizer == "kmer":
            # Vocabulary
            base_pairs = "ACGT"
            if use_cls_token:
                self.special_tokens = ["[MASK]", "[UNK]", "[CLS]"]  # Token IDs: 0=[MASK], 1=[UNK], 2=[CLS]
                self.CLS_TOKEN_ID = 2  # [CLS] is at index 2
            else:
                self.special_tokens = ["[MASK]", "[UNK]"]

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
                    # print(f"Unique taxonomy categories: {self.taxonomy_label_set}")

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
                    # print(f"Unique taxonomy categories: {self.taxonomy_label_set.tolist()}")

                else:
                    print(f"Warning: Column '{taxonomy_column}' not found. Using dummy labels.")
                    self.taxonomy_labels = [0] * len(self.labels)
            else:
                self.taxonomy_labels = [0] * len(self.labels)  # Dummy labels

            # Load BIN labels if requested (BIOSCAN-5M only)
            if self.return_bin_labels:
                if "dna_bin_index" in df.columns:
                    # BIN labels are already integers; use -1 for missing
                    self.bin_labels = df["dna_bin_index"].fillna(-1).astype(int).tolist()
                    num_valid_bin = sum(1 for b in self.bin_labels if b >= 0)
                    print(f"BIN labels: {len(self.bin_labels)} total, {num_valid_bin} valid")
                else:
                    print("Warning: Column 'dna_bin_index' not found. Using dummy BIN labels.")
                    self.bin_labels = [-1] * len(self.labels)
            else:
                self.bin_labels = [-1] * len(self.labels)  # Dummy labels

            # Load family labels if requested (BIOSCAN-5M only)
            if self.return_family_labels:
                if "family_index" in df.columns:
                    # Family labels are already integers; use -1 for missing
                    self.family_labels = df["family_index"].fillna(-1).astype(int).tolist()
                    num_valid_family = sum(1 for f in self.family_labels if f >= 0)
                    print(f"Family labels: {len(self.family_labels)} total, {num_valid_family} valid")
                else:
                    print("Warning: Column 'family_index' not found. Using dummy family labels.")
                    self.family_labels = [-1] * len(self.labels)
            else:
                self.family_labels = [-1] * len(self.labels)  # Dummy labels

        elif dataset_format == "ITS-5M":
            labels_file = file_path.replace(".fasta", "_labels.csv")
            if os.path.isfile(labels_file):
                labels_df = pd.read_csv(labels_file)
            else:
                raise FileNotFoundError("Labels file not found for ITS-5M. Expected: " + labels_file)

            if len(labels_df) != len(self.barcodes):
                raise ValueError(
                    f"Mismatch between barcodes ({len(self.barcodes)}) and labels ({len(labels_df)}). "
                    "Ensure the labels CSV matches the FASTA order."
                )

            # Reindex the target taxonomic level to contiguous [0..C-1]; keep 9999999 rows
            # in the dataset (they contribute to pretraining) but map them to -1 so pair
            # creation skips them gracefully.
            raw_labels = labels_df[taxonomic_level].to_numpy()
            known_mask = raw_labels != 9999999
            n_before = len(raw_labels)

            if self.label2id is None:
                known_vals = raw_labels[known_mask]
                factorized, self.label_set = pd.factorize(known_vals, sort=True)
                self.label2id = {lab: i for i, lab in enumerate(self.label_set)}
                self.id2label = {i: lab for lab, i in self.label2id.items()}
            else:
                self.label_set = np.array(sorted(self.label2id.keys()))

            self.num_labels = len(self.label2id)

            # Map every row: known label → contiguous id, unknown → -1
            mapped = np.where(known_mask, np.vectorize(lambda x: self.label2id.get(x, -1))(raw_labels), -1)
            self.labels = mapped.tolist()

            n_unknown = int((~known_mask).sum())
            print(f"[DNADataset][ITS-5M] {n_before} sequences total, "
                  f"{n_unknown} with unknown '{taxonomic_level}' label (mapped to -1), "
                  f"{self.num_labels} known classes.")

            # Populate taxonomy_labels for jumbo taxonomy classification.
            # Uses the same column requested via return_taxonomy_level (typically "genus").
            # Unknown entries (9999999) are mapped to -1 so pair creation skips them.
            if self.return_taxonomy_level:
                tax_col = labels_df[self.return_taxonomy_level].to_numpy()
                tax_known = tax_col != 9999999
                if self.return_taxonomy_level == taxonomic_level:
                    # Already factorized above — reuse the same mapping
                    self.taxonomy_labels = mapped.tolist()
                else:
                    known_tax_vals = tax_col[tax_known]
                    _, tax_label_set = pd.factorize(known_tax_vals, sort=True)
                    tax_label2id = {lab: i for i, lab in enumerate(tax_label_set)}
                    self.taxonomy_labels = [
                        tax_label2id.get(v, -1) if tax_known[i] else -1
                        for i, v in enumerate(tax_col)
                    ]
                n_tax_known = sum(1 for t in self.taxonomy_labels if t >= 0)
                print(f"[DNADataset][ITS-5M] Taxonomy labels ('{self.return_taxonomy_level}'): "
                      f"{n_tax_known} known, {len(self.taxonomy_labels) - n_tax_known} unknown (-1).")
            else:
                self.taxonomy_labels = [0] * len(self.labels)

            # For finetuning: drop samples whose classification target is unknown (-1).
            # Pretraining should NOT set this flag — all sequences contribute to MAE loss.
            if filter_unknown_labels:
                valid = [i for i, lab in enumerate(self.labels) if lab >= 0]
                n_before = len(self.barcodes)
                self.barcodes = [self.barcodes[i] for i in valid]
                self.labels = [self.labels[i] for i in valid]
                self.taxonomy_labels = [self.taxonomy_labels[i] for i in valid]
                print(f"[DNADataset][ITS-5M] filter_unknown_labels: kept {len(self.barcodes):,} / {n_before:,} "
                      f"samples with known '{taxonomic_level}' label.")

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        if self.randomize_offset:
            offset = torch.randint(self.k_mer, (1,)).item()
        else:
            offset = 0
        processed_barcode, att_mask = self.tokenizer(self.barcodes[idx], offset=offset)

        # Prepend [CLS] token if enabled
        if self.use_cls_token:
            cls_token = torch.tensor([self.CLS_TOKEN_ID], dtype=processed_barcode.dtype)
            cls_mask = torch.tensor([1], dtype=att_mask.dtype)
            processed_barcode = torch.cat([cls_token, processed_barcode])
            att_mask = torch.cat([cls_mask, att_mask])

        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        if self.return_taxonomy_level:
            taxonomy_label = torch.tensor(self.taxonomy_labels[idx], dtype=torch.int64)
            # Optionally include BIN and family labels for BIOSCAN-5M pair augmentation
            if self.return_bin_labels and self.return_family_labels:
                bin_label = torch.tensor(self.bin_labels[idx], dtype=torch.int64)
                family_label = torch.tensor(self.family_labels[idx], dtype=torch.int64)
                return processed_barcode, label, att_mask, taxonomy_label, bin_label, family_label
            elif self.return_bin_labels:
                bin_label = torch.tensor(self.bin_labels[idx], dtype=torch.int64)
                return processed_barcode, label, att_mask, taxonomy_label, bin_label
            elif self.return_family_labels:
                family_label = torch.tensor(self.family_labels[idx], dtype=torch.int64)
                return processed_barcode, label, att_mask, taxonomy_label, family_label
            else:
                return processed_barcode, label, att_mask, taxonomy_label
        else:
            return processed_barcode, label, att_mask


def representations_from_df(
    df, target_level, model, tokenizer, dataset_name, mode=None, mask_rate=None, representation_type="tokens",
    use_cls_token=False,
):
    """
    Extract representations from DNA sequences in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing DNA sequences
    target_level : str
        Taxonomic level to use as labels
    model : torch.nn.Module
        Pretrained model
    tokenizer : Tokenizer
        Tokenizer for DNA sequences
    dataset_name : str
        Dataset name (CANADA-1.5M or BIOSCAN-5M)
    mode : str, optional
        Mode (not currently used)
    mask_rate : float, optional
        Mask rate (not currently used)
    representation_type : str, optional
        Type of representation to extract:
        - "tokens": Mean pooling of sequence tokens only (default, backward compatible)
        - "jumbo": Jumbo representation from jumbo CLS tokens (flattened J*D)
        - "jumbo_avg": Average of jumbo tokens only (averaged over J tokens)
        - "all_tokens": Average of ALL tokens (jumbo + sequence tokens)
        - "cls": CLS token representation from position 0
    use_cls_token : bool, optional
        Whether to prepend [CLS] token (ID=2) to the sequence, matching training behaviour.

    Returns
    -------
    latent : np.ndarray
        Latent representations
    y : np.ndarray
        Labels
    orders : np.ndarray
        Order names
    """
    orders = df["order_name"].to_numpy()
    if dataset_name == "CANADA-1.5M":
        _label_set, y = np.unique(df[target_level], return_inverse=True)
    elif dataset_name == "BIOSCAN-5M":
        # _label_set = np.unique(df[target_level])
        y = df[target_level]
    else:
        raise NotImplementedError("Dataset format is not supported. Must be one of CANADA-1.5M or BIOSCAN-5M")

    dna_embeddings = []

    # Validate that the tokenizer vocabulary contains [CLS] when use_cls_token is True
    if use_cls_token and isinstance(tokenizer, KmerTokenizer):
        vocab_tokens = tokenizer.vocabulary_mapper.get_itos()
        if "[CLS]" not in vocab_tokens:
            raise ValueError(
                "use_cls_token=True but the tokenizer vocabulary does not contain '[CLS]'. "
                "Ensure the vocabulary was built with specials=['[MASK]', '[UNK]', '[CLS]']."
            )

    # Get model device robustly (works for all PyTorch models)
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for barcode in df["nucleotides"]:
            x, att_mask = tokenizer(barcode)

            # Prepend [CLS] token (ID=2) if the model was trained with use_cls_token.
            # DNADataset does this in __getitem__; replicate it here without offset.
            if use_cls_token:
                cls_token = torch.tensor([2], dtype=x.dtype)
                cls_mask = torch.tensor([1], dtype=att_mask.dtype)
                x = torch.cat([cls_token, x])
                att_mask = torch.cat([cls_mask, att_mask])

            x = x.unsqueeze(0).to(model_device)
            att_mask = att_mask.unsqueeze(0).to(model_device)

            # Get model output
            output = model(x, att_mask)

            # Extract representation based on type
            if representation_type == "jumbo":
                # Use jumbo representation if available (flattened J*D)
                if hasattr(output, "jumbo_representation"):
                    embedding = output.jumbo_representation  # (batch_size, J*D)
                else:
                    raise ValueError(
                        "Model does not have jumbo_representation. "
                        "Use representation_type='tokens' or use a Jumbo transformer model."
                    )

            elif representation_type == "jumbo_avg":
                # Average of jumbo tokens only
                if hasattr(output, "jumbo_tokens") and output.jumbo_tokens is not None:
                    jumbo_tokens = output.jumbo_tokens  # (batch_size, J, D)
                    embedding = jumbo_tokens.mean(dim=1)  # (batch_size, D)
                else:
                    raise ValueError(
                        "Model does not have jumbo_tokens. "
                        "Use representation_type='tokens' or use a Jumbo transformer model."
                    )

            elif representation_type == "all_tokens":
                # Average of ALL tokens (jumbo + sequence tokens)
                if hasattr(output, "jumbo_tokens") and output.jumbo_tokens is not None:
                    # Model has jumbo tokens - combine jumbo and sequence tokens
                    jumbo_tokens = output.jumbo_tokens  # (batch_size, J, D)

                    # Get sequence tokens
                    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                        hidden_states = output.last_hidden_state
                    elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                        hidden_states = output.hidden_states
                        if isinstance(hidden_states, tuple):
                            hidden_states = hidden_states[-1]
                    else:
                        hidden_states = output[-1] if isinstance(output, tuple) else output
                        if isinstance(hidden_states, tuple):
                            hidden_states = hidden_states[-1]

                    # Concatenate jumbo and sequence tokens
                    all_tokens = torch.cat([jumbo_tokens, hidden_states], dim=1)  # (batch_size, J+seq_len, D)

                    # Create mask for all tokens (jumbo tokens always have mask=1)
                    batch_size, num_jumbo, _ = jumbo_tokens.shape
                    jumbo_mask = torch.ones(batch_size, num_jumbo, device=att_mask.device, dtype=att_mask.dtype)
                    full_mask = torch.cat([jumbo_mask, att_mask], dim=1)  # (batch_size, J+seq_len)

                    # Mean pooling over all tokens
                    sum_embeddings = (all_tokens * full_mask.unsqueeze(-1)).sum(1)
                    sum_mask = full_mask.sum(1, keepdim=True)
                    embedding = sum_embeddings / sum_mask  # (batch_size, D)
                else:
                    # Model doesn't have jumbo tokens - just use sequence tokens
                    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                        hidden_states = output.last_hidden_state
                    elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                        hidden_states = output.hidden_states
                        if isinstance(hidden_states, tuple):
                            hidden_states = hidden_states[-1]
                    else:
                        hidden_states = output[-1] if isinstance(output, tuple) else output
                        if isinstance(hidden_states, tuple):
                            hidden_states = hidden_states[-1]

                    sum_embeddings = (hidden_states * att_mask.unsqueeze(-1)).sum(1)
                    sum_mask = att_mask.sum(1, keepdim=True)
                    embedding = sum_embeddings / sum_mask

            elif representation_type == "cls":
                # CLS token representation from position 0
                if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                    hidden_states = output.last_hidden_state
                elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hidden_states = output.hidden_states
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]
                else:
                    hidden_states = output[-1] if isinstance(output, tuple) else output

                # Extract CLS token at position 0
                embedding = hidden_states[:, 0, :]  # (batch_size, D)

            elif representation_type == "tokens":
                # Mean pooling of k-mer sequence tokens only, excluding the CLS token.
                if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                    hidden_states = output.last_hidden_state
                elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hidden_states = output.hidden_states
                    # output.hidden_states is a tuple of ALL layer outputs when
                    # output_hidden_states=True in BertForTokenClassification.
                    # Take only the last layer (final representations).
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]
                else:
                    hidden_states = output[-1] if isinstance(output, tuple) else output
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]

                # When CLS is at position 0, exclude it from the mean by zeroing its mask entry.
                seq_mask = att_mask.clone()
                if use_cls_token:
                    seq_mask[:, 0] = 0

                sum_embeddings = (hidden_states * seq_mask.unsqueeze(-1)).sum(1)
                sum_mask = seq_mask.sum(1, keepdim=True)
                embedding = sum_embeddings / sum_mask

            elif representation_type == "tokens_with_cls":
                # Mean pooling of all tokens including the CLS token at position 0.
                if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                    hidden_states = output.last_hidden_state
                elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hidden_states = output.hidden_states
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]
                else:
                    hidden_states = output[-1] if isinstance(output, tuple) else output
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[-1]

                sum_embeddings = (hidden_states * att_mask.unsqueeze(-1)).sum(1)
                sum_mask = att_mask.sum(1, keepdim=True)
                embedding = sum_embeddings / sum_mask

            elif representation_type == "tokens_with_registers":
                # Mean of sequence tokens + register tokens, excluding CLS.
                # Requires a RegisterBertModel that exposes register_hidden_states.
                if not hasattr(output, "register_hidden_states") or output.register_hidden_states is None:
                    raise ValueError(
                        "representation_type='tokens_with_registers' requires a model with register tokens "
                        "(RegisterBertModel). Use 'tokens' for a plain BertModel."
                    )
                register_states = output.register_hidden_states  # (B, R, D)
                hidden_states = output.last_hidden_state          # (B, seq_len, D)

                # Exclude CLS (position 0) from sequence tokens
                seq_mask = att_mask.clone()
                if use_cls_token:
                    seq_mask[:, 0] = 0

                # Registers always attend (mask = 1)
                reg_mask = torch.ones(
                    register_states.shape[0], register_states.shape[1],
                    device=att_mask.device, dtype=att_mask.dtype,
                )
                combined = torch.cat([register_states, hidden_states], dim=1)       # (B, R+seq, D)
                combined_mask = torch.cat([reg_mask, seq_mask], dim=1)              # (B, R+seq)
                sum_embeddings = (combined * combined_mask.unsqueeze(-1)).sum(1)
                sum_mask = combined_mask.sum(1, keepdim=True)
                embedding = sum_embeddings / sum_mask

            elif representation_type == "all_with_registers":
                # Mean of all tokens: sequence tokens + register tokens + CLS.
                # Requires a RegisterBertModel that exposes register_hidden_states.
                if not hasattr(output, "register_hidden_states") or output.register_hidden_states is None:
                    raise ValueError(
                        "representation_type='all_with_registers' requires a model with register tokens "
                        "(RegisterBertModel). Use 'tokens_with_cls' for a plain BertModel."
                    )
                register_states = output.register_hidden_states  # (B, R, D)
                hidden_states = output.last_hidden_state          # (B, seq_len, D)

                reg_mask = torch.ones(
                    register_states.shape[0], register_states.shape[1],
                    device=att_mask.device, dtype=att_mask.dtype,
                )
                combined = torch.cat([register_states, hidden_states], dim=1)       # (B, R+seq, D)
                combined_mask = torch.cat([reg_mask, att_mask], dim=1)              # (B, R+seq)
                sum_embeddings = (combined * combined_mask.unsqueeze(-1)).sum(1)
                sum_mask = combined_mask.sum(1, keepdim=True)
                embedding = sum_embeddings / sum_mask

            else:
                raise ValueError(
                    f"Invalid representation_type: {representation_type}. "
                    "Must be one of: 'tokens', 'tokens_only', 'tokens_with_cls', 'jumbo', 'jumbo_avg', 'all_tokens', 'cls', "
                    "'tokens_with_registers', 'all_with_registers'."
                )

            dna_embeddings.append(embedding.cpu().numpy())

    print(f"There are {len(df)} points in the dataset")
    print(f"Using representation type: {representation_type}")
    latent = np.array(dna_embeddings)
    latent = np.squeeze(latent, 1)
    print(f"Representation shape: {latent.shape}")
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
