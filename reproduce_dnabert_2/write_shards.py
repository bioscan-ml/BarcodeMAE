import csv
import io
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import webdataset as wds
from tqdm import tqdm
from transformers import AutoTokenizer

MAX_LENGTH = 224

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
tokenizer.model_max_length = MAX_LENGTH

tokenized_seq = tokenizer("ACGT", padding='max_length', truncation=True, max_length=MAX_LENGTH)['input_ids']
print(len(tokenized_seq))

# Suppress Hugging Face tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global labels array shared across worker processes via Pool initializer
_labels_array = None


def _init_worker(labels_arr):
    global _labels_array
    _labels_array = labels_arr


def tokenize(seq):
    tokens, _, att_mask = tokenizer(seq, padding='max_length', truncation=True, max_length=MAX_LENGTH).values()
    tokens_np = np.array(tokens).astype(np.uint16)
    att_mask_np = np.array(att_mask).astype(np.uint8)
    return tokens_np, att_mask_np


def npy_to_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def process_line(args):
    seq, split_name, idx, seq_id = args
    global _labels_array
    label = -1
    if _labels_array is not None and 0 < seq_id < len(_labels_array):
        label = int(_labels_array[seq_id])

    tokens, mask = tokenize(seq.strip())
    key = f"{split_name}-{idx:09d}"
    return {
        "__key__": key,
        "tokens": npy_to_bytes(tokens),
        "attention_mask": npy_to_bytes(mask),
        "label": npy_to_bytes(np.array([label], dtype=np.int32)),
    }


def load_labels(tsv_file, confidence_threshold=50.0):
    """
    Load Kraken2 species assignments, keeping only high-confidence classified rows.

    SEQ IDs in the TSV are 1-based (SEQ_1 = train.txt line 0).
    Returns a numpy int32 array indexed by SEQ number (index 0 unused);
    value is species index or -1 for unclassified / below threshold.
    Also returns the species_name -> int index mapping dict.
    """
    print(f"Loading labels from {tsv_file} (threshold >= {confidence_threshold}%)...")
    species_to_idx = {}
    rows = []

    with open(tsv_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["status"] != "C":
                continue
            if float(row["kmer_confidence_pct"]) < confidence_threshold:
                continue
            species = row["species_name"]
            if species not in species_to_idx:
                species_to_idx[species] = len(species_to_idx)
            seq_id = int(row["sequence_id"].replace("SEQ_", ""))
            rows.append((seq_id, species_to_idx[species]))

    if not rows:
        print("WARNING: No high-confidence labels found.")
        return np.array([-1], dtype=np.int32), {}

    max_seq_id = max(r[0] for r in rows)
    labels_array = np.full(max_seq_id + 1, -1, dtype=np.int32)
    for seq_id, label_idx in rows:
        labels_array[seq_id] = label_idx

    n_labeled = int((labels_array >= 0).sum())
    print(f"Loaded {n_labeled:,} labeled sequences, {len(species_to_idx):,} unique species.")
    return labels_array, species_to_idx


def write_shards(
    input_file,
    out_dir,
    split_name,
    labels_array=None,
    seq_id_offset=1,
    target_shard_size_bytes=1e9,
    max_samples_per_shard=300000,
    num_workers=8,
    ratio=1.0,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pattern = os.path.join(out_dir, f"{split_name}-%06d.tar")
    sink = wds.ShardWriter(pattern, maxsize=target_shard_size_bytes, maxcount=max_samples_per_shard)

    with open(input_file) as f:
        lines = [
            (line, split_name, idx, seq_id_offset + idx)
            for idx, line in enumerate(f)
            if random.random() <= ratio
        ]

    with Pool(num_workers, initializer=_init_worker, initargs=(labels_array,)) as pool, \
         tqdm(total=len(lines), desc=f"Writing {split_name} shards") as pbar:
        for record in pool.imap(process_line, lines, chunksize=256):
            sink.write(record)
            pbar.update(1)

    sink.close()
    print(f"Wrote {len(lines)} samples to shards at {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
cpu_count = os.cpu_count()
print(f"Number of available CPU cores: {cpu_count}")

CONFIDENCE_THRESHOLD = 0.0  # include all classified sequences regardless of confidence
RATIO = 0.01

BASE_DATA_DIR = "/home/m4safari/projects/def-lila-ab/m4safari/shards_data/BarcodeMAE/reproduce_dnabert_2/reproduce_dnabert_2"
TSV_FILE = os.path.join(BASE_DATA_DIR, "dnabert_2_kraken_species_assignments.tsv")
TRAIN_FILE = os.path.join(BASE_DATA_DIR, "data/train.txt")
DEV_FILE = os.path.join(BASE_DATA_DIR, "data/dev.txt")
output_dir = f"/scratch/m4safari/dnabert2_wds/shards_{RATIO}"

# Load species labels from TSV (train sequences only; SEQ_1 = line 0)
labels_array, species_to_idx = load_labels(TSV_FILE, confidence_threshold=CONFIDENCE_THRESHOLD)

# Save species vocabulary for downstream training/finetuning
Path(output_dir).mkdir(parents=True, exist_ok=True)
vocab_path = os.path.join(output_dir, "species_vocab.json")
with open(vocab_path, "w") as f:
    json.dump(species_to_idx, f, indent=2)
print(f"Saved species vocab ({len(species_to_idx):,} species) to {vocab_path}")

# Dev shards: no labels (TSV covers train only)
write_shards(DEV_FILE, output_dir, "dev", labels_array=None, ratio=RATIO)

# Train shards: SEQ_1 = line 0 → offset=1 for 1-based SEQ IDs
write_shards(
    TRAIN_FILE,
    output_dir,
    "train",
    labels_array=labels_array,
    seq_id_offset=1,
    max_samples_per_shard=600000,
    num_workers=8,
    ratio=RATIO,
)