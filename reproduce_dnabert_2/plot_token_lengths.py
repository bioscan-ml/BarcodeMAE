"""
Tokenize sequences from train.txt using DNABERT-2 tokenizer and plot length distribution.
Saves an updated plot every CHECKPOINT_EVERY sequences so you can check progress mid-run.
"""

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

DATA_PATH = "/home/m4safari/projects/def-lila-ab/m4safari/shards_data/BarcodeMAE/reproduce_dnabert_2/reproduce_dnabert_2/data/train.txt"
THRESHOLD = 128
CHECKPOINT_EVERY = 1_000_000  # save a new plot every 1M sequences
TEST_MODE = False  # set to True to run on first 10 sequences only


def save_plot(lengths, threshold, total_so_far):
    lengths_arr = np.array(lengths)
    n_over = (lengths_arr > threshold).sum()
    pct_over = 100 * n_over / len(lengths_arr)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths_arr, bins=80, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(threshold, color="red", linewidth=2, linestyle="--", label=f"Length = {threshold}")
    ax.text(
        threshold + 2, ax.get_ylim()[1] * 0.9,
        f"{pct_over:.1f}% > {threshold}",
        color="red", fontsize=11, va="top",
    )
    ax.set_xlabel("Token length after tokenization", fontsize=12)
    ax.set_ylabel("Number of sequences", fontsize=12)
    ax.set_title(f"DNABERT-2 token length distribution — {total_so_far:,} sequences so far", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("token_length_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  [checkpoint] plot saved ({total_so_far:,} seqs | {pct_over:.1f}% > {threshold})")


tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

if TEST_MODE:
    print("TEST MODE: reading first 10 sequences only...")
else:
    print("Tokenizing sequences line by line...")

lengths = []
with open(DATA_PATH) as f:
    for i, line in enumerate(tqdm(f)):
        if TEST_MODE and i >= 10:
            break
        seq = line.strip()
        if not seq:
            continue
        tokens = tokenizer(seq, add_special_tokens=True)["input_ids"]
        if TEST_MODE:
            print(f"  seq {i+1}: {seq[:40]}... → {len(tokens)} tokens")
        lengths.append(len(tokens))

        if not TEST_MODE and len(lengths) % CHECKPOINT_EVERY == 0:
            save_plot(lengths, THRESHOLD, len(lengths))

# Final plot
save_plot(lengths, THRESHOLD, len(lengths))

lengths_arr = np.array(lengths)
print(f"\nTotal sequences:          {len(lengths_arr):,}")
print(f"Sequences longer than {THRESHOLD}: {(lengths_arr > THRESHOLD).sum():,} ({100*(lengths_arr > THRESHOLD).mean():.1f}%)")
print(f"Max length:               {lengths_arr.max()}")
print(f"Mean length:              {lengths_arr.mean():.1f}")
print(f"Median length:            {np.median(lengths_arr):.1f}")
print("\nFinal plot saved to token_length_distribution.png")