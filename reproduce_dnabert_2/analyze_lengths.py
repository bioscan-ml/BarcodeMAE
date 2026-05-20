import webdataset as wds
import numpy as np
import io
import glob
import matplotlib.pyplot as plt
import os

def npy_decoder(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)

from transformers import AutoTokenizer

def analyze_raw_file(file_path):
    print(f"Analyzing raw file: {file_path}")
    print("Loading tokenizer (zhihan1996/DNABERT-2-117M)...")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    
    lengths = []
    MAX_TOKENS = 0
    
    with open(file_path, 'r') as f:
        # Use tqdm to show progress if possible, assuming file has lines
        # Or just stream
        for i, line in enumerate(f):
            if not line.strip(): continue
            
            # Tokenize without padding or truncation to see true length
            token_ids = tokenizer.encode(line.strip(), add_special_tokens=False)
            # Add +1 for potential SEP if packed, or existing special tokens logic
            # Standard BERT has 2 special tokens [CLS] [SEP] usually.
            # DNABERT-2 tokenizer behavior:
            # If we pack, we add [SEP] manually.
            # If we don't pack, we add [CLS] [SEP] via tokenizer.
            # Let's measure pure sequence length in tokens.
            lengths.append(len(token_ids))
            
            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} lines...")
                
            # Limit total reads if file is huge for quick check
            if i > 100000:
                print("Stopping early after 100k samples for speed.")
                break
    
    lengths = np.array(lengths)
    if len(lengths) == 0:
        print("File is empty.")
        return
        
    print_stats(lengths)

def print_stats(lengths):
    print("\n--- Length Analysis ---")
    print(f"Total sequences analyzed: {len(lengths)}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths)}")
    
    # Check heuristic: > 50% sequences < 128
    THRESHOLD = 127 # < 128 means <= 127
    count_below = np.sum(lengths < 128)
    percent_below = (count_below / len(lengths)) * 100
    
    print(f"\nSequences < 128 tokens: {count_below} ({percent_below:.2f}%)")
    
    if percent_below > 50:
        print(f"\n✅ SUCCESS: {percent_below:.1f}% of sequences are shorter than 128. Packing is HIGHLY recommended!")
    else:
        print(f"\n⚠️ CAUTION: Only {percent_below:.1f}% of sequences are shorter than 128. Packing might yield variable returns.")

    # Percentiles
    print("\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(lengths, p)}")

def analyze_shards(shard_pattern):
    print(f"Analyzing shards matching: {shard_pattern}")
    files = sorted(glob.glob(shard_pattern))
    if not files:
        print(f"No shards found matching {shard_pattern}")
        # Fallback to analyzing raw text file if available
        raw_train_file = "data/train.txt"
        if os.path.exists(raw_train_file):
            analyze_raw_file(raw_train_file)
        else:
            print(f"Also checked {raw_train_file} but it does not exist.")
        return

    # Remove .decode() to get raw bytes, so we can use our custom npy_decoder
    dataset = wds.WebDataset(files).to_tuple("attention_mask")
    
    lengths = []
    
    for i, (mask_bytes,) in enumerate(dataset):
        # mask is stored as bytes containing npy
        # We need to wrap bytes in BytesIO
        try:
             mask = np.load(io.BytesIO(mask_bytes), allow_pickle=False)
        except Exception as e:
            print(f"Error decoding mask at index {i}: {e}")
            continue
        # Length is sum of attention mask (1s)
        length = np.sum(mask)
        lengths.append(length)
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} sequences...")

    lengths = np.array(lengths)
    
    if len(lengths) == 0:
        print("No sequences found in shards.")
        return

    print_stats(lengths)

if __name__ == "__main__":
    # Check if shards directory exists
    if not os.path.exists("shards"):
        print("Directory 'shards' does not exist.")
    else:
        # Analyze train shards. Use 'shards/train-*.tar'
        # Adjust pattern if your shards represent the data differently
        analyze_shards("shards/train-*.tar")
