#!/usr/bin/env python3
"""
Precompute k-mer substitution lookup tables that match DNADataset tokenization exactly.

This version ensures token ID compatibility with your KmerTokenizer.
"""

import argparse
import numpy as np
import pickle
import os
from itertools import combinations, product
import hashlib
import time


def create_compatible_vocab(k_mer_size, tokenize_n_nucleotide=False):
    """
    Create vocab that exactly matches DNADataset tokenization

    Returns:
        tuple: (kmer_to_token_dict, token_to_kmer_dict, n_special_tokens)
    """
    # Exactly match DNADataset logic
    base_pairs = "ACGT"
    special_tokens = ["[MASK]", "[UNK]"]

    if tokenize_n_nucleotide:
        base_pairs += "N"

    # Generate k-mers exactly like DNADataset
    kmers = ["".join(kmer) for kmer in product(base_pairs, repeat=k_mer_size)]

    # Handle N-containing k-mers exactly like DNADataset
    if tokenize_n_nucleotide:
        prediction_kmers = []
        other_kmers = []
        for kmer in kmers:
            if "N" in kmer:
                other_kmers.append(kmer)
            else:
                prediction_kmers.append(kmer)
        kmers = prediction_kmers + other_kmers

    # Create token mappings (special tokens first, then k-mers)
    all_tokens = special_tokens + kmers

    kmer_to_token = {}
    token_to_kmer = {}

    for token_id, token in enumerate(all_tokens):
        if token not in special_tokens:  # Only map k-mers
            kmer_to_token[token] = token_id
            token_to_kmer[token_id] = token

    n_special_tokens = len(special_tokens)

    print(f"Created vocab with {len(special_tokens)} special tokens + {len(kmers)} k-mers")
    print(f"Special tokens: {special_tokens}")
    print(f"Token range for k-mers: {n_special_tokens} to {len(all_tokens) - 1}")

    return kmer_to_token, token_to_kmer, n_special_tokens


def kmer_to_sequence(kmer):
    """Convert k-mer string to sequence of base indices"""
    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    return [base_to_idx[base] for base in kmer]


def sequence_to_kmer(sequence):
    """Convert sequence of base indices to k-mer string"""
    idx_to_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
    return ''.join(idx_to_base[idx] for idx in sequence)


def compute_substitution_probability(original_seq, target_seq, substitution_matrix):
    """Compute probability of substituting original_seq -> target_seq"""
    prob = 1.0
    for i in range(len(original_seq)):
        if original_seq[i] != target_seq[i]:
            # Mutation occurred at position i
            if original_seq[i] < len(substitution_matrix) and target_seq[i] < len(substitution_matrix):
                prob *= substitution_matrix[original_seq[i], target_seq[i]]
            else:
                # Handle N base (skip or assign low probability)
                prob *= 0.01  # Low probability for N substitutions
    return prob


def generate_mutations_recursive(original_seq, positions, pos_idx, current_seq,
                                 substitutions, substitution_matrix, exclude_n=True):
    """Recursively generate all mutation combinations"""
    if pos_idx == len(positions):
        # Check if this is actually different from original
        if current_seq != original_seq:
            target_kmer = sequence_to_kmer(current_seq)
            prob = compute_substitution_probability(original_seq, current_seq, substitution_matrix)
            substitutions.append((target_kmer, prob))
        return

    current_pos = positions[pos_idx]
    original_base = original_seq[current_pos]

    # Determine valid bases for substitution
    n_bases = 4 if exclude_n else 5  # A, C, G, T (and possibly N)

    # Try all possible substitutions at this position (except original)
    for new_base in range(n_bases):
        if new_base != original_base:
            current_seq[current_pos] = new_base
            generate_mutations_recursive(
                original_seq, positions, pos_idx + 1, current_seq.copy(),
                substitutions, substitution_matrix, exclude_n
            )

    # Restore original base
    current_seq[current_pos] = original_base


def generate_all_substitutions_for_kmer(kmer, substitution_matrix, tokenize_n_nucleotide=False):
    """Generate all possible substitutions for a given k-mer (excluding itself)"""
    original_seq = kmer_to_sequence(kmer)
    k_mer_size = len(kmer)
    substitutions = []

    # Skip k-mers with N if not handling N
    if not tokenize_n_nucleotide and 'N' in kmer:
        return []

    # For each possible number of mutations (1 to k)
    for num_mutations in range(1, k_mer_size + 1):
        # For each combination of positions to mutate
        for positions in combinations(range(k_mer_size), num_mutations):
            # Generate all possible mutations for these positions
            generate_mutations_recursive(
                original_seq, list(positions), 0, original_seq.copy(),
                substitutions, substitution_matrix, exclude_n=not tokenize_n_nucleotide
            )

    return substitutions


def precompute_kmer_substitutions_compatible(k_mer_size, substitution_matrix, tokenize_n_nucleotide=False):
    """
    Precompute k-mer substitutions with exact compatibility to DNADataset

    Returns:
        dict: {original_token_id: [(target_token_id, cumulative_prob), ...]}
    """
    # Create compatible vocabulary
    kmer_to_token, token_to_kmer, n_special_tokens = create_compatible_vocab(
        k_mer_size, tokenize_n_nucleotide
    )

    cumulative_lookup = {}

    print(f"Precomputing substitutions for {len(token_to_kmer)} k-mers...")
    start_time = time.time()

    processed = 0
    for token_id, kmer in token_to_kmer.items():
        if processed % 500 == 0:
            elapsed = time.time() - start_time
            progress = processed / len(token_to_kmer)
            eta = elapsed / max(progress, 0.001) - elapsed if progress > 0 else 0
            print(f"Progress: {processed}/{len(token_to_kmer)} ({progress * 100:.1f}%) - "
                  f"ETA: {eta / 60:.1f} minutes")

        substitutions = generate_all_substitutions_for_kmer(
            kmer, substitution_matrix, tokenize_n_nucleotide
        )

        if substitutions:
            # Convert k-mer strings to token IDs and filter valid ones
            valid_substitutions = []
            for target_kmer, prob in substitutions:
                if target_kmer in kmer_to_token:
                    target_token_id = kmer_to_token[target_kmer]
                    # Ensure we don't include the original token
                    if target_token_id != token_id:
                        valid_substitutions.append((target_token_id, prob))

            if valid_substitutions:
                # Sort by probability (descending)
                valid_substitutions.sort(key=lambda x: x[1], reverse=True)

                # Compute cumulative probabilities
                total_prob = sum(prob for _, prob in valid_substitutions)
                cumulative_probs = []
                cumulative_sum = 0.0

                for target_token, prob in valid_substitutions:
                    cumulative_sum += prob / total_prob  # Normalize
                    cumulative_probs.append((target_token, cumulative_sum))

                cumulative_lookup[token_id] = cumulative_probs

        processed += 1

    elapsed = time.time() - start_time
    print(f"Precomputation completed in {elapsed / 60:.1f} minutes")
    print(f"Generated lookup table for {len(cumulative_lookup)} k-mers")

    return cumulative_lookup, n_special_tokens


def save_compatible_lookup_table(lookup_table, k_mer_size, substitution_matrix,
                                 n_special_tokens, tokenize_n_nucleotide, output_path):
    """Save precomputed lookup table with compatibility info"""
    data = {
        'k_mer_size': k_mer_size,
        'substitution_matrix': substitution_matrix,
        'cumulative_lookup': lookup_table,
        'n_special_tokens': n_special_tokens,
        'tokenize_n_nucleotide': tokenize_n_nucleotide,
        'special_tokens': ["[MASK]", "[UNK]"],
        'base_pairs': "ACGTN" if tokenize_n_nucleotide else "ACGT",
        'compatibility_version': 'DNADataset_v1'
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    # Print file size and verification
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"Compatible lookup table saved to {output_path}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Special tokens: {data['special_tokens']}")
    print(f"K-mer token range: {n_special_tokens} to {max(lookup_table.keys())}")


def create_substitution_matrix(transition_bias=0.6):
    """Create a biologically plausible substitution matrix"""
    matrix = np.array([
        [0.1, 0.15, transition_bias, 0.15],  # A -> A, C, G, T
        [0.15, 0.1, 0.15, transition_bias],  # C -> A, C, G, T
        [transition_bias, 0.15, 0.1, 0.15],  # G -> A, C, G, T
        [0.15, transition_bias, 0.15, 0.1]  # T -> A, C, G, T
    ])
    return matrix / matrix.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute k-mer substitutions compatible with DNADataset tokenization"
    )
    parser.add_argument(
        "--k-mer", type=int, default=6,
        help="Size of k-mers (default: 6)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="masking_codes/kmer_cache",
        help="Directory to save lookup table (default: ./kmer_cache)"
    )
    parser.add_argument(
        "--transition-bias", type=float, default=0.6,
        help="Transition vs transversion bias (default: 0.6)"
    )
    parser.add_argument(
        "--custom-matrix", type=str, default="masking_codes/lepidoptera_matrix.npy",
        help="Path to custom substitution matrix (.npy file)"
    )
    parser.add_argument(
        "--tokenize-n-nucleotide", action="store_true",
        help="Include N-containing k-mers (matches DNADataset tokenize_n_nucleotide=True)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create or load substitution matrix
    if args.custom_matrix:
        print(f"Loading custom substitution matrix from {args.custom_matrix}")
        substitution_matrix = np.load(args.custom_matrix)
    else:
        print(f"Creating substitution matrix with transition bias {args.transition_bias}")
        substitution_matrix = create_substitution_matrix(args.transition_bias)

    print("Substitution matrix:")
    print("    A     C     G     T")
    base_names = ['A', 'C', 'G', 'T']
    for i, row in enumerate(substitution_matrix):
        print(f"{base_names[i]} {' '.join(f'{x:.3f}' for x in row)}")
    print()

    # Create filename with compatibility info
    n_flag = "_with_n" if args.tokenize_n_nucleotide else ""
    param_str = f"k{args.k_mer}{n_flag}_" + str(substitution_matrix.flatten().tolist())
    cache_key = hashlib.md5(param_str.encode()).hexdigest()[:16]
    output_path = os.path.join(
        args.output_dir,
        f"kmer_substitutions_k{args.k_mer}{n_flag}_compatible_{cache_key}.pkl"
    )

    # Check if already exists
    if os.path.exists(output_path):
        print(f"Compatible lookup table already exists at {output_path}")
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {file_size:.1f} MB")
        return

    # Precompute with compatibility
    lookup_table, n_special_tokens = precompute_kmer_substitutions_compatible(
        args.k_mer, substitution_matrix, args.tokenize_n_nucleotide
    )

    # Save with compatibility metadata
    save_compatible_lookup_table(
        lookup_table, args.k_mer, substitution_matrix,
        n_special_tokens, args.tokenize_n_nucleotide, output_path
    )

    print(f"\nDone! This lookup table is compatible with DNADataset tokenization.")
    print(f"Use with: BiologicalMasker.from_file('{output_path}')")
    print(f"Make sure to use tokenize_n_nucleotide={args.tokenize_n_nucleotide} in your training!")


if __name__ == "__main__":
    main()