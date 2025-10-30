#!/usr/bin/env python3
"""
Precompute k-mer substitution lookup tables for biological masking.

Run this script once to generate the lookup table, then use it in training.

Usage:
    python precompute_kmer_substitutions.py --k-mer 6 --output-dir ./kmer_cache
"""

import argparse
import numpy as np
import pickle
import os
from itertools import combinations
import hashlib
import time


def token_to_sequence(token_id, k_mer_size, n_bases=4):
    """Convert token ID to nucleotide sequence"""
    sequence = []
    temp_id = token_id
    for _ in range(k_mer_size):
        sequence.append(temp_id % n_bases)
        temp_id //= n_bases
    return sequence[::-1]  # Reverse to get correct order


def sequence_to_token(sequence, n_bases=4):
    """Convert nucleotide sequence to token ID"""
    token_id = 0
    for base_idx in sequence:
        token_id = token_id * n_bases + base_idx
    return token_id


def compute_substitution_probability(original_seq, target_seq, substitution_matrix):
    """Compute probability of substituting original_seq -> target_seq"""
    prob = 1.0
    for i in range(len(original_seq)):
        if original_seq[i] != target_seq[i]:
            # Mutation occurred at position i
            prob *= substitution_matrix[original_seq[i], target_seq[i]]
    return prob


def generate_mutations_recursive(original_seq, positions, pos_idx, current_seq,
                                 substitutions, substitution_matrix, n_bases=4):
    """Recursively generate all mutation combinations"""
    if pos_idx == len(positions):
        # Check if this is actually different from original
        if current_seq != original_seq:
            target_token = sequence_to_token(current_seq, n_bases)
            prob = compute_substitution_probability(original_seq, current_seq, substitution_matrix)
            substitutions.append((target_token, prob))
        return

    current_pos = positions[pos_idx]
    original_base = original_seq[current_pos]

    # Try all possible substitutions at this position (except original)
    for new_base in range(n_bases):
        if new_base != original_base:
            current_seq[current_pos] = new_base
            generate_mutations_recursive(
                original_seq, positions, pos_idx + 1, current_seq.copy(),
                substitutions, substitution_matrix, n_bases
            )

    # Restore original base
    current_seq[current_pos] = original_base


def generate_all_substitutions_for_kmer(token_id, k_mer_size, substitution_matrix, n_bases=4):
    """Generate all possible substitutions for a given k-mer token (excluding itself)"""
    original_seq = token_to_sequence(token_id, k_mer_size, n_bases)
    substitutions = []

    # For each possible number of mutations (1 to k)
    for num_mutations in range(1, k_mer_size + 1):
        # For each combination of positions to mutate
        for positions in combinations(range(k_mer_size), num_mutations):
            # Generate all possible mutations for these positions
            generate_mutations_recursive(
                original_seq, list(positions), 0, original_seq.copy(),
                substitutions, substitution_matrix, n_bases
            )

    # Explicitly filter out the original k-mer (extra safety check)
    original_token_id = token_id
    substitutions = [(target_token, prob) for target_token, prob in substitutions
                     if target_token != original_token_id]

    return substitutions


def precompute_kmer_substitutions(k_mer_size, substitution_matrix, n_bases=4):
    """
    Precompute all k-mer substitutions and their probabilities

    Returns:
        dict: {original_token_id: [(target_token_id, cumulative_prob), ...]}
    """
    total_kmers = n_bases ** k_mer_size
    cumulative_lookup = {}

    print(f"Precomputing substitutions for {total_kmers} k-mers of size {k_mer_size}...")
    start_time = time.time()

    for token_id in range(total_kmers):
        if token_id % 500 == 0:
            elapsed = time.time() - start_time
            progress = token_id / total_kmers
            eta = elapsed / max(progress, 0.001) - elapsed
            print(f"Progress: {token_id}/{total_kmers} ({progress * 100:.1f}%) - "
                  f"ETA: {eta / 60:.1f} minutes")

        substitutions = generate_all_substitutions_for_kmer(
            token_id, k_mer_size, substitution_matrix, n_bases
        )

        if substitutions:
            # Sort by probability (descending) for better sampling
            substitutions.sort(key=lambda x: x[1], reverse=True)

            # Compute cumulative probabilities for fast sampling
            total_prob = sum(prob for _, prob in substitutions)
            cumulative_probs = []
            cumulative_sum = 0.0

            for target_token, prob in substitutions:
                cumulative_sum += prob / total_prob  # Normalize
                cumulative_probs.append((target_token, cumulative_sum))

            cumulative_lookup[token_id] = cumulative_probs

    elapsed = time.time() - start_time
    print(f"Precomputation completed in {elapsed / 60:.1f} minutes")
    print(f"Generated lookup table for {len(cumulative_lookup)} k-mers")

    return cumulative_lookup


def save_lookup_table(lookup_table, k_mer_size, substitution_matrix, output_path):
    """Save precomputed lookup table to disk"""
    data = {
        'k_mer_size': k_mer_size,
        'substitution_matrix': substitution_matrix,
        'cumulative_lookup': lookup_table,
        'n_bases': 4
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"Lookup table saved to {output_path}")
    print(f"File size: {file_size:.1f} MB")


def create_substitution_matrix(transition_bias=0.6):
    """
    Create a biologically plausible substitution matrix

    Args:
        transition_bias: Higher values favor transitions (A<->G, C<->T) over transversions

    Returns:
        4x4 normalized substitution matrix
    """
    # Create matrix favoring transitions over transversions
    matrix = np.array([
        [0.1, 0.15, transition_bias, 0.15],  # A -> A, C, G, T (A->G is transition)
        [0.15, 0.1, 0.15, transition_bias],  # C -> A, C, G, T (C->T is transition)
        [transition_bias, 0.15, 0.1, 0.15],  # G -> A, C, G, T (G->A is transition)
        [0.15, transition_bias, 0.15, 0.1]  # T -> A, C, G, T (T->C is transition)
    ])

    # Normalize rows to sum to 1
    return matrix / matrix.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute k-mer substitution lookup tables for biological masking"
    )
    parser.add_argument(
        "--k-mer", type=int, default=6,
        help="Size of k-mers (default: 6)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./kmer_cache",
        help="Directory to save lookup table (default: ./kmer_cache)"
    )
    parser.add_argument(
        "--transition-bias", type=float, default=0.6,
        help="Transition vs transversion bias (default: 0.6)"
    )
    parser.add_argument(
        "--custom-matrix", type=str, default="lepidoptera_matrix.npy",
        help="Path to custom substitution matrix (.npy file)"
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

    # Create filename based on parameters
    param_str = f"k{args.k_mer}_" + str(substitution_matrix.flatten().tolist())
    cache_key = hashlib.md5(param_str.encode()).hexdigest()[:16]
    output_path = os.path.join(args.output_dir, f"kmer_substitutions_k{args.k_mer}_{cache_key}.pkl")

    # Check if already exists
    if os.path.exists(output_path):
        print(f"Lookup table already exists at {output_path}")
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {file_size:.1f} MB")
        return

    # Precompute
    lookup_table = precompute_kmer_substitutions(args.k_mer, substitution_matrix)

    # Save
    save_lookup_table(lookup_table, args.k_mer, substitution_matrix, output_path)

    print(f"\nDone! Use this file in your training with:")
    print(f"biological_masker = BiologicalMasker.from_file('{output_path}')")


if __name__ == "__main__":
    main()