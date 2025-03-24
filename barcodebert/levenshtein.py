"""
Functions for computing Levenshtein distances for training with soft targets.
"""

import numpy as np
import torch

OUTPUT_VOCAB = [b"A", b"C", b"G", b"T"]


def optimized_levenshtein_distance(num1, num2, k):
    mapping_in = ["A", "C", "G", "T", "N"]
    mapping_out = ["A", "C", "G", "T"]
    distance = 0

    for _ in range(k):
        nucleotide1 = mapping_in[num1 % 5]
        if nucleotide1 != "N":
            nucleotide2 = mapping_out[num2 % 4]
            if nucleotide1 != nucleotide2:
                distance += 1
        num1 //= 5
        num2 //= 4

    return distance


def create_optimized_levenshtein_matrix(kmers_int_array, k):
    max_kmer_int = 4**k - 1
    matrix = [[0] * (max_kmer_int + 1) for _ in range(len(kmers_int_array))]

    for i, num1 in enumerate(kmers_int_array):
        for j in range(max_kmer_int + 1):
            matrix[i][j] = optimized_levenshtein_distance(num1, j, k)

    return matrix


def build_lookup_table(k):
    """Build a lookup table for Levenshtein distances considering 'N' in input k-mers."""
    input_vocab_size = 5  # A, C, G, T, N
    output_vocab_size = 4  # A, C, G, T
    input_max_kmer_int = input_vocab_size**k - 1
    output_max_kmer_int = output_vocab_size**k - 1

    # Initialize the lookup table
    lookup_table = torch.zeros((input_max_kmer_int + 1, output_max_kmer_int + 1), dtype=torch.float32)

    # Compute distances for all combinations
    for i in range(input_max_kmer_int + 1):
        for j in range(output_max_kmer_int + 1):
            lookup_table[i, j] = optimized_levenshtein_distance(i, j, k)

    return lookup_table


def softmax_batch_levenshtein_matrices(batch_input_tensor, lookup_table):
    """:)"""

    # Expand the lookup table indices to match the input tensor shape
    batch_size, seq_len = batch_input_tensor.size()
    output_size = lookup_table.size(1)

    # Gather the rows from the lookup table based on the input tensor indices
    # Adjust indices for <MASK> token in input
    adjusted_indices = batch_input_tensor - 1
    output_tensor = lookup_table[adjusted_indices]

    # Ensure the output tensor is of the shape (batch, seq_len, output_size)
    output_tensor = output_tensor.view(batch_size, seq_len, output_size)
    output_tensor = torch.softmax(output_tensor, dim=1)

    return output_tensor


def measure_hamming_distances_wildcard(x, k=3, n_vocab=4):
    """
    Measure distance between vectors and all possible vectors of that length, allowing for wildcard inputs.

    Parameters
    ----------
    x : torch.Tensor shaped [N, S, V]
        Input vector to measure distances to/from.
        Any entries larger than ``n_vocab-1`` will be treated as wild.
    k : int, default=3
        Length of vector.
    n_vocab : int, default=4
        Size of element space to compare against

    Outputs
    -------
    d : torch.Tensor shaped [N, S, n_vocab**k]
        Distance for each possible state.
    """
    # Generate all possible vectors of length k
    y = torch.cartesian_prod(*[torch.arange(n_vocab, dtype=torch.uint8) for i in range(k)])
    # Use a distance of 1 for each element that is not equal
    d = torch.unsqueeze(x, -2) != torch.unsqueeze(torch.unsqueeze(y, 0), 0)
    # Set the distance to 0 for each wildcard element outside the vocabulary
    select = torch.unsqueeze(x, -2) >= n_vocab
    d[select.expand(-1, -1, len(y), -1)] = False
    # Total L1 distance is the sum of distances for each vector element
    d = d.sum(axis=-1).type(torch.FloatTensor)
    return d


def softmax_batch_levenshtein_matrices_vectorized(batch, k, vocab):
    """
    Convert a batch of token indices to a batch of soft target labels.

    Parameters
    ----------
    batch : torch.Tensor of ints shaped [N, S]
        Batch of token indices.
    k : int
        Length of tokenized string.
    vocab : torchtext.vocab.Vocab
        Vocabulary for the tokens.

    Outputs
    -------
    w : torch.Tensor shaped [N, S, n_vocab**k]
        Soft labels for each token.
    """
    # Map from token index to the input string that is represented
    # Need a list input for lookup_tokens; this needs to be flattened too
    batch_l = list(batch.reshape(-1).cpu().numpy())
    v = vocab.lookup_tokens(batch_l)  # list of strings
    # Convert list of strings to 2D numpy array of bytes
    char_array = np.fromiter("".join(v), dtype="S1").reshape(len(v), -1)
    # Convert vocab letters into indices to simplify distance measurement
    encoded_x = 4 * np.ones((len(v), k), dtype=np.int8)
    for i, letter in enumerate(OUTPUT_VOCAB):
        encoded_x[char_array == letter] = i
    # Undo the flattening step
    encoded_x = torch.tensor(encoded_x.reshape(*batch.shape, k))  # [b, len, k]
    # Measure distances for each element in the output space
    d = measure_hamming_distances_wildcard(encoded_x, k=k, n_vocab=len(OUTPUT_VOCAB))
    # Use softmax to determine the weights for each output
    w = torch.softmax(-d, dim=-1)
    return w
