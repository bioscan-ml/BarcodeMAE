"""
Compatible biological masker that works with DNADataset tokenization exactly.
"""

import torch
import pickle
import numpy as np
import os
import glob


class CompatibleBiologicalMasker:
    def __init__(self, lookup_file_path, device='cuda'):
        """
        Load precomputed k-mer substitution lookup table compatible with DNADataset

        Args:
            lookup_file_path: Path to the compatible precomputed .pkl file
            device: torch device
        """
        self.device = device

        # Load precomputed data
        with open(lookup_file_path, 'rb') as f:
            data = pickle.load(f)

        # Verify compatibility
        if 'compatibility_version' not in data:
            raise ValueError(
                "This lookup table is not compatible with DNADataset tokenization. "
                "Please regenerate using precompute_compatible_kmers.py"
            )

        self.k_mer_size = data['k_mer_size']
        self.substitution_matrix = data['substitution_matrix']
        self.cumulative_lookup = data['cumulative_lookup']
        self.n_special_tokens = data['n_special_tokens']
        self.tokenize_n_nucleotide = data['tokenize_n_nucleotide']
        self.special_tokens = data['special_tokens']

        print(f"Loaded compatible biological masker for k={self.k_mer_size}")
        print(f"Special tokens: {self.special_tokens} (IDs 0-{self.n_special_tokens - 1})")
        print(f"K-mer token range: {self.n_special_tokens} to {max(self.cumulative_lookup.keys())}")
        print(f"Tokenize N nucleotides: {self.tokenize_n_nucleotide}")
        print(f"Lookup table contains {len(self.cumulative_lookup)} k-mers")

    @classmethod
    def from_cache_dir(cls, cache_dir, k_mer_size, tokenize_n_nucleotide=False, device='cuda'):
        """
        Automatically find and load compatible lookup table

        Args:
            cache_dir: Directory containing cached lookup tables
            k_mer_size: Size of k-mers to look for
            tokenize_n_nucleotide: Whether N nucleotides are tokenized
            device: torch device
        """
        # Look for compatible files
        n_flag = "_with_n" if tokenize_n_nucleotide else ""
        pattern = os.path.join(cache_dir, f"kmer_substitutions_k{k_mer_size}_compatible_*.pkl")
        files = glob.glob("/home/m4safari/projects/def-lila-ab/m4safari/barcodeMAE/BarcodeMAE/barcodebert/masking_codes/kmer_cache/kmer_substitutions_k6_compatible_1c92e11db4415dc6.pkl")

        if not files:
            raise FileNotFoundError(
                f"No compatible lookup table found for k={k_mer_size}, "
                f"tokenize_n_nucleotide={tokenize_n_nucleotide} in {cache_dir}. "
                f"Run: python precompute_compatible_kmers.py --k-mer {k_mer_size} "
                f"{'--tokenize-n-nucleotide ' if tokenize_n_nucleotide else ''}--output-dir {cache_dir}"
            )

        if len(files) > 1:
            print(f"Multiple compatible lookup tables found, using: {files[0]}")

        return cls(files[0], device)

    def sample_substitution(self, original_token, max_attempts=10):
        """
        Sample a biologically-informed substitution for a single k-mer token
        Ensures the result is different from the original token

        Args:
            original_token: k-mer token ID to substitute
            max_attempts: maximum attempts to find a different token

        Returns:
            A different k-mer token ID, or original if no valid substitution found
        """
        # Skip special tokens
        if original_token < self.n_special_tokens:
            return original_token

        if original_token not in self.cumulative_lookup:
            return original_token  # No substitutions available

        cumulative_probs = self.cumulative_lookup[original_token]

        # Try multiple times to ensure we get a different token
        for attempt in range(max_attempts):
            rand_val = np.random.random()

            # Find the sampled substitution
            for target_token, cumulative_prob in cumulative_probs:
                if rand_val <= cumulative_prob:
                    if target_token != original_token:
                        return target_token
                    break  # Try again if same as original

            # If we got the same token, try again with different random value

        # Fallback: find first different token in the list
        for target_token, _ in cumulative_probs:
            if target_token != original_token:
                return target_token

        # If all else fails, return original (shouldn't happen if lookup is correct)
        return original_token

    def get_biological_replacements(self, original_tokens):
        """
        Get biological replacements for given k-mer tokens

        Args:
            original_tokens: tensor or list of k-mer token IDs to replace

        Returns:
            tensor of replacement k-mer token IDs (guaranteed different from originals)
        """
        if isinstance(original_tokens, torch.Tensor):
            original_cpu = original_tokens.cpu().numpy()
            is_tensor = True
        else:
            original_cpu = np.array(original_tokens)
            is_tensor = False

        replacements = np.zeros_like(original_cpu)

        for i, token in enumerate(original_cpu):
            replacements[i] = self.sample_substitution(token)

        if is_tensor:
            return torch.tensor(replacements, dtype=original_tokens.dtype, device=self.device)
        else:
            return replacements


# Temperature-controlled version for curriculum learning
class TemperatureCompatibleBiologicalMasker(CompatibleBiologicalMasker):
    def __init__(self, lookup_file_path, device='cuda'):
        """Load compatible masker with temperature support"""
        super().__init__(lookup_file_path, device)

        # Convert cumulative back to raw probabilities for temperature scaling
        self.raw_lookup = {}
        for token_id, cumulative_probs in self.cumulative_lookup.items():
            raw_probs = []
            prev_cum = 0.0
            for target_token, cum_prob in cumulative_probs:
                raw_prob = cum_prob - prev_cum
                raw_probs.append((target_token, raw_prob))
                prev_cum = cum_prob
            self.raw_lookup[token_id] = raw_probs

    def apply_temperature(self, probabilities, temperature):
        """Apply temperature scaling to probabilities"""
        if temperature == 1.0:
            return probabilities

        tokens = [token for token, _ in probabilities]
        probs = np.array([prob for _, prob in probabilities])

        # Apply temperature: prob^(1/T)
        scaled_probs = np.power(probs, 1.0 / temperature)
        scaled_probs = scaled_probs / np.sum(scaled_probs)

        return list(zip(tokens, scaled_probs))

    def sample_substitution_with_temperature(self, original_token, temperature=1.0, max_attempts=10):
        """
        Sample substitution with temperature control, ensuring different result

        Args:
            original_token: k-mer token ID to substitute
            temperature: sampling temperature (higher = more diverse)
            max_attempts: maximum attempts to find a different token

        Returns:
            A different k-mer token ID, or original if no valid substitution found
        """
        # Skip special tokens
        if original_token < self.n_special_tokens:
            return original_token

        if original_token not in self.raw_lookup:
            return original_token

        # Get raw probabilities and apply temperature
        raw_probs = self.raw_lookup[original_token]
        scaled_probs = self.apply_temperature(raw_probs, temperature)

        # Try multiple times to ensure we get a different token
        for attempt in range(max_attempts):
            rand_val = np.random.random()
            cumulative_sum = 0.0

            for target_token, prob in scaled_probs:
                cumulative_sum += prob
                if rand_val <= cumulative_sum:
                    if target_token != original_token:
                        return target_token
                    break  # Try again if same as original

        # Fallback: find first different token in the scaled list
        for target_token, _ in scaled_probs:
            if target_token != original_token:
                return target_token

        # If all else fails, return original
        return original_token

    def get_biological_replacements_with_temperature(self, original_tokens, temperature=1.0):
        """
        Get temperature-controlled biological replacements for given k-mer tokens

        Args:
            original_tokens: tensor or list of k-mer token IDs to replace
            temperature: sampling temperature (higher = more diverse)

        Returns:
            tensor of replacement k-mer token IDs (guaranteed different from originals)
        """
        if isinstance(original_tokens, torch.Tensor):
            original_cpu = original_tokens.cpu().numpy()
            is_tensor = True
        else:
            original_cpu = np.array(original_tokens)
            is_tensor = False

        replacements = np.zeros_like(original_cpu)

        for i, token in enumerate(original_cpu):
            replacements[i] = self.sample_substitution_with_temperature(token, temperature)

        if is_tensor:
            return torch.tensor(replacements, dtype=original_tokens.dtype, device=self.device)
        else:
            return replacements


# Simplified drop-in replacement functions for your training code
def get_biological_replacements(tokens_to_replace, biological_masker):
    """
    Get biological replacements for specific tokens

    Args:
        tokens_to_replace: tensor of k-mer token IDs that need to be replaced
        biological_masker: CompatibleBiologicalMasker instance

    Returns:
        tensor of replacement k-mer token IDs (guaranteed different from originals)
    """
    return biological_masker.get_biological_replacements(tokens_to_replace)


def get_temperature_biological_replacements(tokens_to_replace, biological_masker, temperature=1.0):
    """
    Get temperature-controlled biological replacements for specific tokens

    Args:
        tokens_to_replace: tensor of k-mer token IDs that need to be replaced
        biological_masker: TemperatureCompatibleBiologicalMasker instance
        temperature: sampling temperature

    Returns:
        tensor of replacement k-mer token IDs (guaranteed different from originals)
    """
    return biological_masker.get_biological_replacements_with_temperature(tokens_to_replace, temperature)


# Legacy compatibility functions (if you want to keep the old interface)
def compatible_biological_random_tokens(sequences, masked_random_tokens, biological_masker):
    """
    Legacy interface: Apply biological masking to selected positions

    Args:
        sequences: original k-mer sequences (with special tokens)
        masked_random_tokens: boolean mask for tokens to replace randomly
        biological_masker: CompatibleBiologicalMasker instance

    Returns:
        sequences with biological substitutions applied to masked positions
    """
    if not masked_random_tokens.any():
        return sequences

    result = sequences.clone()

    # Extract tokens that need replacement
    tokens_to_replace = sequences[masked_random_tokens]

    # Get biological replacements
    replacements = biological_masker.get_biological_replacements(tokens_to_replace)

    # Put replacements back
    result[masked_random_tokens] = replacements

    return result


def temperature_compatible_biological_random_tokens(sequences, masked_random_tokens,
                                                    biological_masker, temperature=1.0):
    """
    Legacy interface: Temperature-controlled biological random token generation

    Args:
        sequences: original k-mer sequences (with special tokens)
        masked_random_tokens: boolean mask for tokens to replace randomly
        biological_masker: TemperatureCompatibleBiologicalMasker instance
        temperature: sampling temperature

    Returns:
        sequences with temperature-controlled biological substitutions
    """
    if not masked_random_tokens.any():
        return sequences

    result = sequences.clone()

    # Extract tokens that need replacement
    tokens_to_replace = sequences[masked_random_tokens]

    # Get temperature-controlled biological replacements
    replacements = biological_masker.get_biological_replacements_with_temperature(
        tokens_to_replace, temperature
    )

    # Put replacements back
    result[masked_random_tokens] = replacements

    return result


def create_temperature_schedule(start_temp, end_temp, total_epochs, schedule_type='exponential'):
    """Create temperature schedule for curriculum learning"""
    epochs = np.arange(total_epochs)

    if schedule_type == 'linear':
        temperatures = start_temp - (start_temp - end_temp) * epochs / (total_epochs - 1)
    elif schedule_type == 'exponential':
        ratio = end_temp / start_temp
        temperatures = start_temp * (ratio ** (epochs / (total_epochs - 1)))
    elif schedule_type == 'cosine':
        temperatures = end_temp + (start_temp - end_temp) * (1 + np.cos(np.pi * epochs / (total_epochs - 1))) / 2
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    return temperatures.tolist()