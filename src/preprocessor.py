import numpy as np
import random
from text_iterator import TextIterator
from vocab_builder import VocabBuilder
from subsampler import Subsampler


class SkipGramPreprocessor:
    """Generates and saves skip-gram pairs to disk for fast training."""

    def __init__(self, text_iterator: TextIterator, vocab: dict, subsampler: Subsampler,
                 window_size: int = 5, use_random_window: bool = True):
        self.text_iterator = text_iterator
        self.vocab = vocab
        self.subsampler = subsampler
        self.window_size = window_size
        self.use_random_window = use_random_window

    def generate_pairs(self, progress_callback=None):
        """
        Generate all (center, context) skip-gram pairs.

        Returns numpy array of shape (num_pairs, 2) with dtype int32.
        """
        pairs = []
        chunk_count = 0

        for tokens in self.text_iterator:
            # Subsample frequent words
            subsampled_tokens = self.subsampler.subsample_tokens(tokens)

            # Convert to indices, skip OOV words
            token_indices = [
                self.vocab[t] for t in subsampled_tokens if t in self.vocab
            ]

            # Generate skip-gram pairs
            for i, center_idx in enumerate(token_indices):
                if self.use_random_window:
                    actual_window = random.randint(1, self.window_size)
                else:
                    actual_window = self.window_size

                start = max(0, i - actual_window)
                end = min(len(token_indices), i + actual_window + 1)

                for j in range(start, end):
                    if i != j:
                        context_idx = token_indices[j]
                        pairs.append((center_idx, context_idx))

            chunk_count += 1
            if progress_callback and chunk_count % 100 == 0:
                progress_callback(chunk_count, len(pairs))

        return np.array(pairs, dtype=np.int32)

    @staticmethod
    def save_pairs(pairs: np.ndarray, filepath: str):
        """Save pairs to disk as memory-mappable numpy file."""
        np.save(filepath, pairs)

    @staticmethod
    def load_pairs(filepath: str, mmap_mode: str = 'r'):
        """Load pairs from disk with optional memory mapping."""
        return np.load(filepath, mmap_mode=mmap_mode)
