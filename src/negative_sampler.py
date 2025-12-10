import numpy as np

class NegativeSampler:
    def __init__(self, word_counts, vocab, power=0.75):
        """
        Negative samples creates a list of bad pairs that do not match our training objective.
        We use these as a secondary training objective, punishing the model for predicting these pairs.

        Sample negatives proportional to freq^power
        power=0.75 is from the original paper
        """
        self.vocab_size = len(vocab)
        
        # Build sampling distribution
        words = list(vocab.keys())
        freqs = np.array([word_counts.get(w, 1) for w in words], dtype=np.float64)
        freqs = freqs ** power
        self.sampling_probs = freqs / freqs.sum()
        
        # Pre-generate a large table for fast sampling (like original word2vec)
        self.table_size = 10_000_000
        self.table = np.random.choice(
            self.vocab_size, 
            size=self.table_size, 
            p=self.sampling_probs
        )
        self.table_idx = 0
    
    def sample(self, num_samples, exclude_idx=None):
        """Fast negative sampling from pre-built table"""
        samples = []
        while len(samples) < num_samples:
            neg = self.table[self.table_idx]
            self.table_idx = (self.table_idx + 1) % self.table_size
            if neg != exclude_idx:
                samples.append(neg)
        return samples
