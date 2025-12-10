import math
import random

class Subsampler:

    def __init__(self, word_counts, subsample_threshold=1e-3):
        """
        subsample_threshold: words with frequency > this get subsampled
        Original paper used 1e-3 to 1e-5
        """
        self.total_words = sum(word_counts.values())
        self.subsample_threshold = subsample_threshold
        
        self.discard_prob = {}
        for word, count in word_counts.items():
            freq = count / self.total_words
            # this formula is taken from the original word2vec paper
            if freq > subsample_threshold:
                self.discard_prob[word] = 1 - math.sqrt(subsample_threshold / freq)
            else:
                self.discard_prob[word] = 0
    
    def should_keep(self, word):
        """Returns True if we should keep this word occurrence"""
        if word not in self.discard_prob:
            return True
        return random.random() > self.discard_prob[word]
    
    def subsample_tokens(self, tokens):
        """Subsamples a list of tokens"""
        return [t for t in tokens if self.should_keep(t)]
