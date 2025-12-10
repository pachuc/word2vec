from collections import Counter
from text_iterator import TextIterator

class VocabBuilder:
    def __init__(self, text_iterator: TextIterator, min_freq=5, max_vocab_size=None):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word_counts = Counter()
        self.text_iterator = text_iterator
        
    def _update_from_iterator(self):
        """
        Itterate through our corpus and count word frequencies
        """
        for token_set in self.text_iterator:
            self.word_counts.update(token_set)


    def build_vocab(self):
        """
        Build the vocabulary dictionary mapping words to indices
        """
        self._update_from_iterator()
        vocab = {}
        idx_to_word = {}
        idx = 0
        for word, count in self.word_counts.most_common():
            if count >= self.min_freq:
                vocab[word] = idx
                idx_to_word[idx] = word
                idx += 1
            
            if self.max_vocab_size and len(vocab) >= self.max_vocab_size:
                break

        return vocab, idx_to_word, self.word_counts
