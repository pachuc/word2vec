from torch.utils.data import IterableDataset
from text_iterator import TextIterator
from subsampler import Subsampler
from negative_sampler import NegativeSampler
import random
from torch import tensor

class StreamingSkipGramDataset(IterableDataset):
    def __init__(self, text_iterator: TextIterator, subsampler: Subsampler, negative_sampler: NegativeSampler, vocab, window_size=5, num_negatives=5):
        self.text_iterator = text_iterator
        self.subsampler = subsampler
        self.negative_sampler = negative_sampler
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.word_list = list(vocab.keys())
        
    def __iter__(self):
        for tokens in self.text_iterator:
            subsampled_tokens = self.subsampler.subsample_tokens(tokens)
            token_indices = [
                self.vocab[t] for t in subsampled_tokens if t in self.vocab
            ]
            
            for i, token_idx in enumerate(token_indices):
                # We use a random window size here, bounded by the passed in window_size
                # the reason for this random window size is that words closer to the center word 
                # are usually more related than words further away. 
                # By randomly picking a smaller window, closer words get sampled more often.
                # It also improves training efficiency by creating fewer pairs per sentence on average,
                # and adding noise/variance as a method of regualarization.
                actual_window = random.randint(1, self.window_size)
                    
                start = max(0, i - actual_window)
                end = min(len(token_indices), i + actual_window + 1)
                    
                for j in range(start, end):
                    if i != j:
                        context_idx = token_indices[j]
                        
                        negatives = self.negative_sampler.sample(
                            self.num_negatives, 
                            exclude_idx=context_idx
                        )
                        
                        yield (
                            tensor(token_idx),
                            tensor(context_idx),
                            tensor(negatives)
                        )
