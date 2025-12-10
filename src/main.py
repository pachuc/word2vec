from text_iterator import TextIterator
from vocab_builder import VocabBuilder
from streaming_skip_gram_dataset import StreamingSkipGramDataset
from torch.utils.data import DataLoader
from subsampler import Subsampler
from negative_sampler import NegativeSampler
from word2vec_model import Word2VecModel

from torch import optim

CORPUS = '../data/text8'
MIN_WORD_FREQ = 5
MAX_VOCAB_SIZE = 50000
BATCH_SIZE = 512
WINDOW_SIZE = 5
EMBEDDING_DIM = 100
LEARNING_RATE = 0.01
EPOCHS = 100

def main():
    print("Building vocabulary...")
    text_iterator = TextIterator(CORPUS)
    vocab_builder = VocabBuilder(text_iterator, min_freq=MIN_WORD_FREQ, max_vocab_size=MAX_VOCAB_SIZE)
    vocab, idx_to_word, word_counts = vocab_builder.build_vocab()

    print("Setting up dataset and dataloader...")
    subsampler = Subsampler(word_counts)
    negative_sampler = NegativeSampler(word_counts, vocab)
    dataset = StreamingSkipGramDataset(text_iterator, subsampler, negative_sampler, vocab, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    print("Initializing model and optimizer...")
    model = Word2VecModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    # Training
    for epoch in range(EPOCHS):
        total_loss = 0
        for center, context, negatives in dataloader:
            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    main()

