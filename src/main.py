from text_iterator import TextIterator
from vocab_builder import VocabBuilder
from streaming_skip_gram_dataset import StreamingSkipGramDataset
from torch.utils.data import DataLoader
from subsampler import Subsampler
from negative_sampler import NegativeSampler
from word2vec_model import Word2VecModel

import click
import torch
import torch.nn.functional as F
from torch import optim

CORPUS = '../data/text8'
MIN_WORD_FREQ = 5
MAX_VOCAB_SIZE = 50000
BATCH_SIZE = 512
WINDOW_SIZE = 5
EMBEDDING_DIM = 100
LEARNING_RATE = 0.001
EPOCHS = 15


@click.group()
def cli():
    """Word2Vec training and inference CLI."""
    pass


@cli.command()
@click.option('--checkpoint-path', default='checkpoint.pt', help='Path to save model checkpoint')
def train(checkpoint_path):
    """Train the Word2Vec model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    click.echo(f"Using device: {device}")

    click.echo("Building vocabulary...")
    text_iterator = TextIterator(CORPUS)
    vocab_builder = VocabBuilder(text_iterator, min_freq=MIN_WORD_FREQ, max_vocab_size=MAX_VOCAB_SIZE)
    vocab, idx_to_word, word_counts = vocab_builder.build_vocab()

    click.echo("Setting up dataset and dataloader...")
    subsampler = Subsampler(word_counts)
    negative_sampler = NegativeSampler(word_counts, vocab)
    dataset = StreamingSkipGramDataset(text_iterator, subsampler, negative_sampler, vocab, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    click.echo("Initializing model and optimizer...")
    model = Word2VecModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    click.echo("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0

        for center, context, negatives in dataloader:
            center = center.to(device)
            context = context.to(device)
            negatives = negatives.to(device)

            optimizer.zero_grad()
            loss = model(center, context, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            batch_count += 1
            if batch_count % 1000 == 0:
                click.echo(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {loss.item():.4f}")

        if (epoch + 1) % 20 == 0:
            click.echo(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/batch_count:.4f}")

    click.echo(f"Training complete. Saving model checkpoint to {checkpoint_path}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        'vocab': vocab,
        'idx_to_word': idx_to_word,
        'embedding_dim': EMBEDDING_DIM,
    }, checkpoint_path)
    click.echo("Done!")


def find_similar(word, vocab, idx_to_word, embeddings, top_n):
    """Find the most similar words using cosine similarity."""
    word_idx = vocab[word]
    word_vec = embeddings[word_idx].unsqueeze(0)  # (1, embed_dim)

    # Cosine similarity: normalize then dot product
    word_vec_norm = F.normalize(word_vec, dim=1)
    embeddings_norm = F.normalize(embeddings, dim=1)
    similarities = torch.mm(word_vec_norm, embeddings_norm.t()).squeeze(0)

    # Exclude the word itself
    similarities[word_idx] = -float('inf')
    top_indices = similarities.argsort(descending=True)[:top_n]

    return [(idx_to_word[idx.item()], similarities[idx].item()) for idx in top_indices]


@cli.command()
@click.option('--checkpoint-path', required=True, help='Path to model checkpoint')
@click.option('--top-n', default=10, help='Number of similar words to return')
def similar(checkpoint_path, top_n):
    """Interactive similar word lookup."""
    click.echo(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    vocab = checkpoint['vocab']
    idx_to_word = checkpoint['idx_to_word']
    embedding_dim = checkpoint['embedding_dim']

    # Rebuild model and load weights
    model = Word2VecModel(vocab_size=len(vocab), embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    embeddings = model.center_embeddings.weight.detach()

    click.echo(f"Loaded model with {len(vocab)} words, {embedding_dim}-dim embeddings")
    click.echo('Enter a word to find similar words (or "quit" to exit)\n')

    while True:
        word = click.prompt('Word')
        if word.lower() == 'quit':
            click.echo("Goodbye!")
            break
        if word not in vocab:
            click.echo(f'Word "{word}" not in vocabulary\n')
            continue

        results = find_similar(word, vocab, idx_to_word, embeddings, top_n)
        click.echo(f"\nTop {top_n} similar words to '{word}':")
        for i, (similar_word, score) in enumerate(results, 1):
            click.echo(f"  {i:2}. {similar_word:<20} {score:.4f}")
        click.echo()


if __name__ == "__main__":
    cli()
