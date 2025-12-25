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
import yaml
import os
from dataclasses import dataclass

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')


@dataclass
class Config:
    corpus: str
    min_word_freq: int
    max_vocab_size: int
    batch_size: int
    window_size: int
    embedding_dim: int
    learning_rate: float
    epochs: int


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)


@click.group()
def cli():
    """Word2Vec training and inference CLI."""
    pass


@cli.command()
@click.option('--config', default=DEFAULT_CONFIG_PATH, help='Path to config file')
@click.option('--checkpoint-path', default='checkpoint.pt', help='Path to save model checkpoint')
@click.option('--resume-from', default=None, help='Path to checkpoint file to resume training from')
def train(config, checkpoint_path, resume_from):
    """Train the Word2Vec model."""
    cfg = load_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    click.echo(f"Using device: {device}")

    start_epoch = 0

    if resume_from:
        click.echo(f"Loading checkpoint from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        vocab = checkpoint['vocab']
        idx_to_word = checkpoint['idx_to_word']
        word_counts = checkpoint.get('word_counts')
        start_epoch = checkpoint['epoch'] + 1
        click.echo(f"Resuming from epoch {start_epoch}")

        # Rebuild text iterator for dataset
        text_iterator = TextIterator(cfg.corpus)

        if word_counts is None:
            click.echo("Rebuilding word counts from corpus...")
            vocab_builder = VocabBuilder(text_iterator, min_freq=cfg.min_word_freq, max_vocab_size=cfg.max_vocab_size)
            _, _, word_counts = vocab_builder.build_vocab()
    else:
        click.echo("Building vocabulary...")
        text_iterator = TextIterator(cfg.corpus)
        vocab_builder = VocabBuilder(text_iterator, min_freq=cfg.min_word_freq, max_vocab_size=cfg.max_vocab_size)
        vocab, idx_to_word, word_counts = vocab_builder.build_vocab()

    click.echo("Setting up dataset and dataloader...")
    subsampler = Subsampler(word_counts)
    negative_sampler = NegativeSampler(word_counts, vocab)
    dataset = StreamingSkipGramDataset(text_iterator, subsampler, negative_sampler, vocab, window_size=cfg.window_size)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size)

    click.echo("Initializing model and optimizer...")
    model = Word2VecModel(vocab_size=len(vocab), embedding_dim=cfg.embedding_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if resume_from:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        click.echo("Loaded model and optimizer state from checkpoint")

    total_epochs = start_epoch + cfg.epochs
    click.echo(f"Starting training from epoch {start_epoch} to {total_epochs}...")
    for epoch in range(start_epoch, total_epochs):
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
            click.echo(f"Epoch {epoch+1}/{total_epochs}, Loss: {total_loss/batch_count:.4f}")

    click.echo(f"Training complete. Saving model checkpoint to {checkpoint_path}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        'vocab': vocab,
        'idx_to_word': idx_to_word,
        'word_counts': word_counts,
        'embedding_dim': cfg.embedding_dim,
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
