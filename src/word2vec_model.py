from torch import nn
import torch

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
        
    def forward(self, center_words, context_words, negative_words):
        """
        center_words: (batch_size,)
        context_words: (batch_size,)
        negative_words: (batch_size, num_negatives)
        """
        batch_size = center_words.size(0)
        
        # Get embeddings
        center_embeds = self.center_embeddings(center_words)      # (batch, embed_dim)
        context_embeds = self.context_embeddings(context_words)   # (batch, embed_dim)
        negative_embeds = self.context_embeddings(negative_words) # (batch, num_neg, embed_dim)
        
        # Positive score: dot product of center and context
        positive_score = torch.sum(center_embeds * context_embeds, dim=1)  # (batch,)
        positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-10)
        
        # Negative score: dot product of center and negatives
        # center_embeds: (batch, embed_dim) -> (batch, embed_dim, 1)
        center_embeds = center_embeds.unsqueeze(2)
        # negative_embeds: (batch, num_neg, embed_dim)
        negative_score = torch.bmm(negative_embeds, center_embeds).squeeze(2)  # (batch, num_neg)
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score) + 1e-10), dim=1)
        
        # Total loss
        loss = torch.mean(positive_loss + negative_loss)
        return loss
    
    def get_embedding(self, word_idx):
        return self.center_embeddings.weight[word_idx].detach()