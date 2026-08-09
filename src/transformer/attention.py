import math

import torch


class TokenAndPositionEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, max_sequence_length: int, embed_dim: int):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.token_embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = torch.nn.Embedding(max_sequence_length, embed_dim)

    def forward(self, idx: torch.Tensor):
        sequence_length = idx.shape[1]
        positions = torch.arange(sequence_length, device=idx.device)
        return self.token_embeddings(idx) + self.position_embeddings(positions)


class SingleHeadCausalSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.w_q = torch.nn.Linear(embed_dim, head_dim)
        self.w_k = torch.nn.Linear(embed_dim, head_dim)
        self.w_v = torch.nn.Linear(embed_dim, head_dim)
        self.out_proj = torch.nn.Linear(head_dim, embed_dim)

    def scaled_dot_product(self, q: torch.Tensor, k: torch.Tensor):
        k_transposed = k.transpose(-2, -1)
        a = q @ k_transposed
        scaled_scores = a / math.sqrt(self.head_dim)
        return scaled_scores

    def apply_causal_mask(self, scores: torch.Tensor):
        sequence_length = scores.shape[-1]
        indices = torch.arange(sequence_length, device=scores.device)
        column_positions = indices[:, None]
        row_positions = indices[None, :]
        mask = column_positions >= row_positions
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        return masked_scores

    def attention_weights(self, q: torch.Tensor, k: torch.Tensor):
        scaled_scores = self.scaled_dot_product(q, k)
        masked_scores = self.apply_causal_mask(scaled_scores)
        attention_weights = torch.softmax(masked_scores, dim=-1)
        return attention_weights

    def forward(self, x: torch.Tensor):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        attention_weights = self.attention_weights(q, k)
        o = attention_weights @ v
        return self.out_proj(o)
