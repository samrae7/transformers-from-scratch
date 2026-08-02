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
        self.w_q = torch.nn.Linear(embed_dim, head_dim)
        self.w_k = torch.nn.Linear(embed_dim, head_dim)
        self.w_v = torch.nn.Linear(embed_dim, head_dim)

    def forward(self, x: torch.Tensor):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        k_transposed = k.transpose(-2, -1)
        a = q @ k_transposed
        o = a @ v
        return o
