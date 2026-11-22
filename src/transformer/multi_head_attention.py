import torch
from torch import nn

from language_modeling.tokenizer import CharTokenizer
from transformer.single_head_attention import SingleHeadCausalSelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        embed_dim: int,
    ):
        super().__init__()
        if embed_dim % num_heads > 0:
            raise ValueError()
        self.heads = [
            SingleHeadCausalSelfAttention(
                embed_dim=embed_dim,
                head_dim=head_dim,
            )
            for _ in range(num_heads)
        ]
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim)

    def forward(self, idx: torch.Tensor):
        outputs = [head(idx) for head in self.heads]
        concatenated = torch.cat(outputs, dim=-1)
        o = self.out_proj(concatenated)
        return o
