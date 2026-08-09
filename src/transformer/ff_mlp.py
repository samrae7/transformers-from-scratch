import torch
from torch import nn


class FFMLP(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc_in = nn.Linear(embed_dim, embed_dim * 4)
        self.activation = nn.ReLU()
        self.fc_out = nn.Linear(embed_dim * 4, embed_dim)

    def forward(self, input: torch.Tensor):
        out = self.fc_in(input)
        out = self.activation(out)
        return self.fc_out(out)
