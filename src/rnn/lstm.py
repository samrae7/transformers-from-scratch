from torch import nn
import torch


class LSTMCell(nn.Module):
    def __init__(self, embed_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.w_fx = nn.Linear(embed_dim, hidden_dim)
        self.w_fh = nn.Linear(hidden_dim, hidden_dim)

    def _compute_gates(self, x_t, h_prev):
        f_t = torch.sigmoid(self.w_fx(x_t) + self.w_fh(h_prev))
        return f_t

    def forward(self, x_t, h_prev, c_prev):
        return x_t
