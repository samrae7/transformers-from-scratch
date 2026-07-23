from torch import nn
import torch

from language_modeling.tokenizer import CharTokenizer


class LSTMCell(nn.Module):
    def __init__(self, embed_dim: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.w_fx = nn.Linear(embed_dim, hidden_dim)
        self.w_fh = nn.Linear(hidden_dim, hidden_dim)
        self.w_ix = nn.Linear(embed_dim, hidden_dim)
        self.w_ih = nn.Linear(hidden_dim, hidden_dim)
        self.w_gx = nn.Linear(embed_dim, hidden_dim)
        self.w_gh = nn.Linear(hidden_dim, hidden_dim)
        self.w_ox = nn.Linear(embed_dim, hidden_dim)
        self.w_oh = nn.Linear(hidden_dim, hidden_dim)

    def _compute_gates(self, x_t, h_prev):
        f_t = torch.sigmoid(self.w_fx(x_t) + self.w_fh(h_prev))
        i_t = torch.sigmoid(self.w_ix(x_t) + self.w_ih(h_prev))
        g_t = torch.tanh(self.w_gx(x_t) + self.w_gh(h_prev))
        o_t = torch.sigmoid(self.w_ox(x_t) + self.w_oh(h_prev))
        return f_t, i_t, g_t, o_t

    def forward(self, x_t, h_prev, c_prev):
        f_t, i_t, g_t, o_t = self._compute_gates(x_t, h_prev)
        c_t = f_t * c_prev + i_t * g_t
        h_t = torch.tanh(c_t) * o_t
        return h_t, c_t


class LSTM(nn.Module):
    def __init__(
        self, cell: LSTMCell, tokenizer: CharTokenizer, embed_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.cell = cell
        self.hidden_dim = hidden_dim
        self.tokenizer = tokenizer
        self.embeddings = nn.Embedding(
            num_embeddings=tokenizer.vocab_size, embedding_dim=embed_dim
        )
        self.hidden_to_logits = nn.Linear(hidden_dim, tokenizer.vocab_size)

    def forward(self, inputs: torch.Tensor):
        #  inputs shape is B, T
        # cell takes B, E
        batch_size, sequence_length = inputs.shape
        embs = self.embeddings(inputs)
        # embs.new_zeros -> h_t,c_t have same device and d_type as embs
        h_t = embs.new_zeros([batch_size, self.hidden_dim])
        c_t = embs.new_zeros([batch_size, self.hidden_dim])
        hidden_states = []
        for t in range(sequence_length):
            emb = embs[:, t, :]
            h_t, c_t = self.cell(emb, h_t, c_t)
            hidden_states.append(h_t)
        #  hidden_to_logits takes T hidden states, which individually
        #  are B,H and turns them into logits which are B,T,V
        #  so we want a stack of hidden_states that are B,T,H
        hidden_states_stack = torch.stack(hidden_states, dim=1)
        logits = self.hidden_to_logits(hidden_states_stack)
        return logits, (h_t, c_t)
