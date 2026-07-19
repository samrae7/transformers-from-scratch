import torch

from language_modeling.data import encode_data, get_splits
from language_modeling.tokenizer import CharTokenizer
from rnn.data import get_batch
from rnn.lstm import LSTMCell


# x_t, h_prev, c_prev  ->  h_t, c_t
def test_lstm():
    batch_size = 5
    hidden_dim = 32
    embed_dim = 64

    cell = LSTMCell(embed_dim=embed_dim, hidden_dim=hidden_dim)

    h_prev = torch.zeros(batch_size, hidden_dim)
    # c_prev = torch.zeros(batch_size, hidden_dim)
    x_t = torch.randn(batch_size, embed_dim)

    # h_t, c_t = cell(x_t, h_prev, c_prev)
    f_t, i_t, g_t, o_t = cell._compute_gates(x_t, h_prev)
    assert f_t.shape == (batch_size, hidden_dim)
    assert i_t.shape == (batch_size, hidden_dim)
    assert g_t.shape == (batch_size, hidden_dim)
    assert o_t.shape == (batch_size, hidden_dim)

    assert torch.all((0 <= f_t) & (f_t <= 1))
    assert torch.all((0 <= i_t) & (i_t <= 1))
    assert torch.all((-1 <= g_t) & (g_t <= 1))
    assert torch.all((0 <= o_t) & (o_t <= 1))


# Inputs:

# ```text
# x_t:     [batch, n_embd]
# h_(t-1): [batch, hidden_dim]
# c_(t-1): [batch, hidden_dim]
# ```

# Outputs:

# ```text
# h_t: [batch, hidden_dim]
# c_t: [batch, hidden_dim]
# ```

# Test:

# - Every gate shape.
# - New hidden-state shape.
# - New cell-state shape.
# - Values from sigmoid gates lie between 0 and 1.
# - Gradients reach the cell parameters.
