import torch
import torch.nn.functional as F

from language_modeling.data import encode_data, get_splits
from language_modeling.tokenizer import CharTokenizer
from rnn.data import get_batch
from rnn.lstm import LSTM, LSTMCell


# x_t, h_prev, c_prev  ->  h_t, c_t
def test_gates():
    batch_size = 5
    hidden_dim = 32
    embed_dim = 64

    cell = LSTMCell(embed_dim=embed_dim, hidden_dim=hidden_dim)

    h_prev = torch.zeros(batch_size, hidden_dim)
    # c_prev = torch.zeros(batch_size, hidden_dim)
    x_t = torch.randn(batch_size, embed_dim)

    f_t, i_t, g_t, o_t = cell._compute_gates(x_t, h_prev)
    assert f_t.shape == (batch_size, hidden_dim)
    assert i_t.shape == (batch_size, hidden_dim)
    assert g_t.shape == (batch_size, hidden_dim)
    assert o_t.shape == (batch_size, hidden_dim)

    assert torch.all((0 <= f_t) & (f_t <= 1))
    assert torch.all((0 <= i_t) & (i_t <= 1))
    assert torch.all((-1 <= g_t) & (g_t <= 1))
    assert torch.all((0 <= o_t) & (o_t <= 1))


def test_lstm():
    batch_size = 5
    hidden_dim = 32
    embed_dim = 64

    cell = LSTMCell(embed_dim=embed_dim, hidden_dim=hidden_dim)
    c_prev = torch.zeros(batch_size, hidden_dim)
    h_prev = torch.zeros(batch_size, hidden_dim)
    x_t = torch.randn(batch_size, embed_dim)
    h_t, c_t = cell(x_t, h_prev, c_prev)
    assert c_t.shape == (batch_size, hidden_dim)
    assert h_t.shape == (batch_size, hidden_dim)


def test_gradients_reach_cell_params():
    batch_size = 5
    hidden_dim = 32
    embed_dim = 64

    cell = LSTMCell(embed_dim=embed_dim, hidden_dim=hidden_dim)
    c_prev = torch.randn(batch_size, hidden_dim)
    h_prev = torch.randn(batch_size, hidden_dim)
    x_t = torch.randn(batch_size, embed_dim)
    h_t, c_t = cell(x_t, h_prev, c_prev)

    loss = h_t.square().mean() + c_t.square().mean()
    loss.backward()

    for name, parameter in cell.named_parameters():
        assert parameter.grad is not None, name
        assert torch.any(parameter.grad != 0), name


def test_lstm_forward():
    text = "abcdefghijklmnopqrstuvwxyzfsdfsdfsdfvsfsfsfvsvfsgsdvsvss"
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer)

    train_data = get_splits(data)["train"]
    batch_size = 5
    sequence_length = 10
    x, _ = get_batch(
        split=train_data, batch_size=batch_size, sequence_length=sequence_length
    )
    embed_dim = 64
    hidden_dim = 32
    cell = LSTMCell(embed_dim=embed_dim, hidden_dim=hidden_dim)

    model = LSTM(
        tokenizer=tokenizer, embed_dim=embed_dim, cell=cell, hidden_dim=hidden_dim
    )

    logits, (h_t, c_t) = model(x)
    assert logits.shape == (batch_size, sequence_length, tokenizer.vocab_size)
    assert h_t.shape == (batch_size, hidden_dim)
    assert c_t.shape == (batch_size, hidden_dim)


def test_single_train_batch():
    text = "abcdefghijklmnopqrstuvwxyzfsdfsdfsdfvsfsfsfvsvfsgsdvsvss"
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer)

    train_data = get_splits(data)["train"]
    batch_size = 5
    sequence_length = 10
    embed_dim = 64
    hidden_dim = 32
    cell = LSTMCell(embed_dim=embed_dim, hidden_dim=hidden_dim)

    model = LSTM(
        tokenizer=tokenizer, embed_dim=embed_dim, cell=cell, hidden_dim=hidden_dim
    )

    # get batch
    x, y = get_batch(
        split=train_data, batch_size=batch_size, sequence_length=sequence_length
    )
    #  run model
    logits, _ = model(x)
    #  calc cross entropy
    #  logits have shape B,T,V
    #  cross_entropy take N,C, which means we need to reshape to B*T, V
    #  shape of targets is B,T, C.E take shape N so we reshape to B * T
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(0, 1))
    #  call backward
    loss.backward()
    #  assert loss i scalar
    assert loss.shape == ()
    #  assert at least one model param has a gradient
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.any(parameter.grad != 0), name


def test_generate():
    text = "abcdefghijklmnopqrstuvwxyzfsdfsdfsdfvsfsfsfvs vfsgsdvsvsstobeornottobeT"
    tokenizer = CharTokenizer(text)
    embed_dim = 64
    hidden_dim = 32
    cell = LSTMCell(embed_dim=embed_dim, hidden_dim=hidden_dim)
    model = LSTM(
        tokenizer=tokenizer, embed_dim=embed_dim, cell=cell, hidden_dim=hidden_dim
    )
    result = model.generate("To be or not to be", 100)
    assert len(result) == 100
    assert isinstance(result, str)
