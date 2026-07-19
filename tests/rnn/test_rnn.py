import torch

from language_modeling.data import encode_data, get_splits
from language_modeling.tokenizer import CharTokenizer
from rnn.data import get_batch
from rnn.rnn import RNN
import torch.nn.functional as F


def test_rnn_logits():
    text = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer)

    train_data = get_splits(data)["train"]
    batch_size = 5
    sequence_length = 10
    x, _ = get_batch(
        split=train_data, batch_size=batch_size, sequence_length=sequence_length
    )
    emb_dim = 64
    hidden_dim = 32

    model = RNN(tokenizer=tokenizer, embed_dim=emb_dim, hidden_dim=hidden_dim)

    logits, hidden = model(x)
    assert logits.shape == (batch_size, sequence_length, tokenizer.vocab_size)
    assert hidden.shape == (batch_size, hidden_dim)


def test_gradient_norm_is_positive_after_backward():
    text = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer)

    train_data = get_splits(data)["train"]
    batch_size = 5
    sequence_length = 10
    emb_dim = 64
    hidden_dim = 32
    lr = 0.001

    model = RNN(tokenizer=tokenizer, embed_dim=emb_dim, hidden_dim=hidden_dim)

    idx, targets = get_batch(
        split=train_data, batch_size=batch_size, sequence_length=sequence_length
    )
    logits, _ = model(idx)
    logits = logits.flatten(0, 1)
    targets = targets.flatten(0, 1)
    loss = F.cross_entropy(logits, targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    optimizer.zero_grad()
    assert model.get_total_gradient_norm() == 0.0
    loss.backward()

    assert model.get_total_gradient_norm() > 0.0
