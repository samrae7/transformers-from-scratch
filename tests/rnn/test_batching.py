import torch

from language_modeling.tokenizer import CharTokenizer
from language_modeling.data import encode_data, get_splits
from rnn.rnn import get_batch


def test_get_batch():
    text = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer)

    train_data = get_splits(data)["train"]
    batch_size = 5
    sequence_length = 10
    x, y = get_batch(
        split=train_data, batch_size=batch_size, sequence_length=sequence_length
    )
    assert x.size() == (batch_size, sequence_length)
    assert y.size() == (batch_size, sequence_length)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    assert torch.equal(y[:, :-1], x[:, 1:])
