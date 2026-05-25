import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bigram import (
    BigramModel,
    stoi,
    itos,
    encode,
    decode,
    vocab_size,
    data,
    test_data,
    validation_data,
    get_batch,
    batch_size,
    block_size,
)
import torch


def test_stoi_and_itos_are_inverses():
    for char, idx in stoi.items():
        assert itos[idx] == char


def test_encode_decode_round_trip():
    sample = "First Citizen:\n"

    encoded = encode(sample)
    decoded = decode(encoded)

    assert decoded == sample


def test_encode_returns_integer_token_ids():
    sample = "hello"

    encoded = encode(sample)

    assert all(isinstance(idx, int) for idx in encoded)
    assert all(0 <= idx < vocab_size for idx in encoded)


def test_decode_returns_string():
    ids = [
        stoi["h"],
        stoi["e"],
        stoi["l"],
        stoi["l"],
        stoi["o"],
    ]

    decoded = decode(ids)

    assert decoded == "hello"


def test_encoded_data_is_one_dimensional_tensor():
    assert isinstance(data, torch.Tensor)
    assert data.ndim == 1


def test_train_and_validation_dont_overlap():
    length_data = data.size()[0]
    length_test_data = test_data.size()[0]
    length_validation_data = validation_data.size()[0]
    assert length_data == length_test_data + length_validation_data


def test_split_returns_original_token_order():
    split_idx = len(test_data)

    assert torch.equal(test_data, data[:split_idx])
    assert torch.equal(validation_data, data[split_idx:])


def test_get_batch():
    x, y = get_batch(test_data)
    assert x.size() == (batch_size, block_size)
    assert y.size() == (batch_size, block_size)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    # Check that y is the next-token target for x
    for i in range(batch_size):
        for j in range(block_size - 1):
            assert y[i, j] == x[i, j + 1]


def test_bigram_model():
    model = BigramModel()
    idx, targets = get_batch(test_data)
    loss, logits = model.forward(idx, targets)
    assert loss.ndim == 0
    assert logits.size() == (batch_size, block_size, vocab_size)


def test_forward_returns_loss_as_none_whe_targets_notprovided():
    model = BigramModel()
    idx, _targets = get_batch(test_data)
    loss, logits = model.forward(idx)
    assert loss is None
    assert logits.size() == (batch_size, block_size, vocab_size)


def test_get_next_token():
    model = BigramModel()
    start = stoi["\n"]
    next_token = model._get_next_token(start)

    assert isinstance(next_token, int)


def test_generate():
    model = BigramModel()
    start_char = "\n"
    generated = model.generate(start_char, 10)
    assert len(generated) == 10
