from bigram.bigram import (
    batch_size,
    block_size,
    data,
    get_batch,
    test_data,
    validation_data,
)
import torch


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
    # Check that y is the next-token target for x.
    for i in range(batch_size):
        for j in range(block_size - 1):
            assert y[i, j] == x[i, j + 1]
