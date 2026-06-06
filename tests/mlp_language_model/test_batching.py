"""Fixed-context MLP batching tests will live here."""

from mlp_language_model.mlp import get_batch, get_data, get_splits


def test_get_batch():
    data = get_data()
    train_data = get_splits(data)["train"]
    x, y = get_batch(split=train_data, batch_size=10, context_size=10)
    assert x.size() == (10, 10)
    assert y.size() == (10,)
