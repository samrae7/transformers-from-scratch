"""Fixed-context MLP batching tests will live here."""

from language_modeling.tokenizer import CharTokenizer
from mlp_language_model.mlp import encode_data, get_batch, get_splits


def test_get_batch():
    text = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer)

    train_data = get_splits(data)["train"]
    x, y = get_batch(split=train_data, batch_size=10, context_size=10)
    assert x.size() == (10, 10)
    assert y.size() == (10,)
