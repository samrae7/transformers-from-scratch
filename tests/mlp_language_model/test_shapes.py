"""Fixed-context MLP shape tests will live here."""

from language_modeling.tokenizer import CharTokenizer

from mlp_language_model.mlp import MLP, encode_data, get_batch, get_splits


def test_embeddings_have_expected_shape():
    text = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer)

    train_data = get_splits(data)["train"]
    x, _ = get_batch(split=train_data, batch_size=5, context_size=10)
    assert x.size() == (5, 10)

    vocab_size = 65
    model = MLP(vocab_size=65, context_size=10, embed_dim=8)
    emb = model.forward(x)
    assert emb.size() == (5, vocab_size)
