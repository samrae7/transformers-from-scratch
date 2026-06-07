"""Fixed-context MLP shape tests will live here."""

from mlp_language_model.mlp import MLP, get_batch, get_data, get_splits


def test_embeddings_have_expected_shape():
    data = get_data()
    train_data = get_splits(data)["train"]
    x, _ = get_batch(split=train_data, batch_size=5, context_size=10)
    assert x.size() == (5, 10)
    model = MLP(embed_dim=8)
    emb = model.forward(x)
    assert emb.size() == (5, 80)
