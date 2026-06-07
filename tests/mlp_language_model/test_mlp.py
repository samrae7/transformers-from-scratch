import torch
import torch.nn.functional as F

from language_modeling.tokenizer import CharTokenizer

from mlp_language_model.mlp import MLP, encode_data, get_batch, get_splits

text = "abcdefghijklmnopqrstuvwxyz"
tokenizer = CharTokenizer(text)
data = encode_data(text, tokenizer)


def test_logits_have_expected_shape():
    text = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer)

    train_data = get_splits(data)["train"]
    x, _ = get_batch(split=train_data, batch_size=5, context_size=10)
    assert x.size() == (5, 10)

    model = MLP(vocab_size=tokenizer.vocab_size, context_size=10, embed_dim=8)
    logits = model(x)
    assert logits.size() == (5, tokenizer.vocab_size)


def test_cross_entropy_loss_is_scalar():
    train_data = get_splits(data)["train"]
    x, y = get_batch(split=train_data, batch_size=5, context_size=10)
    model = MLP(vocab_size=tokenizer.vocab_size, context_size=10, embed_dim=8)

    logits = model(x)
    loss = F.cross_entropy(logits, y)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
