from bigram.bigram import BigramModel, batch_size, block_size, get_batch, test_data, vocab_size


def test_bigram_model():
    model = BigramModel()
    idx, targets = get_batch(test_data)
    loss, logits = model.forward(idx, targets)
    assert loss.ndim == 0
    assert logits.size() == (batch_size, block_size, vocab_size)


def test_forward_returns_loss_as_none_when_targets_not_provided():
    model = BigramModel()
    idx, _targets = get_batch(test_data)
    loss, logits = model.forward(idx)
    assert loss is None
    assert logits.size() == (batch_size, block_size, vocab_size)
