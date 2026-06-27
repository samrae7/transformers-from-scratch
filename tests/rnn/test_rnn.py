from language_modeling.data import encode_data, get_splits
from language_modeling.tokenizer import CharTokenizer
from rnn.rnn import RNN, get_batch


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
