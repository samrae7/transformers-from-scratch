from bigram.bigram import tokenizer


def test_stoi_and_itos_are_inverses():
    for char, idx in tokenizer.stoi.items():
        assert tokenizer.itos[idx] == char


def test_encode_decode_round_trip():
    sample = "First Citizen:\n"

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    assert decoded == sample


def test_encode_returns_integer_token_ids():
    sample = "hello"

    encoded = tokenizer.encode(sample)

    assert all(isinstance(idx, int) for idx in encoded)
    assert all(0 <= idx < tokenizer.vocab_size for idx in encoded)


def test_decode_returns_string():
    ids = [
        tokenizer.stoi["h"],
        tokenizer.stoi["e"],
        tokenizer.stoi["l"],
        tokenizer.stoi["l"],
        tokenizer.stoi["o"],
    ]

    decoded = tokenizer.decode(ids)

    assert decoded == "hello"
