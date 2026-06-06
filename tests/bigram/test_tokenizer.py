from bigram.bigram import decode, encode, itos, stoi, vocab_size


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
