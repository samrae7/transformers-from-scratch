import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bigram


def test_stoi_and_itos_are_inverses():
    for char, idx in bigram.stoi.items():
        assert bigram.itos[idx] == char


def test_encode_decode_round_trip():
    sample = "First Citizen:\n"

    encoded = bigram.encode(sample)
    decoded = bigram.decode(encoded)

    assert decoded == sample


def test_encode_returns_integer_token_ids():
    sample = "hello"

    encoded = bigram.encode(sample)

    assert all(isinstance(idx, int) for idx in encoded)
    assert all(0 <= idx < bigram.vocab_size for idx in encoded)


def test_decode_returns_string():
    ids = [bigram.stoi["h"], bigram.stoi["e"], bigram.stoi["l"], bigram.stoi["l"], bigram.stoi["o"]]

    decoded = bigram.decode(ids)

    assert decoded == "hello"
