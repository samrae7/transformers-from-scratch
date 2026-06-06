from bigram.bigram import BigramModel, stoi


def test_get_next_token():
    model = BigramModel()
    start = stoi["\n"]
    next_token = model._get_next_token(start)

    assert isinstance(next_token, int)


def test_generate():
    model = BigramModel()
    start_char = "\n"
    generated = model.generate(start_char, 10)
    assert len(generated) == 10
