import torch

from language_modeling.tokenizer import CharTokenizer
from rnn.data import get_batch
from transformer.transformer import Transformer


def test_forward():
    batch_size = 2
    sequence_length = 10
    embed_dim = 8
    head_dim = 4
    tokenizer = CharTokenizer("asfcadscnadoscmnadocnadoscnmasocnasocamnscoasnca")
    single_head = Transformer(
        embed_dim=embed_dim,
        head_dim=head_dim,
        tokenizer=tokenizer,
        max_sequence_length=sequence_length,
        num_heads=2,
    )
    x = torch.randint(0, tokenizer.vocab_size, size=(batch_size, sequence_length))
    logits = single_head(x)
    assert logits.shape == (batch_size, sequence_length, tokenizer.vocab_size)


def test_generate():
    sequence_length = 10
    embed_dim = 8
    head_dim = 4
    tokenizer = CharTokenizer("asfcadscnadoscmnadocnadoscnmasocn asocamnscoasncahello")
    single_head = Transformer(
        embed_dim=embed_dim,
        head_dim=head_dim,
        tokenizer=tokenizer,
        max_sequence_length=sequence_length,
        num_heads=2,
    )
    result = single_head.generate("hello", length=4)
    assert result is not None
