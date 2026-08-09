import torch

from language_modeling.tokenizer import CharTokenizer
from rnn.data import get_batch
from transformer.single_head import SingleHeadTransformer


def test_single_head():
    batch_size = 2
    sequence_length = 10
    embed_dim = 4
    head_dim = 4
    tokenizer = CharTokenizer("asfcadscnadoscmnadocnadoscnmasocnasocamnscoasnca")
    single_head = SingleHeadTransformer(
        embed_dim=embed_dim,
        head_dim=head_dim,
        tokenizer=tokenizer,
        max_sequence_length=sequence_length,
    )
    x = torch.randint(0, tokenizer.vocab_size, size=(batch_size, sequence_length))
    logits = single_head(x)
    assert logits.shape == (batch_size, sequence_length, tokenizer.vocab_size)
