import torch

from language_modeling.tokenizer import CharTokenizer
from transformer.multi_head_attention import MultiHeadAttention


def test_multi_head():
    batch_size = 2
    sequence_length = 10
    embed_dim = 8
    head_dim = 4
    num_heads = 2
    tokenizer = CharTokenizer("asfcadscnadoscmnadocnadoscnmasocnasocamnscoasnca")
    x = torch.randn(batch_size, sequence_length, embed_dim)

    multi_head = MultiHeadAttention(
        num_heads=num_heads, head_dim=head_dim, embed_dim=embed_dim
    )
    logits = multi_head(x)

    assert logits.shape == (batch_size, sequence_length, tokenizer.vocab_size)
