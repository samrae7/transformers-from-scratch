import torch

from transformer.attention import (
    SingleHeadCausalSelfAttention,
    TokenAndPositionEmbedding,
)


def test_token_and_position_embeddings_have_expected_shapes_and_encode_position():
    batch_size = 2
    sequence_length = 4
    max_sequence_length = 4
    embed_dim = 8
    vocab_size = 10

    embeddings = TokenAndPositionEmbedding(
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embed_dim=embed_dim,
    )
    idx = torch.tensor(
        [
            [3, 3, 5, 7],
            [1, 2, 4, 6],
        ]
    )

    token_embeddings = embeddings.token_embeddings(idx)
    positions = torch.arange(sequence_length)
    position_embeddings = embeddings.position_embeddings(positions)
    x = embeddings(idx)

    assert token_embeddings.shape == (batch_size, sequence_length, embed_dim)
    assert position_embeddings.shape == (sequence_length, embed_dim)
    assert x.shape == (batch_size, sequence_length, embed_dim)
    assert torch.allclose(
        x,
        token_embeddings + position_embeddings,
    )
    assert not torch.allclose(x[0, 0], x[0, 1])


def test_single_head_causal_self_attention():

    batch_size = 2
    sequence_length = 4
    embed_dim = 8
    head_dim = 4

    attention = SingleHeadCausalSelfAttention(embed_dim=embed_dim, head_dim=head_dim)

    x = torch.randn(batch_size, sequence_length, embed_dim)

    q = attention.w_q(x)
    k = attention.w_k(x)
    v = attention.w_v(x)

    assert q.shape == (batch_size, sequence_length, head_dim)
    assert k.shape == (batch_size, sequence_length, head_dim)
    assert v.shape == (batch_size, sequence_length, head_dim)

    o = attention(x)
    assert o.shape == (batch_size, sequence_length, head_dim)

    o.sum().backward()

    assert attention.w_q.weight.grad is not None
    assert attention.w_k.weight.grad is not None
    assert attention.w_v.weight.grad is not None


def test_scaled_dot_product_scores():
    head_dim = 4
    attention = SingleHeadCausalSelfAttention(
        embed_dim=8,
        head_dim=head_dim,
    )

    q = torch.ones(1, 2, 4)
    k = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ]
    )

    scores = attention.scaled_dot_product(q, k)

    expected = torch.tensor(
        [
            [
                [2.0, 4.0],
                [2.0, 4.0],
            ]
        ]
    )

    assert scores.shape == (1, 2, 2)
    assert torch.allclose(scores, expected)
