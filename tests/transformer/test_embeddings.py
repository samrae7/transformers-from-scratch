import torch

from transformer.attention import TokenAndPositionEmbedding


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
