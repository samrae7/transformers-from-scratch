import torch
import random

from language_modeling.data import encode_data, get_corpus_text, get_splits
from language_modeling.tokenizer import CharTokenizer


def get_batch(split: torch.Tensor, batch_size: int, sequence_length) ->tuple[torch.Tensor, torch.Tensor]:
    """
    x: input sequence. Shape: batch_size, sequence_length
    y: target sequence (same as x but shifted to the left): Shape: batch_size, sequence_length
    """
    x = []
    y = []
    greatest_start_idx = len(split) - sequence_length - 1
    for _ in range(batch_size):
        start_idx = random.randint(0, greatest_start_idx)
        end_idx = start_idx + sequence_length
        x_row = split[start_idx:end_idx]
        y_row = split[start_idx + 1 : end_idx + 1]
        x.append(x_row)
        y.append(y_row)
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


if __name__ == "__main__":
    text = get_corpus_text()
    tokenizer = CharTokenizer(text)
    data = encode_data(text, tokenizer=tokenizer)
    splits = get_splits(data)
    train_data = splits["train"]
