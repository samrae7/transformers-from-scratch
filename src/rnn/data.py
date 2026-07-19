import random

import torch


def get_batch(
    split: torch.Tensor, batch_size: int = 5, sequence_length: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Output tuple:
    x: input sequence. Shape: batch_size, sequence_length
    y: target sequence, same as x but shifted left. Shape: batch_size, sequence_length
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
