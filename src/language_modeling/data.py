from pathlib import Path
from typing import TypedDict

import torch

from language_modeling.tokenizer import CharTokenizer


class DataSplits(TypedDict):
    train: torch.Tensor
    validation: torch.Tensor


def get_corpus_text(path: str | Path = "input.txt") -> str:
    return Path(path).read_text()


def encode_data(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def get_splits(data: torch.Tensor) -> DataSplits:
    data_length = data.size()[0]
    split_idx = data_length * 90 // 100
    train_data = data[:split_idx]
    validation_data = data[split_idx:]
    return {"train": train_data, "validation": validation_data}
