from pathlib import Path
import numpy as np

file = Path("input.txt")
text = file.read_text()
print(f"length: {len(text)}")
print(f"first 100 chars:{text[:100]}")

chars = sorted(set(text))
vocab_size = np.unique(chars).size
print(f"{vocab_size=}")

