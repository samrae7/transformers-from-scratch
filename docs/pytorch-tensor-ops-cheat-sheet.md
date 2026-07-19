# PyTorch Tensor Operations Cheat Sheet

Quick reference for multiplication-style operations that show up in neural
network code.

```text
a * b           elementwise products
(a * b).sum()   dot product
a @ b           dot product for 1D vectors, matrix multiply for matrices
torch.cross     cross product for 3D vectors
```

In neural nets, you mostly use `*` and `@`.

Dot products appear inside matrix multiplication, attention scores, and linear
layers. Cross products are rarely relevant for standard neural-network layers.

## Examples

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20, 30])

a * b
# tensor([10, 40, 90])

(a * b).sum()
# tensor(140)

a @ b
# tensor(140)
```

For 2D tensors, `@` performs matrix multiplication:

```python
A = torch.tensor([[1, 2],
                  [3, 4]])

B = torch.tensor([[10, 20],
                  [30, 40]])

A * B
# tensor([[ 10,  40],
#         [ 90, 160]])

A @ B
# tensor([[ 70, 100],
#         [150, 220]])
```

The important distinction for LSTMs is:

```text
* means pointwise gate application
@ means dimension mixing through dot products / matrix multiplication
```
