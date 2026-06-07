# Linear Layers: Mathematics vs PyTorch

## 1. The mathematical view

Suppose:

- Input dimension = 4
- Output dimension = 3

A linear layer is:

$$
y = Wx
$$

where:

$$
W \in \mathbb{R}^{3 \times 4}
$$

and

$$
x \in \mathbb{R}^{4 \times 1}
$$

Therefore:

$$
y \in \mathbb{R}^{3 \times 1}
$$

### Shapes

$$
(3 \times 4)(4 \times 1)
=
(3 \times 1)
$$

The important thing:

- Rows of $begin:math:text$W$end:math:text$ correspond to output neurons.
- Columns of $begin:math:text$W$end:math:text$ correspond to input features.

Example:

$$
W=
\begin{bmatrix}
- & - & - & - \\
- & - & - & - \\
- & - & - & -
\end{bmatrix}
$$

Each row contains all weights needed to compute one output feature.

So:

- Row 1 → output neuron 1
- Row 2 → output neuron 2
- Row 3 → output neuron 3

This is why the standard mathematical shape is:

$$
W = (\text{out features}, \text{in features})
$$

---

## 2. Why PyTorch looks different

PyTorch processes batches.

Instead of a single column vector:

$$
x =
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{bmatrix}
$$

it stores many input vectors as rows:

$$
X=
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12
\end{bmatrix}
$$

Shape:

$$
(\text{batch size}, \text{in features})
$$

For this example:

$$
X \in \mathbb{R}^{3 \times 4}
$$

where:

- batch size = 3
- input features = 4

Each row is one training example.

---

## 3. Why PyTorch multiplies by \(W^T\)

PyTorch keeps the weight matrix stored as:

$$
W \in \mathbb{R}^{3 \times 4}
$$

exactly the same convention as mathematics:

$$
(\text{out features}, \text{in features})
$$

However, inputs are stored as row vectors.

To make dimensions line up:

$$
XW^T
$$

is used.

Shapes:

$$
(3 \times 4)(4 \times 3)
=
(3 \times 3)
$$

Result:

$$
Y \in \mathbb{R}^{3 \times 3}
$$

which means:

- 3 examples in the batch
- 3 output features per example

---

## 4. Why not store weights the other way round?

A natural question:

Why not store

$$
W \in \mathbb{R}^{4 \times 3}
$$

and compute

$$
XW
$$

without needing a transpose?

Because the current convention has a useful interpretation:

$$
W = (\text{out features}, \text{in features})
$$

Each row is literally the weight vector for one output neuron.

Example:

$$
W=
\begin{bmatrix}
w_1 \\
w_2 \\
w_3
\end{bmatrix}
$$

where:

$$
w_i \in \mathbb{R}^{4}
$$

Each output neuron owns one row.

This matches the mathematical definition:

$$
y_i = w_i^T x
$$

So the layout makes conceptual sense.

---

## 5. What PyTorch actually does

Internally, `nn.Linear` stores:

```python
weight.shape == (out_features, in_features)
```

and computes:

```python
output = input @ weight.T + bias
```

For example:

```python
nn.Linear(4, 3)
```

stores:

```python
weight.shape == (3, 4)
```

If:

```python
input.shape == (batch_size, 4)
```

then:

```python
output.shape == (batch_size, 3)
```

because:

```python
(batch_size, 4) @ (4, 3)
=
(batch_size, 3)
```

---

## 6. The key insight

There is actually no disagreement between mathematics and PyTorch.

The weight matrix is stored the same way:

$$
W = (\text{out features}, \text{in features})
$$

The only difference is that:

**Mathematics usually writes individual inputs as column vectors.**

$$
y = Wx
$$

whereas

**PyTorch stores batches as row vectors.**

$$
Y = XW^T
$$

The transpose appears purely because batches are stored row-wise, not because the meaning of the weight matrix has changed.

---

## Mental Model

When looking at a PyTorch linear layer:

```python
nn.Linear(in_features=4, out_features=3)
```

think:

```text
weight:
┌──────────────────────┐
│ neuron 1 weights     │
│ neuron 2 weights     │
│ neuron 3 weights     │
└──────────────────────┘

shape = (3, 4)
```

Each row corresponds to one output neuron.

Each neuron takes all 4 input features and produces one output value.

PyTorch stores batches as rows:

```text
batch_size × in_features
```

so the forward pass becomes:

```text
(batch_size × in_features)
    @
(in_features × out_features)
=
(batch_size × out_features)
```

which is implemented as:

```python
output = input @ weight.T
```

The transpose exists only because batches are stored row-wise.
