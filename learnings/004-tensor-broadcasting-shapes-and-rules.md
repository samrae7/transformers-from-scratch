# Tensor Broadcasting: Shapes and Compatibility Rules

## Core Idea

Broadcasting lets elementwise operations combine tensors with different but
compatible shapes. PyTorch behaves as though size-one dimensions were expanded
to match the other tensor, usually without physically copying the repeated
values.

Examples of elementwise operations that use broadcasting include addition,
subtraction, multiplication, and comparisons.

## The Compatibility Rule

Compare tensor dimensions from right to left. At each aligned dimension, the
sizes are compatible when:

1. The sizes are equal.
2. One of the sizes is `1`.
3. One tensor has no dimension there.

A missing leading dimension behaves like a size-one dimension.

For example:

```text
[2, 3, 4]
[   3, 4]
```

Right alignment makes this equivalent to:

```text
[2, 3, 4]
[1, 3, 4]
```

The result has shape `[2,3,4]`.

Different sizes greater than one are not compatible:

```text
[2, 3, 4]
[2, 5, 4]
```

The aligned dimensions `3` and `5` neither match nor contain a `1`, so the
operation fails.

## Inserting Size-One Dimensions

Given a one-dimensional tensor:

```python
positions = torch.arange(4)
```

its shape is `[4]`. Using `None` in an index inserts a new dimension where the
`None` appears:

```python
column_positions = positions[:, None]  # [4, 1]
row_positions = positions[None, :]     # [1, 4]
```

These are equivalent to:

```python
column_positions = positions.unsqueeze(1)
row_positions = positions.unsqueeze(0)
```

`None` and `unsqueeze` change the view of the tensor; they do not add or copy
values.

## Broadcasting a Column Against a Row

The shapes `[T,1]` and `[1,T]` broadcast to `[T,T]`:

```text
column: [T, 1]
row:    [1, T]
result: [T, T]
```

Conceptually, the column is repeated across columns:

```text
[[0, 0, 0],
 [1, 1, 1],
 [2, 2, 2]]
```

and the row is repeated across rows:

```text
[[0, 1, 2],
 [0, 1, 2],
 [0, 1, 2]]
```

PyTorch normally performs this expansion virtually. The elementwise operation
creates the result tensor, but the expanded input views do not require full
copies.

## Broadcasted Comparisons

Comparison operators return boolean tensors. For example:

```python
equality = column_positions == row_positions
```

produces a `[T,T]` boolean tensor with `True` on the diagonal.

A causal relationship compares each query position `i` with each key position
`j`:

```text
allow attention when j <= i
```

Using the column for query positions and the row for key positions:

```python
causal_mask = column_positions >= row_positions
```

produces:

```text
[[ True, False, False, False],
 [ True,  True, False, False],
 [ True,  True,  True, False],
 [ True,  True,  True,  True]]
```

Each row represents one query position. Each column represents one key
position.

## Broadcasting Across a Batch

Attention scores have shape `[B,T,T]`, while a causal mask can have shape
`[T,T]`:

```text
scores: [B, T, T]
mask:      [T, T]
```

Right alignment treats the mask as `[1,T,T]`. Its two `T` dimensions match the
score dimensions, and the missing batch dimension broadcasts from `1` to `B`.
The same causal rule is therefore applied to every batch example.

## Quick Shape Checklist

When deciding whether two tensors broadcast:

1. Write their shapes one above the other, aligned on the right.
2. Compare dimensions from right to left.
3. Require equal sizes, a size of `1`, or a missing leading dimension.
4. Use the larger compatible size for each result dimension.

```text
[2, 3, 4] with [3, 4]    -> [2, 3, 4]
[2, 3, 4] with [2, 1, 4] -> [2, 3, 4]
[T, 1]    with [1, T]    -> [T, T]
[B, T, T] with [T, T]    -> [B, T, T]
[2, 3, 4] with [2, 5, 4] -> incompatible
```

## Key Takeaway

Broadcasting is not an arbitrary reshaping rule. It is a mechanical process of
right-aligning shapes and expanding only size-one or missing leading
dimensions. In attention, this makes it possible to construct one `[T,T]`
causal mask and apply it uniformly across a batch of `[B,T,T]` score matrices.
