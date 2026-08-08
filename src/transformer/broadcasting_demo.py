import torch


def show(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(tensor.shape)}")
    print(tensor)
    print()


def main() -> None:
    sequence_length = 4
    positions = torch.arange(sequence_length)

    column_positions = positions[:, None]
    row_positions = positions[None, :]

    show("positions", positions)
    show("column_positions", column_positions)
    show("row_positions", row_positions)

    expanded_columns = column_positions.expand(sequence_length, sequence_length)
    expanded_rows = row_positions.expand(sequence_length, sequence_length)

    show("expanded_columns", expanded_columns)
    show("expanded_rows", expanded_rows)

    equality_comparison = column_positions == row_positions
    show("column_positions == row_positions", equality_comparison)

    assert column_positions.shape == (sequence_length, 1)
    assert row_positions.shape == (1, sequence_length)
    assert equality_comparison.shape == (sequence_length, sequence_length)
    assert torch.equal(
        equality_comparison,
        expanded_columns == expanded_rows,
    )

    print("Try another comparison in the debugger.")
    print("Aim for True on the diagonal and below it, and False above it.")

    greater_than_comparison = column_positions >= row_positions
    show("greater_than_comparison", greater_than_comparison)


if __name__ == "__main__":
    main()
