import jax
from spjax import SparseTensor


def print_sparse_matrix(m: SparseTensor):
    rows = m.row.tolist()
    cols = m.col.tolist()
    vals = m.values.tolist()

    num_rows = max(m.row) + 1
    num_cols = max(m.col) + 1

    sparse_map = {}
    for r, c, v in zip(rows, cols, vals):
        coords = (int(r), int(c))
        sparse_map[coords] = sparse_map.get(coords, 0) + v

    for r in range(num_rows):
        row_str = []
        for c in range(num_cols):
            val = int(sparse_map.get((r, c), 0))
            row_str.append(f"{val:>4}" if val != 0 else "   .")
        print(" ".join(row_str))


@jax.jit
def add(x, y):
    return x + y


def main() -> None:
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_file("./matrix/Hamrle1.mtx")
    c = add(a, b)
    print_sparse_matrix(c)


if __name__ == "__main__":
    main()
