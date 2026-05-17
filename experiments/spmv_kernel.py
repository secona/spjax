import argparse

import jax
from spjax import SparseTensor, sparse_dot


@jax.jit
def spmv(X: SparseTensor, y: jax.Array) -> jax.Array:
    return sparse_dot(X, y)


def main() -> None:
    parser = argparse.ArgumentParser(description="JIT spMV kernel for profiling")
    parser.add_argument("matrix", help="Path to a .mtx file")
    args = parser.parse_args()

    X = SparseTensor.from_file(args.matrix)
    num_cols = X.shape[1]
    dtype = X.values.dtype

    key = jax.random.PRNGKey(42)
    y = jax.random.normal(key, (num_cols,), dtype=dtype)

    spmv(X, y).block_until_ready()


if __name__ == "__main__":
    main()
