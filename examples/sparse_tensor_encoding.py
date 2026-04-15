import jax.numpy as jnp

from spjax import DimLevelType, SparseEncoding, SparseTensor, SparseTensorSpec


def build_csr_sparse_tensor() -> SparseTensor:
    # Represents the dense matrix:
    # [[1, 0, 2, 0],
    #  [0, 0, 0, 0],
    #  [0, 3, 0, 4]]
    data = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    col_idx = jnp.array([0, 2, 1, 3], dtype=jnp.int32)
    row_ptr = jnp.array([0, 2, 2, 4], dtype=jnp.int32)

    encoding = SparseEncoding(
        lvl_types=(DimLevelType.DENSE, DimLevelType.COMPRESSED),
        lvl_order=(0, 1),
        ordered=(True, True),
        unique=(True, True),
        pos_width=32,
        crd_width=32,
    )

    spec = SparseTensorSpec(shape=(3, 4), dtype=jnp.float32, encoding=encoding)

    return SparseTensor(
        buffers=(data, col_idx, row_ptr),
        buffer_kinds=("data", "indices", "indptr"),
        spec=spec,
    )


def main() -> None:
    tensor = build_csr_sparse_tensor()

    print("shape:", tensor.spec.shape)
    print("dtype:", tensor.spec.dtype)
    print("level types:", tensor.spec.encoding.lvl_types)
    print("level order:", tensor.spec.encoding.lvl_order)
    print("ordered per level:", tensor.spec.encoding.ordered)
    print("unique per level:", tensor.spec.encoding.unique)

    print("data:", tensor.buffer("data"))
    print("indices:", tensor.buffer("indices"))
    print("indptr:", tensor.buffer("indptr"))


if __name__ == "__main__":
    main()
