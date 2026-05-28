import jax
import jax.numpy as jnp

from spjax.levels import CompressedSpec, CompressedStorage, SparseLevel
from spjax.tensor import SparseTensor


@jax.jit(static_argnums=(2,))
def spdot(a, b, size):
    _, idx_a, idx_b = jnp.intersect1d(
        a.lvls[0].storage.crd,
        b.lvls[0].storage.crd,
        size=size,
        fill_value=0,
        return_indices=True,
    )

    products = a.values[idx_a] * b.values[idx_b]
    return jnp.sum(products)


@jax.jit(static_argnums=(2,))
def spdot_densify(a, b, size):
    a_crd = a.lvls[0].storage.crd
    b_crd = b.lvls[0].storage.crd
    dense_a = jnp.zeros(size, dtype=a.values.dtype)
    dense_a = dense_a.at[a_crd].set(a.values)
    products = dense_a[b_crd] * b.values
    return jnp.sum(products)


def main() -> None:
    vals = jnp.array([1, 2, 3, 4])
    shape = (10,)
    lvl = SparseLevel(
        CompressedSpec(),
        CompressedStorage(pos=jnp.array([0, 4]), crd=jnp.array([0, 1, 5, 7])),
    )

    a = SparseTensor(vals, shape, [lvl])
    b = SparseTensor(vals, shape, [lvl])

    size = a.lvls[0].storage.pos[1].item()
    print(spdot(a, b, size))
    print(spdot.lower(a, b, size).as_text())

    size = a.shape[0]
    print(spdot_densify(a, b, size))
    print(spdot_densify.lower(a, b, size).as_text())


if __name__ == "__main__":
    main()
