import jax
import jax.numpy as jnp

from spjax import encoding
from spjax.tensor import SparseTensor

@jax.jit(static_argnums=(2,))
def spdot(a, b, size):
    _, idx_a, idx_b = jnp.intersect1d(
        a.lvls[0].crd,
        b.lvls[0].crd,
        size=size,
        fill_value=0,
        return_indices=True
    )

    products = a.values[idx_a] * b.values[idx_b]
    return jnp.sum(products)

def main() -> None:
    vals = jnp.array([1, 2, 3, 4])
    shape = (10,)
    lvl = encoding.CompressedLevel(
        pos=jnp.array([0, 4]),
        crd=jnp.array([0, 1, 5, 7])
    )

    a = SparseTensor(vals, shape, [lvl])
    b = SparseTensor(vals, shape, [lvl])
    size = a.lvls[0].pos[1].item()
    print(spdot(a, b, size))
    print(spdot.lower(a, b, size).as_text())

if __name__ == '__main__':
    main()
