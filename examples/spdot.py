import jax
import jax.numpy as jnp

from spjax import encoding
from spjax.tensor import SparseTensor
from spjax.primitive import sparse_dot

@jax.jit
def spdot(a, b):
    return sparse_dot(a, b)

def main() -> None:
    vals = jnp.array([1, 2, 3, 4])
    shape = (10,)
    lvl = encoding.CompressedLevel(
        pos=jnp.array([0, 4]),
        crd=jnp.array([0, 1, 5, 7])
    )

    a = SparseTensor(vals, shape, [lvl])
    b = SparseTensor(vals, shape, [lvl])
    print(spdot(a, b))
    print(spdot.lower(a, b).as_text())

if __name__ == '__main__':
    main()
