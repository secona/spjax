from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .tensor import SparseTensor


# ------------------------------------------------------------------------------
# sparse_add
# ------------------------------------------------------------------------------


def sparse_add(a: SparseTensor, b: SparseTensor) -> jax.Array:
    a_vals = a.values
    a_row = a.lvls[0].crd
    a_col = a.lvls[1].crd

    b_vals = b.values
    b_row = b.lvls[0].crd
    b_col = b.lvls[1].crd

    return _sparse_add_jit(
        a_vals,
        a_row,
        a_col,
        b_vals,
        b_row,
        b_col,
        shape=a.shape,
    )


@partial(jax.jit, static_argnames=("shape",))
def _sparse_add_jit(
    a_vals,
    a_row,
    a_col,
    b_vals,
    b_row,
    b_col,
    *,
    shape,
):
    out = jnp.zeros(shape, dtype=a_vals.dtype)

    def scatter_a(i, acc):
        r = a_row[i]
        c = a_col[i]
        return acc.at[r, c].add(a_vals[i])

    def scatter_b(i, acc):
        r = b_row[i]
        c = b_col[i]
        return acc.at[r, c].add(b_vals[i])

    out = lax.fori_loop(0, a_vals.shape[0], scatter_a, out)
    out = lax.fori_loop(0, b_vals.shape[0], scatter_b, out)
    return out


# ------------------------------------------------------------------------------
# sparse_mul  (placeholder)
# ------------------------------------------------------------------------------


def sparse_mul(a: SparseTensor, b: SparseTensor) -> jax.Array:
    del a, b
    raise NotImplementedError("sparse_mul not yet implemented")


# ------------------------------------------------------------------------------
# sparse_dot  (placeholder)
# ------------------------------------------------------------------------------


def sparse_dot(a: SparseTensor, x: jax.Array) -> jax.Array:
    del a, x
    raise NotImplementedError("sparse_dot not yet implemented")
