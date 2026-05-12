import jax
import jax.numpy as jnp
from jax._src.core import Primitive, ShapedArray
from jax.interpreters import mlir

from .tensor import SparseTensor

sparse_dot_p = Primitive("sparse_dot")


def sparse_dot_abstract_eval(values, pos, crd, dense_vec, *, shape):
    del pos, crd, dense_vec
    return ShapedArray((shape[0],), values.dtype)


def sparse_dot_impl(values, pos, crd, dense_vec, *, shape):
    del pos
    out = jnp.zeros(shape[0], dtype=values.dtype)
    out = out.at[crd[0]].add(values * dense_vec[crd[1]])
    return out


sparse_dot_p.def_abstract_eval(sparse_dot_abstract_eval)
sparse_dot_p.def_impl(sparse_dot_impl)
mlir.register_lowering(
    sparse_dot_p, mlir.lower_fun(sparse_dot_impl, multiple_results=False)
)


def _spmm(X: SparseTensor, Y: SparseTensor) -> jax.Array:
    M, K = X.shape
    K2, N = Y.shape
    assert K == K2

    valid = X.crd[1][:, None] == Y.crd[0][None, :]
    vals = X.values[:, None] * Y.values[None, :]
    vals = jnp.where(valid, vals, 0)

    rows = X.crd[0][:, None]
    cols = Y.crd[1][None, :]

    out = jnp.zeros((M, N), dtype=X.values.dtype)
    return out.at[rows, cols].add(vals)


def sparse_dot(X: SparseTensor, y: SparseTensor | jax.Array) -> jax.Array:
    if isinstance(y, SparseTensor):
        return _spmm(X, y)
    return sparse_dot_p.bind(X.values, X.pos, X.crd, y, shape=X.shape)
