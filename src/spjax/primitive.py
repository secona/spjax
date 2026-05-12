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


def sparse_dot(X: SparseTensor, y: jax.Array) -> jax.Array:
    return sparse_dot_p.bind(X.values, X.pos, X.crd, y, shape=X.shape)
