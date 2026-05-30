import jax
import jax.numpy as jnp
from jax import lax

from spjax.levels import CompressedSpec, DenseSpec, SingletonSpec, SparseLevel
from spjax.storage import CompressedStorage, DenseStorage, SingletonStorage


def sparse_add(a: "SparseTensor", b: "SparseTensor") -> jax.Array:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if len(a.lvls) != len(b.lvls):
        raise ValueError(f"Level count mismatch: {len(a.lvls)} vs {len(b.lvls)}")

    for i, (la, lb) in enumerate(zip(a.lvls, b.lvls)):
        if type(la.spec) is not type(lb.spec):
            raise ValueError(
                f"Level {i} format mismatch: {type(la.spec).__name__} vs {type(lb.spec).__name__}"
            )

    out = jnp.zeros(a.shape, dtype=a.values.dtype)
    out = _scatter_tensor(a, out)
    out = _scatter_tensor(b, out)
    return out


def sparse_dot(a: "SparseTensor", x: jax.Array) -> jax.Array:
    from spjax.tensor import SparseTensor

    if isinstance(x, SparseTensor):
        x_dense = jnp.zeros(x.shape, dtype=x.values.dtype)
        x_dense = _scatter_tensor(x, x_dense)
        return sparse_dot(a, x_dense)

    if a.shape[1] != x.shape[0]:
        raise ValueError(f"Shape mismatch for dot: {a.shape} @ {x.shape}")

    result_shape: tuple[int, ...]
    if x.ndim == 1:
        result_shape = (a.shape[0],)
    elif x.ndim == 2:
        result_shape = (a.shape[0], x.shape[1])
    else:
        raise ValueError(f"dot only supports 1D or 2D x, got {x.ndim}D")

    result_dtype = jnp.result_type(a.values.dtype, x.dtype)
    out = jnp.zeros(result_shape, dtype=result_dtype)
    return _gather_dot_tensor(a, x, out)


def _gather_dot_inner(
    lvl1: SparseLevel,
    vals: jax.Array,
    parent_pos: int,
    coord0: int,
    x: jax.Array,
    out: jax.Array,
) -> jax.Array:
    p1_begin, p1_end = lvl1.iter_bounds(parent_pos)

    def inner_body(p1, out):
        coord1 = lvl1.iter_coord(p1)
        vi = lvl1.value_index(parent_pos, p1)
        return out.at[coord0].add(vals[vi] * x[coord1])

    return lax.fori_loop(p1_begin, p1_end, inner_body, out)


def _gather_dot_tensor(a: "SparseTensor", x: jax.Array, out: jax.Array) -> jax.Array:
    lvls = a.lvls
    vals = a.values

    if len(lvls) != 2:
        raise NotImplementedError("dot currently supports only 2D tensors")

    if vals.size == 0:
        return out

    lvl0 = lvls[0]
    lvl1 = lvls[1]

    p0_begin, p0_end = lvl0.iter_bounds(0)

    def outer_body(p0, out):
        coord0 = lvl0.iter_coord(p0)
        return _gather_dot_inner(lvl1, vals, p0, coord0, x, out)

    return lax.fori_loop(p0_begin, p0_end, outer_body, out)


def _scatter_inner(
    lvl1: SparseLevel,
    vals: jax.Array,
    parent_pos: int,
    coord0: int,
    out: jax.Array,
) -> jax.Array:
    p1_begin, p1_end = lvl1.iter_bounds(parent_pos)

    def inner_body(p1, out):
        coord1 = lvl1.iter_coord(p1)
        vi = lvl1.value_index(parent_pos, p1)
        return out.at[coord0, coord1].add(vals[vi])

    return lax.fori_loop(p1_begin, p1_end, inner_body, out)


def _scatter_tensor(a: "SparseTensor", out: jax.Array) -> jax.Array:
    lvls = a.lvls
    vals = a.values

    if len(lvls) != 2:
        raise NotImplementedError("add currently supports only 2D tensors")

    if vals.size == 0:
        return out

    lvl0 = lvls[0]
    lvl1 = lvls[1]

    p0_begin, p0_end = lvl0.iter_bounds(0)

    def outer_body(p0, out):
        coord0 = lvl0.iter_coord(p0)
        return _scatter_inner(lvl1, vals, p0, coord0, out)

    return lax.fori_loop(p0_begin, p0_end, outer_body, out)
