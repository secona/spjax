from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from spjax.merge_lattice import Dimension, IndexVar

from .tensor import SparseTensor


# ------------------------------------------------------------------------------
# sparse_add
# ------------------------------------------------------------------------------


def _scatter_inner(lvl1, vals, parent_pos, coord0, out):
    """Scatter inner level of a 2D sparse tensor into a dense output."""
    if lvl1.supports_coord_pos_iter:
        p1_begin, p1_end = lvl1.pos_bounds(parent_pos)

        def inner_body(p1, out):
            coord1, _ = lvl1.pos_access(p1)
            return out.at[coord0, coord1].add(vals[p1])

        return lax.fori_loop(p1_begin, p1_end, inner_body, out)

    elif lvl1.supports_coord_value_iter:
        i1_begin, i1_end = lvl1.coord_bounds(parent_pos)

        def inner_body(i1, out):
            coord1, _ = lvl1.coord_access(parent_pos, i1)
            inner_size = i1_end - i1_begin  # == lvl1.size
            return out.at[coord0, coord1].add(vals[coord0 * inner_size + coord1])

        return lax.fori_loop(i1_begin, i1_end, inner_body, out)

    else:
        raise ValueError(
            f"Level 1 format {type(lvl1).__name__} does not support iteration"
        )


def _scatter_tensor(tensor: SparseTensor, out: jax.Array) -> jax.Array:
    """Generic scatter of a 2D sparse tensor into a dense output array."""
    lvls = tensor.lvls
    vals = tensor.values

    if len(lvls) != 2:
        raise NotImplementedError("sparse_add currently supports only 2D tensors")

    if vals.size == 0:
        return out

    lvl0 = lvls[0]
    lvl1 = lvls[1]

    if lvl0.supports_coord_pos_iter:
        p0_begin, p0_end = lvl0.pos_bounds(0)

        def outer_body(p0, out):
            coord0, _ = lvl0.pos_access(p0)
            return _scatter_inner(lvl1, vals, p0, coord0, out)

        return lax.fori_loop(p0_begin, p0_end, outer_body, out)

    elif lvl0.supports_coord_value_iter:
        i0_begin, i0_end = lvl0.coord_bounds(0)

        def outer_body(i0, out):
            coord0, _ = lvl0.coord_access(0, i0)
            return _scatter_inner(lvl1, vals, coord0, coord0, out)

        return lax.fori_loop(i0_begin, i0_end, outer_body, out)

    else:
        raise ValueError(
            f"Level 0 format {type(lvl0).__name__} does not support iteration"
        )


def sparse_add(a: SparseTensor, b: SparseTensor) -> jax.Array:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    if len(a.lvls) != len(b.lvls):
        raise ValueError(f"Level count mismatch: {len(a.lvls)} vs {len(b.lvls)}")

    for i, (la, lb) in enumerate(zip(a.lvls, b.lvls)):
        if type(la) != type(lb):
            raise ValueError(
                f"Level {i} format mismatch: {type(la).__name__} vs {type(lb).__name__}"
            )

    out = jnp.zeros(a.shape, dtype=a.values.dtype)
    out = _scatter_tensor(a, out)
    out = _scatter_tensor(b, out)
    return out


# ------------------------------------------------------------------------------
# sparse_dot
# ------------------------------------------------------------------------------


def _gather_dot_inner(lvl1, vals, parent_pos, coord0, x, out):
    """Gather inner level of a 2D sparse tensor and dot with x."""
    if lvl1.supports_coord_pos_iter:
        p1_begin, p1_end = lvl1.pos_bounds(parent_pos)

        def inner_body(p1, out):
            coord1, _ = lvl1.pos_access(p1)
            return out.at[coord0].add(vals[p1] * x[coord1])

        return lax.fori_loop(p1_begin, p1_end, inner_body, out)

    elif lvl1.supports_coord_value_iter:
        i1_begin, i1_end = lvl1.coord_bounds(parent_pos)

        def inner_body(i1, out):
            coord1, _ = lvl1.coord_access(parent_pos, i1)
            inner_size = i1_end - i1_begin  # == lvl1.size
            return out.at[coord0].add(vals[coord0 * inner_size + coord1] * x[coord1])

        return lax.fori_loop(i1_begin, i1_end, inner_body, out)

    else:
        raise ValueError(
            f"Level 1 format {type(lvl1).__name__} does not support iteration"
        )


def _gather_dot_tensor(tensor: SparseTensor, x: jax.Array, out: jax.Array) -> jax.Array:
    """Generic gather-dot of a 2D sparse tensor with a dense vector/matrix."""
    lvls = tensor.lvls
    vals = tensor.values

    if len(lvls) != 2:
        raise NotImplementedError("sparse_dot currently supports only 2D tensors")

    if vals.size == 0:
        return out

    lvl0 = lvls[0]
    lvl1 = lvls[1]

    if lvl0.supports_coord_pos_iter:
        p0_begin, p0_end = lvl0.pos_bounds(0)

        def outer_body(p0, out):
            coord0, _ = lvl0.pos_access(p0)
            return _gather_dot_inner(lvl1, vals, p0, coord0, x, out)

        return lax.fori_loop(p0_begin, p0_end, outer_body, out)

    elif lvl0.supports_coord_value_iter:
        i0_begin, i0_end = lvl0.coord_bounds(0)

        def outer_body(i0, out):
            coord0, _ = lvl0.coord_access(0, i0)
            return _gather_dot_inner(lvl1, vals, coord0, coord0, x, out)

        return lax.fori_loop(i0_begin, i0_end, outer_body, out)

    else:
        raise ValueError(
            f"Level 0 format {type(lvl0).__name__} does not support iteration"
        )


def sparse_dot(a: SparseTensor, x: jax.Array) -> jax.Array:
    if isinstance(x, SparseTensor):
        x_dense = _scatter_tensor(x, jnp.zeros(x.shape, dtype=x.values.dtype))
        return sparse_dot(a, x_dense)

    if a.shape[1] != x.shape[0]:
        raise ValueError(f"Shape mismatch for sparse_dot: {a.shape} @ {x.shape}")

    result_shape: tuple[int, ...]
    if x.ndim == 1:
        result_shape = (a.shape[0],)
    elif x.ndim == 2:
        result_shape = (a.shape[0], x.shape[1])
    else:
        raise ValueError(f"sparse_dot only supports 1D or 2D x, got {x.ndim}D")

    result_dtype = jnp.result_type(a.values.dtype, x.dtype)
    out = jnp.zeros(result_shape, dtype=result_dtype)
    return _gather_dot_tensor(a, x, out)
