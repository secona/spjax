from itertools import product

import jax
import jax.numpy as jnp
from jax import lax

from spjax.levels import (
    CompressedSpec,
    CompressedStorage,
    DenseSpec,
    DenseStorage,
    SingletonSpec,
    SingletonStorage,
    SparseLevel,
)


@jax.tree_util.register_pytree_node_class
class SparseTensor:
    values: jax.Array
    shape: tuple[int, ...]
    lvls: list[SparseLevel]

    def __init__(
        self,
        values: jax.Array,
        shape: tuple[int, ...],
        lvls: list[SparseLevel],
    ) -> None:
        self.values = values
        self.shape = shape
        self.lvls = lvls

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def _from_coo(cls, m_coo, shape=None) -> "SparseTensor":
        nnz = m_coo.nnz
        values = jnp.asarray(m_coo.data)
        if shape is None:
            shape = (int(m_coo.row.max()) + 1, int(m_coo.col.max()) + 1)

        compressed_lvl = SparseLevel(
            CompressedSpec(is_unique=False),
            CompressedStorage(pos=jnp.array([0, nnz]), crd=jnp.asarray(m_coo.row)),
        )

        singleton_lvl = SparseLevel(
            SingletonSpec(),
            SingletonStorage(crd=jnp.asarray(m_coo.col)),
        )

        lvls = [compressed_lvl, singleton_lvl]
        return cls(values, shape, lvls)

    @classmethod
    def _from_csr(cls, m_csr, shape=None) -> "SparseTensor":
        indptr = jnp.asarray(m_csr.indptr)
        indices = jnp.asarray(m_csr.indices)
        values = jnp.asarray(m_csr.data)
        if shape is None:
            shape = (m_csr.shape[0], m_csr.shape[1])

        dense_lvl = SparseLevel(DenseSpec(shape[0]), DenseStorage(shape[0]))
        compressed_lvl = SparseLevel(
            CompressedSpec(),
            CompressedStorage(pos=indptr, crd=indices),
        )

        lvls = [dense_lvl, compressed_lvl]
        return cls(values, shape, lvls)

    @classmethod
    def from_file(cls, filename, fmt="coo") -> "SparseTensor":
        try:
            import scipy
        except ImportError:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        m_coo = scipy.io.mmread(filename)
        if fmt == "coo":
            return cls._from_coo(m_coo)
        elif fmt == "csr":
            m_csr = m_coo.tocsr()
            return cls._from_csr(m_csr, shape=m_coo.shape)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    @classmethod
    def from_dense(cls, arr) -> "SparseTensor":
        try:
            import scipy
        except ImportError:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        coo = scipy.sparse.coo_matrix(arr)
        return cls._from_coo(coo, shape=arr.shape)

    # ---------------------------------------------------------------------------
    # String representation
    # ---------------------------------------------------------------------------

    def _collect_entries(self) -> dict[tuple[int, ...], float]:
        sparse_map: dict[tuple[int, ...], float] = {}
        vals = self.values.tolist()

        def walk(level_idx: int, parent_pos: int, coords: tuple[int, ...]):
            lvl = self.lvls[level_idx]
            p_begin, p_end = lvl.iter_bounds(parent_pos)
            for p in range(int(p_begin), int(p_end)):
                coord = int(lvl.iter_coord(p))
                vi = lvl.value_index(parent_pos, p)
                new_coords = coords + (coord,)
                if level_idx == len(self.lvls) - 1:
                    key = new_coords
                    sparse_map[key] = sparse_map.get(key, 0.0) + float(vals[vi])
                else:
                    walk(level_idx + 1, p, new_coords)

        root_lvl = self.lvls[0]
        if isinstance(root_lvl.spec, DenseSpec):
            p_begin, p_end = 0, root_lvl.spec.size
        elif isinstance(root_lvl.spec, CompressedSpec):
            p_begin, p_end = 0, len(root_lvl.storage.crd)
        elif isinstance(root_lvl.spec, SingletonSpec):
            p_begin, p_end = 0, len(root_lvl.storage.crd)
        else:
            raise TypeError(
                f"Unsupported root level type: {type(root_lvl.spec).__name__}"
            )

        for p in range(p_begin, p_end):
            coord = int(root_lvl.iter_coord(p))
            vi = root_lvl.value_index(0, p)
            coords = (coord,)
            if len(self.lvls) == 1:
                sparse_map[coords] = sparse_map.get(coords, 0.0) + float(vals[vi])
            else:
                walk(1, p, coords)

        return sparse_map

    @staticmethod
    def _format_row(sparse_map: dict, prefix: tuple[int, ...], size: int) -> str:
        row_str = []
        for c in range(size):
            key = prefix + (c,)
            val = sparse_map.get(key, 0.0)
            row_str.append(f"{val:>4.2f}" if val != 0.0 else "   .")
        return " ".join(row_str)

    @staticmethod
    def _format_grid(sparse_map: dict, shape: tuple[int, ...]) -> str:
        if len(shape) == 1:
            return SparseTensor._format_row(sparse_map, (), shape[0])

        num_rows, num_cols = shape[-2], shape[-1]
        prefix_shape = shape[:-2]

        lines = []
        if prefix_shape:
            for prefix in product(*[range(s) for s in prefix_shape]):
                header = f"[{', '.join(map(str, prefix))}, :, :]:"
                lines.append(header)
                for r in range(num_rows):
                    row = SparseTensor._format_row(sparse_map, prefix + (r,), num_cols)
                    lines.append(row)
        else:
            for r in range(num_rows):
                row = SparseTensor._format_row(sparse_map, (r,), num_cols)
                lines.append(row)

        return "\n".join(lines)

    def __repr__(self) -> str:
        if len(self.shape) == 0:
            return f"SparseTensor(shape={self.shape})"

        sparse_map = self._collect_entries()
        return self._format_grid(sparse_map, self.shape)

    # ---------------------------------------------------------------------------
    # PyTree
    # ---------------------------------------------------------------------------

    def tree_flatten(self):
        children = (self.values, self.lvls)
        aux_data = (self.shape,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        values, lvls = children
        (shape,) = aux_data
        return cls(values, shape, lvls)

    # ---------------------------------------------------------------------------
    # Operations
    # ---------------------------------------------------------------------------

    def add(self, other: "SparseTensor") -> jax.Array:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        if len(self.lvls) != len(other.lvls):
            raise ValueError(
                f"Level count mismatch: {len(self.lvls)} vs {len(other.lvls)}"
            )

        for i, (la, lb) in enumerate(zip(self.lvls, other.lvls)):
            if type(la.spec) is not type(lb.spec):
                raise ValueError(
                    f"Level {i} format mismatch: {type(la.spec).__name__} vs {type(lb.spec).__name__}"
                )

        out = jnp.zeros(self.shape, dtype=self.values.dtype)
        out = self._scatter_tensor(out)
        out = other._scatter_tensor(out)
        return out

    def dot(self, x: jax.Array) -> jax.Array:
        if isinstance(x, SparseTensor):
            x_dense = jnp.zeros(x.shape, dtype=x.values.dtype)
            x_dense = x._scatter_tensor(x_dense)
            return self.dot(x_dense)

        if self.shape[1] != x.shape[0]:
            raise ValueError(f"Shape mismatch for dot: {self.shape} @ {x.shape}")

        result_shape: tuple[int, ...]
        if x.ndim == 1:
            result_shape = (self.shape[0],)
        elif x.ndim == 2:
            result_shape = (self.shape[0], x.shape[1])
        else:
            raise ValueError(f"dot only supports 1D or 2D x, got {x.ndim}D")

        result_dtype = jnp.result_type(self.values.dtype, x.dtype)
        out = jnp.zeros(result_shape, dtype=result_dtype)
        return self._gather_dot_tensor(x, out)

    # ---------------------------------------------------------------------------
    # Internal Operations
    # ---------------------------------------------------------------------------

    def _gather_dot_inner(self, lvl1, vals, parent_pos, coord0, x, out):
        p1_begin, p1_end = lvl1.iter_bounds(parent_pos)

        def inner_body(p1, out):
            coord1 = lvl1.iter_coord(p1)
            vi = lvl1.value_index(parent_pos, p1)
            return out.at[coord0].add(vals[vi] * x[coord1])

        return lax.fori_loop(p1_begin, p1_end, inner_body, out)

    def _gather_dot_tensor(self, x, out):
        lvls = self.lvls
        vals = self.values

        if len(lvls) != 2:
            raise NotImplementedError("dot currently supports only 2D tensors")

        if vals.size == 0:
            return out

        lvl0 = lvls[0]
        lvl1 = lvls[1]

        p0_begin, p0_end = lvl0.iter_bounds(0)

        def outer_body(p0, out):
            coord0 = lvl0.iter_coord(p0)
            return self._gather_dot_inner(lvl1, vals, p0, coord0, x, out)

        return lax.fori_loop(p0_begin, p0_end, outer_body, out)

    def _scatter_inner(self, lvl1, vals, parent_pos, coord0, out):
        p1_begin, p1_end = lvl1.iter_bounds(parent_pos)

        def inner_body(p1, out):
            coord1 = lvl1.iter_coord(p1)
            vi = lvl1.value_index(parent_pos, p1)
            return out.at[coord0, coord1].add(vals[vi])

        return lax.fori_loop(p1_begin, p1_end, inner_body, out)

    def _scatter_tensor(self, out):
        lvls = self.lvls
        vals = self.values

        if len(lvls) != 2:
            raise NotImplementedError("add currently supports only 2D tensors")

        if vals.size == 0:
            return out

        lvl0 = lvls[0]
        lvl1 = lvls[1]

        p0_begin, p0_end = lvl0.iter_bounds(0)

        def outer_body(p0, out):
            coord0 = lvl0.iter_coord(p0)
            return self._scatter_inner(lvl1, vals, p0, coord0, out)

        return lax.fori_loop(p0_begin, p0_end, outer_body, out)
