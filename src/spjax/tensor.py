import jax
import jax.numpy as jnp
from jax import lax

from spjax import encoding


@jax.tree_util.register_pytree_node_class
class SparseTensor:
    values: jax.Array
    shape: tuple[int, ...]
    lvls: list[encoding.LevelType]

    def __init__(
        self,
        values: jax.Array,
        shape: tuple[int, ...],
        lvls: list[encoding.LevelType],
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

        compressed_lvl = encoding.CompressedLevel(
            pos=jnp.array([0, nnz]),
            crd=jnp.asarray(m_coo.row),
        )
        compressed_lvl.is_unique = False  # COO rows may repeat

        singleton_lvl = encoding.SingletonLevel(crd=jnp.asarray(m_coo.col))

        lvls = [compressed_lvl, singleton_lvl]
        return cls(values, shape, lvls)

    @classmethod
    def _from_csr(cls, m_csr, shape=None) -> "SparseTensor":
        indptr = jnp.asarray(m_csr.indptr)
        indices = jnp.asarray(m_csr.indices)
        values = jnp.asarray(m_csr.data)
        if shape is None:
            shape = (m_csr.shape[0], m_csr.shape[1])

        dense_lvl = encoding.DenseLevel(shape[0])
        compressed_lvl = encoding.CompressedLevel(pos=indptr, crd=indices)

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

    def __repr__(self) -> str:
        """Pretty-print a 2D tensor as a dense grid (for debugging)."""
        if len(self.shape) != 2:
            raise ValueError("to_dense_str only supports 2D tensors")

        num_rows, num_cols = self.shape
        rows: list[int] = []
        cols: list[int] = []

        lvl0 = self.lvls[0]
        lvl1 = self.lvls[1]

        vals = self.values.tolist()
        sparse_map: dict[tuple[int, int], float] = {}

        if isinstance(lvl0, encoding.DenseLevel):
            for i in range(num_rows):
                p_begin = int(lvl1.pos[i])
                p_end = int(lvl1.pos[i + 1])
                for p in range(p_begin, p_end):
                    sparse_map[(i, int(lvl1.crd[p]))] = (
                        sparse_map.get((i, int(lvl1.crd[p])), 0.0) + vals[p]
                    )
        elif hasattr(lvl0, "crd"):
            for p in range(len(lvl0.crd)):
                rows.append(int(lvl0.crd[p]))
                cols.append(int(lvl1.crd[p]))
            for r, c, v in zip(rows, cols, vals):
                sparse_map[(r, c)] = sparse_map.get((r, c), 0.0) + v
        else:
            raise ValueError("Unsupported level 0 format")

        lines = []
        for r in range(num_rows):
            row_str = []
            for c in range(num_cols):
                val = float(sparse_map.get((r, c), 0.0))
                row_str.append(f"{val:>4.2f}" if val != 0.0 else "   .")
            lines.append(" ".join(row_str))
        return "\n".join(lines)

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
            if type(la) != type(lb):
                raise ValueError(
                    f"Level {i} format mismatch: {type(la).__name__} vs {type(lb).__name__}"
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
                inner_size = i1_end - i1_begin
                return out.at[coord0].add(
                    vals[coord0 * inner_size + coord1] * x[coord1]
                )

            return lax.fori_loop(i1_begin, i1_end, inner_body, out)

        else:
            raise ValueError(
                f"Level 1 format {type(lvl1).__name__} does not support iteration"
            )

    def _gather_dot_tensor(self, x, out):
        lvls = self.lvls
        vals = self.values

        if len(lvls) != 2:
            raise NotImplementedError("dot currently supports only 2D tensors")

        if vals.size == 0:
            return out

        lvl0 = lvls[0]
        lvl1 = lvls[1]

        if lvl0.supports_coord_pos_iter:
            p0_begin, p0_end = lvl0.pos_bounds(0)

            def outer_body(p0, out):
                coord0, _ = lvl0.pos_access(p0)
                return self._gather_dot_inner(lvl1, vals, p0, coord0, x, out)

            return lax.fori_loop(p0_begin, p0_end, outer_body, out)

        elif lvl0.supports_coord_value_iter:
            i0_begin, i0_end = lvl0.coord_bounds(0)

            def outer_body(i0, out):
                coord0, _ = lvl0.coord_access(0, i0)
                return self._gather_dot_inner(lvl1, vals, coord0, coord0, x, out)

            return lax.fori_loop(i0_begin, i0_end, outer_body, out)

        else:
            raise ValueError(
                f"Level 0 format {type(lvl0).__name__} does not support iteration"
            )

    def _scatter_inner(self, lvl1, vals, parent_pos, coord0, out):
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
                inner_size = i1_end - i1_begin
                return out.at[coord0, coord1].add(vals[coord0 * inner_size + coord1])

            return lax.fori_loop(i1_begin, i1_end, inner_body, out)

        else:
            raise ValueError(
                f"Level 1 format {type(lvl1).__name__} does not support iteration"
            )

    def _scatter_tensor(self, out):
        lvls = self.lvls
        vals = self.values

        if len(lvls) != 2:
            raise NotImplementedError("add currently supports only 2D tensors")

        if vals.size == 0:
            return out

        lvl0 = lvls[0]
        lvl1 = lvls[1]

        if lvl0.supports_coord_pos_iter:
            p0_begin, p0_end = lvl0.pos_bounds(0)

            def outer_body(p0, out):
                coord0, _ = lvl0.pos_access(p0)
                return self._scatter_inner(lvl1, vals, p0, coord0, out)

            return lax.fori_loop(p0_begin, p0_end, outer_body, out)

        elif lvl0.supports_coord_value_iter:
            i0_begin, i0_end = lvl0.coord_bounds(0)

            def outer_body(i0, out):
                coord0, _ = lvl0.coord_access(0, i0)
                return self._scatter_inner(lvl1, vals, coord0, coord0, out)

            return lax.fori_loop(i0_begin, i0_end, outer_body, out)

        else:
            raise ValueError(
                f"Level 0 format {type(lvl0).__name__} does not support iteration"
            )
