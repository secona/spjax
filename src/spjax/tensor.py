import jax
import jax.numpy as jnp

from spjax import encoding


class TensorSpec:
    """Static metadata describing a sparse tensor's shape and level formats.

    Used at trace time by the JIT compiler to build merge lattices and
    emit the correct iteration loops.
    """

    def __init__(self, shape: tuple[int, ...], lvls: list[encoding.LevelType]) -> None:
        self.shape = shape
        self.lvls = lvls

    def __repr__(self) -> str:
        fmts = [lvl.fmt.name for lvl in self.lvls]
        return f"TensorSpec(shape={self.shape}, lvls={fmts})"

    def level_names(self) -> tuple[str, ...]:
        return tuple(lvl.fmt.name for lvl in self.lvls)


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
    # Flatten / unflatten for JAX primitive binding
    # ------------------------------------------------------------------

    def spec(self) -> TensorSpec:
        return TensorSpec(self.shape, self.lvls)

    def to_flat_arrays(self):
        """Return (values, pos, crd, spec) where pos/crd are concrete arrays."""
        # Flatten level arrays into pos and crd buffers.
        # For now, handle 2D tensors: level 0 (row), level 1 (col).
        pos_list = []
        crd_list = []
        for lvl in self.lvls:
            if hasattr(lvl, "pos") and lvl.pos is not None:
                pos_list.append(lvl.pos)
            if hasattr(lvl, "crd") and lvl.crd is not None:
                crd_list.append(lvl.crd)
        pos = jnp.concatenate(pos_list) if pos_list else jnp.array([0])
        crd = jnp.concatenate(crd_list) if crd_list else jnp.array([])
        return self.values, pos, crd, self.spec()

    @staticmethod
    def from_flat_arrays(values, pos, crd, spec):
        """Reconstruct from flat arrays and spec (not yet fully implemented)."""
        # For now, just return a stub; this is only needed for sparse output.
        return SparseTensor(values, spec.shape, spec.lvls)

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
    def from_file(cls, filename) -> "SparseTensor":
        try:
            import scipy
        except ImportError:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        m_coo = scipy.io.mmread(filename)
        return cls._from_coo(m_coo)

    @classmethod
    def from_dense(cls, arr) -> "SparseTensor":
        try:
            import scipy
        except ImportError:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        coo = scipy.sparse.coo_matrix(arr)
        return cls._from_coo(coo, shape=arr.shape)

    def to_dense_str(self) -> str:
        """Pretty-print a 2D tensor as a dense grid (for debugging)."""
        if len(self.shape) != 2:
            raise ValueError("to_dense_str only supports 2D tensors")

        num_rows, num_cols = self.shape
        rows: list[int] = []
        cols: list[int] = []

        lvl0 = self.lvls[0]
        lvl1 = self.lvls[1]

        if not hasattr(lvl0, "crd") or not hasattr(lvl1, "crd"):
            raise ValueError("to_dense_str requires level types with 'crd' attributes")

        for p in range(len(lvl0.crd)):
            rows.append(int(lvl0.crd[p]))
            cols.append(int(lvl1.crd[p]))

        vals = self.values.tolist()
        sparse_map: dict[tuple[int, int], float] = {}
        for r, c, v in zip(rows, cols, vals):
            coords = (int(r), int(c))
            sparse_map[coords] = sparse_map.get(coords, 0.0) + v

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
