import jax
import jax.numpy as jnp

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

    def __add__(self, other) -> "SparseTensor":
        from .primitive import sparse_add

        return sparse_add(self, other)

    def __mul__(self, other) -> "SparseTensor":
        from .primitive import sparse_mul

        return sparse_mul(self, other)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def _from_coo(cls, m_coo) -> "SparseTensor":
        nnz = m_coo.nnz
        values = jnp.asarray(m_coo.data)
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
        return cls._from_coo(coo)

    def to_dense_str(self) -> str:
        """Pretty-print a 2D tensor as a dense grid (for debugging)."""
        if len(self.shape) != 2:
            raise ValueError("to_dense_str only supports 2D tensors")

        num_rows, num_cols = self.shape
        rows: list[int] = []
        cols: list[int] = []

        # Walk the level hierarchy to extract coordinates.
        # COO-like: level 0 = compressed (row), level 1 = singleton (col)
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
                row_str.append(f"{val:>6.2f}" if val != 0.0 else "     .")
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
