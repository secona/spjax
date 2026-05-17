import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node


class SparseTensor:
    nnz: int
    values: jax.Array
    pos: jax.Array
    crd: jax.Array
    shape: tuple[int, ...]

    def __init__(self, nnz, values, pos, crd, shape) -> None:
        self.nnz = nnz
        self.values = values
        self.pos = pos
        self.crd = crd
        self.shape = shape

    @property
    def row(self) -> jax.Array:
        return self.crd[0]

    @property
    def col(self) -> jax.Array:
        return self.crd[1]

    def __add__(self, other) -> "SparseTensor":
        from .primitive import sparse_add

        return sparse_add(self, other)

    @classmethod
    def from_file(cls, filename) -> "SparseTensor":
        try:
            import scipy
        except:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        m_coo = scipy.io.mmread(filename)
        nnz = m_coo.nnz
        pos = jnp.array([0, nnz])
        crd = jnp.stack(
            [
                jnp.asarray(m_coo.row),
                jnp.asarray(m_coo.col),
            ]
        )
        shape = (int(m_coo.row.max()) + 1, int(m_coo.col.max()) + 1)
        return cls(nnz, jnp.asarray(m_coo.data), pos, crd, shape)

    @classmethod
    def from_dense(cls, arr) -> "SparseTensor":
        try:
            import scipy
        except:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        coo = scipy.sparse.coo_matrix(arr)
        nnz = coo.nnz
        pos = jnp.array([0, nnz])
        crd = jnp.stack([jnp.asarray(coo.row), jnp.asarray(coo.col)])
        shape = arr.shape
        return SparseTensor(
            nnz, jnp.asarray(coo.data, dtype=arr.dtype), pos, crd, shape
        )

    def to_dense_str(self) -> str:
        if self.crd.shape[0] != 2:
            raise ValueError("to_dense_str only supports 2D tensors")

        rows = self.row.tolist()
        cols = self.col.tolist()
        vals = self.values.tolist()

        num_rows = self.shape[0]
        num_cols = self.shape[1]

        sparse_map = {}
        for r, c, v in zip(rows, cols, vals):
            coords = (int(r), int(c))
            sparse_map[coords] = sparse_map.get(coords, 0) + v

        lines = []
        for r in range(num_rows):
            row_str = []
            for c in range(num_cols):
                val = int(sparse_map.get((r, c), 0))
                row_str.append(f"{val:>4}" if val != 0 else "   .")
            lines.append(" ".join(row_str))
        return "\n".join(lines)


def sparse_tensor_flatten(obj):
    children = (obj.values, obj.pos, obj.crd)
    aux_data = (obj.nnz, obj.shape)
    return (children, aux_data)


def sparse_tensor_unflatten(aux_data, children):
    values, pos, crd = children
    (nnz, shape) = aux_data
    return SparseTensor(nnz, values, pos, crd, shape)


register_pytree_node(SparseTensor, sparse_tensor_flatten, sparse_tensor_unflatten)
