import jax.numpy as jnp
from jax.tree_util import register_pytree_node

class SparseTensor:
    nnz: int

    def __init__(self, nnz, values, row, col) -> None:
        self.nnz = nnz
        self.values = jnp.asarray(values)
        self.row = jnp.asarray(row)
        self.col = jnp.asarray(col)

    def __add__(self, other) -> SparseTensor:
        if isinstance(other, SparseTensor):
            new_values = jnp.concatenate([self.values, other.values])
            new_row = jnp.concatenate([self.row, other.row])
            new_col = jnp.concatenate([self.col, other.col])
            nnz = len(new_values)

            return SparseTensor(nnz, new_values, new_row, new_col)

        raise NotImplemented

    @classmethod
    def from_file(cls, filename) -> SparseTensor:
        try:
            import scipy
        except:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        m_coo = scipy.io.mmread(filename)
        return SparseTensor(m_coo.nnz, m_coo.data, m_coo.row, m_coo.col)

def sparse_tensor_flatten(obj):
    children = (obj.values, obj.row, obj.col)
    aux_data = (obj.nnz,)
    return (children, aux_data)

def sparse_tensor_unflatten(aux_data, children):
    values, row, col = children
    nnz, = aux_data
    return SparseTensor(nnz, values, row, col)

register_pytree_node(SparseTensor, sparse_tensor_flatten, sparse_tensor_unflatten)
