import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

class SparseTensor:
    nnz: int
    values: jax.Array
    pos: jax.Array
    crd: jax.Array

    def __init__(self, nnz, values, pos, crd) -> None:
        self.nnz = nnz
        self.values = jnp.asarray(values)
        self.pos = jnp.asarray(pos)
        self.crd = jnp.asarray(crd)

    @property
    def row(self) -> jax.Array:
        return self.crd[0]

    @property
    def col(self) -> jax.Array:
        return self.crd[1]

    def __add__(self, other) -> SparseTensor:
        if isinstance(other, SparseTensor):
            new_values = jnp.concatenate([self.values, other.values])
            new_crd = jnp.concatenate([self.crd, other.crd], axis=1)
            new_nnz = new_values.shape[0]
            new_pos = jnp.array([0, new_nnz])
            return SparseTensor(new_nnz, new_values, new_pos, new_crd)

        raise NotImplemented

    @classmethod
    def from_file(cls, filename) -> SparseTensor:
        try:
            import scipy
        except:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        m_coo = scipy.io.mmread(filename)
        nnz = m_coo.nnz
        pos = jnp.array([0, nnz])
        crd = jnp.stack([
            jnp.asarray(m_coo.row),
            jnp.asarray(m_coo.col),
        ])
        return cls(nnz, jnp.asarray(m_coo.data), pos, crd)

def sparse_tensor_flatten(obj):
    children = (obj.values, obj.pos, obj.crd)
    aux_data = (obj.nnz,)
    return (children, aux_data)

def sparse_tensor_unflatten(aux_data, children):
    values, pos, crd = children
    nnz, = aux_data
    return SparseTensor(nnz, values, pos, crd)

register_pytree_node(SparseTensor, sparse_tensor_flatten, sparse_tensor_unflatten)
