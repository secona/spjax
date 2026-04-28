from jax.tree_util import register_pytree_node

class SparseTensor:
    nnz: int

    def __init__(self, nnz) -> None:
        self.nnz = nnz

    @classmethod
    def from_file(cls, filename) -> SparseTensor:
        try:
            import scipy
        except:
            raise ImportError("Failed to import SciPy for reading sparse tensor")

        m_coo = scipy.io.mmread(filename)
        return SparseTensor(m_coo.nnz)

def sparse_tensor_flatten(obj):
    children = (obj.nnz,)
    aux_data = None
    return (children, aux_data)

def sparse_tensor_unflatten(_, children):
    return SparseTensor(*children)

register_pytree_node(SparseTensor, sparse_tensor_flatten, sparse_tensor_unflatten)
