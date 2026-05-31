from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp


from spjax import legacy_ops
from spjax.levels import (
    CompressedSpec,
    DenseSpec,
    LevelSpec,
    SingletonSpec,
    SparseLevel,
)
from spjax.storage import (
    CompressedStorage,
    DenseStorage,
    SingletonStorage,
)


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    level_specs: tuple[LevelSpec, ...]

    def order(self) -> int:
        return len(self.shape)


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

    @property
    def tensor_type(self) -> TensorType:
        return TensorType(
            shape=self.shape,
            level_specs=tuple(lvl.spec for lvl in self.lvls),
        )

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

    @classmethod
    def from_1d(cls, arr, mode="sparse") -> "SparseTensor":
        if mode == "dense":
            shape = (len(arr),)
            values = jnp.asarray(arr)
            lvl = SparseLevel(
                DenseSpec(len(arr)),
                DenseStorage(len(arr)),
            )
            return cls(values, shape, [lvl])
        elif mode == "sparse":
            values, crd = arr
            nnz = len(values)
            shape = (int(crd.max()) + 1,)
            lvl = SparseLevel(
                CompressedSpec(),
                CompressedStorage(pos=jnp.array([0, nnz]), crd=crd),
            )
            return cls(values, shape, [lvl])
        else:
            raise ValueError(
                f"Unsupported mode: {mode!r}. Expected 'dense' or 'sparse'."
            )

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
        return legacy_ops.sparse_add(self, other)

    def dot(self, x: jax.Array) -> jax.Array:
        return legacy_ops.sparse_dot(self, x)
