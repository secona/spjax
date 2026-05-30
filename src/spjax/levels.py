from dataclasses import dataclass
from enum import Enum, auto

import jax

from spjax.storage import CompressedStorage, DenseStorage, SingletonStorage


class IterationKind(Enum):
    POSITION = auto()
    COORDINATE = auto()
    LOCATE = auto()


@dataclass(frozen=True)
class LevelProperties:
    is_full: bool
    """every valid coordinate present (superset of any sparse dim)"""

    is_unique: bool
    """no duplicate coordinates under the same parent"""

    is_ordered: bool
    """coordinates in increasing order"""

    is_branchless: bool
    """each parent has exactly one child"""

    is_compact: bool
    """no unlabeled padding between coordinates"""


@dataclass(frozen=True)
class LevelSpec:
    name: str

    supported_iteration_kinds: frozenset[IterationKind]

    properties: LevelProperties

    supports_insert: bool = False
    """assembly: insert at arbitrary position"""

    supports_append: bool = False
    """assembly: append in sorted order"""

    def supports(self, kind: IterationKind) -> bool:
        return kind in self.supported_iteration_kinds

    @property
    def supports_coord_pos_iter(self) -> bool:
        return self.supports(IterationKind.POSITION)

    @property
    def supports_coord_value_iter(self) -> bool:
        return self.supports(IterationKind.COORDINATE)

    @property
    def supports_locate(self) -> bool:
        return self.supports(IterationKind.LOCATE)


@dataclass(frozen=True)
class DenseSpec(LevelSpec):
    size: int = 0

    def __init__(self, size: int):
        object.__setattr__(self, "size", size)

        super().__init__(
            name="dense",
            supported_iteration_kinds=frozenset(
                {
                    IterationKind.COORDINATE,
                    IterationKind.LOCATE,
                }
            ),
            properties=LevelProperties(
                is_full=True,
                is_unique=True,
                is_ordered=True,
                is_branchless=False,
                is_compact=True,
            ),
        )


@dataclass(frozen=True)
class CompressedSpec(LevelSpec):
    def __init__(self, is_unique=None):
        super().__init__(
            name="compressed",
            supported_iteration_kinds=frozenset(
                {
                    IterationKind.POSITION,
                }
            ),
            properties=LevelProperties(
                is_full=False,
                is_unique=True if is_unique is None else is_unique,
                is_ordered=True,
                is_branchless=False,
                is_compact=True,
            ),
            supports_append=True,
        )


@dataclass(frozen=True)
class SingletonSpec(LevelSpec):
    def __init__(self):
        super().__init__(
            name="singleton",
            supported_iteration_kinds=frozenset(
                {
                    IterationKind.POSITION,
                }
            ),
            properties=LevelProperties(
                is_full=False,
                is_unique=True,
                is_ordered=True,
                is_branchless=True,
                is_compact=True,
            ),
            supports_append=True,
        )


@jax.tree_util.register_pytree_node_class
class SparseLevel:
    spec: LevelSpec
    storage: DenseStorage | CompressedStorage | SingletonStorage

    def __init__(
        self,
        spec: LevelSpec,
        storage: DenseStorage | CompressedStorage | SingletonStorage,
    ):
        self.spec = spec
        self.storage = storage

    def iter_bounds(self, parent_pos):
        if isinstance(self.spec, DenseSpec):
            return 0, self.spec.size
        if isinstance(self.spec, CompressedSpec):
            return self.storage.pos[parent_pos], self.storage.pos[parent_pos + 1]
        if isinstance(self.spec, SingletonSpec):
            return parent_pos, parent_pos + 1
        raise TypeError(f"Unsupported spec: {type(self.spec).__name__}")

    def iter_coord(self, p):
        if isinstance(self.spec, DenseSpec):
            return p
        if isinstance(self.spec, (CompressedSpec, SingletonSpec)):
            return self.storage.crd[p]
        raise TypeError(f"Unsupported spec: {type(self.spec).__name__}")

    def value_index(self, parent_pos, p):
        if isinstance(self.spec, DenseSpec):
            return parent_pos * self.spec.size + p
        return p

    def tree_flatten(self):
        if isinstance(self.storage, DenseStorage):
            children = ()
            aux_data = (self.spec, self.storage.size)
        elif isinstance(self.storage, CompressedStorage):
            children = (self.storage.pos, self.storage.crd)
            aux_data = (self.spec,)
        elif isinstance(self.storage, SingletonStorage):
            children = (self.storage.crd,)
            aux_data = (self.spec,)
        else:
            raise TypeError(f"Unknown storage type: {type(self.storage)}")
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        spec = aux_data[0]
        if isinstance(spec, DenseSpec):
            size = aux_data[1]
            return cls(spec, DenseStorage(size))
        if isinstance(spec, CompressedSpec):
            pos, crd = children
            return cls(spec, CompressedStorage(pos, crd))
        if isinstance(spec, SingletonSpec):
            (crd,) = children
            return cls(spec, SingletonStorage(crd))
        raise TypeError(f"Unknown level spec: {type(spec)}")
