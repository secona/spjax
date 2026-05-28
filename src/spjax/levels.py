from dataclasses import dataclass
from enum import Enum, auto

import jax

# ------------------------------------------------------------------------------
# Iteration Semantics
# ------------------------------------------------------------------------------


class IterationKind(Enum):
    POSITION = auto()
    COORDINATE = auto()
    LOCATE = auto()


class MergeBehavior(Enum):
    UNION = auto()
    INTERSECTION = auto()
    LOCATE = auto()
    DENSE = auto()


# ------------------------------------------------------------------------------
# Level Properties
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# Level Specification (Compiler Semantic Layer)
# ------------------------------------------------------------------------------


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


# ------------------------------------------------------------------------------
# Runtime Storage Layer
# ------------------------------------------------------------------------------


@dataclass
class LevelStorage:
    pass


@dataclass
class DenseStorage(LevelStorage):
    size: int


@dataclass
class CompressedStorage(LevelStorage):
    pos: object
    crd: object


@dataclass
class SingletonStorage(LevelStorage):
    crd: object


# ------------------------------------------------------------------------------
# Sparse Level
# ------------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
class SparseLevel:
    spec: LevelSpec
    storage: LevelStorage

    def __init__(self, spec: LevelSpec, storage: LevelStorage):
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

    # --------------------------------------------------------------------------
    # PyTree
    # --------------------------------------------------------------------------

    def tree_flatten(self):
        if isinstance(self.storage, DenseStorage):
            children = ()
            aux_data = (self.spec, "dense", self.storage.size)
        elif isinstance(self.storage, CompressedStorage):
            children = (self.storage.pos, self.storage.crd)
            aux_data = (self.spec, "compressed")
        elif isinstance(self.storage, SingletonStorage):
            children = (self.storage.crd,)
            aux_data = (self.spec, "singleton")
        else:
            raise TypeError(f"Unknown storage type: {type(self.storage)}")
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kind = aux_data[1]
        if kind == "dense":
            spec, _, size = aux_data
            return cls(spec, DenseStorage(size))
        if kind == "compressed":
            spec, _ = aux_data
            pos, crd = children
            return cls(spec, CompressedStorage(pos, crd))
        if kind == "singleton":
            spec, _ = aux_data
            (crd,) = children
            return cls(spec, SingletonStorage(crd))
        raise TypeError(f"Unknown storage kind: {kind}")


# ------------------------------------------------------------------------------
# Sparse Iterators
# ------------------------------------------------------------------------------


class SparseIterator:
    def valid(self) -> bool: ...

    def coord(self) -> int: ...

    def pos(self) -> int: ...

    def next(self) -> None: ...

    def seek(self, coord: int) -> None: ...


class DenseCoordinateIterator(SparseIterator):
    def __init__(self, size: int):
        self.i = 0
        self.size = size

    def valid(self) -> bool:
        return self.i < self.size

    def coord(self) -> int:
        return self.i

    def pos(self) -> int:
        return self.i

    def next(self) -> None:
        self.i += 1

    def seek(self, coord: int) -> None:
        self.i = coord


class CompressedIterator(SparseIterator):
    def __init__(self, crd, p_begin: int, p_end: int):
        self.crd_arr = crd
        self.p = p_begin
        self.p_end = p_end

    def valid(self) -> bool:
        return self.p < self.p_end

    def coord(self) -> int:
        return int(self.crd_arr[self.p])

    def pos(self) -> int:
        return self.p

    def next(self) -> None:
        self.p += 1

    def seek(self, coord: int) -> None:
        while self.valid() and self.coord() < coord:
            self.next()


class IteratorFactory:
    @staticmethod
    def make_iterator(
        spec: LevelSpec,
        storage,
        *,
        parent_pos: int | None = None,
    ) -> SparseIterator:
        if isinstance(spec, DenseSpec):
            return DenseCoordinateIterator(spec.size)

        if isinstance(spec, CompressedSpec):
            p_begin = storage.pos[parent_pos]
            p_end = storage.pos[parent_pos + 1]

            return CompressedIterator(
                storage.crd,
                int(p_begin),
                int(p_end),
            )

        if isinstance(spec, SingletonSpec):
            return CompressedIterator(
                storage.crd,
                parent_pos,
                parent_pos + 1,
            )

        raise TypeError(f"Unsupported level spec: {type(spec).__name__}")

    @staticmethod
    def make_root_iterator(level: SparseLevel) -> SparseIterator:
        if isinstance(level.spec, DenseSpec):
            return IteratorFactory.make_iterator(level.spec, level.storage)
        return IteratorFactory.make_iterator(level.spec, level.storage, parent_pos=0)


# ------------------------------------------------------------------------------
# Merge Planning Structures
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexVar:
    name: str


@dataclass
class IteratorRef:
    tensor: str
    iv: IndexVar

    level: SparseLevel

    iteration_kind: IterationKind


@dataclass
class LatticePoint:
    iterators: list[IteratorRef]

    behavior: MergeBehavior

    expr: object | None = None
