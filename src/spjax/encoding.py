from enum import Enum, auto
from abc import ABC, abstractmethod

import jax


class LevelFormat(Enum):
    DENSE = auto()
    COMPRESSED = auto()
    SINGLETON = auto()


class LevelType(ABC):
    fmt: LevelFormat

    # --------------------------------------------------------------------------
    # Capabilities
    # --------------------------------------------------------------------------

    supports_coord_value_iter: bool = False
    """exposes coord_bounds / coord_access"""

    supports_coord_pos_iter: bool = False
    """exposes pos_bounds / pos_access"""

    supports_locate: bool = False
    """exposes locate() for O(1) random access"""

    supports_insert: bool = False
    """assembly: insert at arbitrary position"""

    supports_append: bool = False
    """assembly: append in sorted order"""

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    is_full: bool = False
    """every valid coordinate present (superset of any sparse dim)"""

    is_unique: bool = True
    """no duplicate coordinates under the same parent"""

    is_ordered: bool = True
    """coordinates in increasing order"""

    is_branchless: bool = False
    """each parent has exactly one child"""

    is_compact: bool = True
    """no unlabeled padding between coordinates"""

    # --------------------------------------------------------------------------
    # Functions
    # --------------------------------------------------------------------------

    @abstractmethod
    def pos_bounds(self, p_k_minus_1) -> tuple[jax.Array, jax.Array]: ...

    @abstractmethod
    def pos_access(self, p_k) -> tuple[jax.Array, bool]: ...

    @abstractmethod
    def coord_bounds(self, p_k_minus_1) -> tuple[int, int]: ...

    @abstractmethod
    def coord_access(self, p_k_minus_1, i_k) -> tuple[int, bool]: ...

    @abstractmethod
    def locate(self, p_k_minus_1, i_k) -> tuple[int, bool]: ...

    @abstractmethod
    def append_coord(self, p_k, i_k): ...

    @abstractmethod
    def append_edges(self, p_k_minus_1, pbegin_k, pend_k): ...

    @abstractmethod
    def append_init(self, sz_k_minus_1, sz_k): ...

    @abstractmethod
    def append_finalize(self, sz_k_minus_1, sz_k): ...

    @abstractmethod
    def insert_coord(self, p_k, i_k): ...

    @abstractmethod
    def insert_init(self, sz_k_minus_1, sz_k): ...

    @abstractmethod
    def insert_finalize(self, sz_k_minus_1, sz_k): ...


@jax.tree_util.register_pytree_node_class
class DenseLevel(LevelType):
    size: int

    def __init__(self, size: int) -> None:
        self.size = size
        self.fmt = LevelFormat.DENSE

        self.supports_coord_value_iter = True
        self.supports_locate = True

        self.is_full = True
        self.is_unique = True
        self.is_ordered = True
        self.is_branchless = False
        self.is_compact = True

    def pos_bounds(self, p_k_minus_1) -> tuple[jax.Array, jax.Array]:
        del p_k_minus_1
        raise NotImplementedError("Dense level does not support pos_bounds")

    def pos_access(self, p_k) -> tuple[jax.Array, bool]:
        del p_k
        raise NotImplementedError("Dense level does not support pos_access")

    def coord_bounds(self, p_k_minus_1) -> tuple[int, int]:
        del p_k_minus_1
        return 0, self.size

    def coord_access(self, p_k_minus_1, i_k) -> tuple[jax.Array, jax.Array]:
        del p_k_minus_1
        return i_k, (0 <= i_k) & (i_k < self.size)

    def locate(self, p_k_minus_1, i_k) -> tuple[int, bool]:
        del p_k_minus_1
        if 0 <= i_k < self.size:
            return i_k, True
        return 0, False

    def append_coord(self, p_k, i_k):
        del p_k, i_k
        raise NotImplementedError("Dense level does not support append")

    def append_edges(self, p_k_minus_1, pbegin_k, pend_k):
        del p_k_minus_1, pbegin_k, pend_k
        raise NotImplementedError("Dense level does not support append")

    def append_init(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        raise NotImplementedError("Dense level does not support append")

    def append_finalize(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        raise NotImplementedError("Dense level does not support append")

    def insert_coord(self, p_k, i_k):
        del p_k, i_k
        raise NotImplementedError("Dense level does not support insert")

    def insert_init(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        raise NotImplementedError("Dense level does not support insert")

    def insert_finalize(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        raise NotImplementedError("Dense level does not support insert")

    # --------------------------------------------------------------------------
    # PyTree
    # --------------------------------------------------------------------------

    def tree_flatten(self):
        children = ()
        aux_data = (self.size,)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        (size,) = aux_data
        return cls(size)


@jax.tree_util.register_pytree_node_class
class CompressedLevel(LevelType):
    pos: jax.Array
    crd: jax.Array

    def __init__(self, pos: jax.Array, crd: jax.Array) -> None:
        self.pos = pos
        self.crd = crd
        self.fmt = LevelFormat.COMPRESSED

        self.supports_coord_pos_iter = True
        self.supports_append = True

        self.is_full = False
        self.is_unique = True
        self.is_ordered = True
        self.is_branchless = False
        self.is_compact = True

    def pos_bounds(self, p_k_minus_1) -> tuple[jax.Array, jax.Array]:
        p_begin = self.pos[p_k_minus_1]
        p_end = self.pos[p_k_minus_1 + 1]
        return p_begin, p_end

    def pos_access(self, p_k) -> tuple[jax.Array, bool]:
        i_k = self.crd[p_k]
        return i_k, True

    def coord_bounds(self, p_k_minus_1) -> tuple[int, int]:
        del p_k_minus_1
        raise NotImplementedError("Compressed level does not support coord_bounds")

    def coord_access(self, p_k_minus_1, i_k) -> tuple[int, bool]:
        del p_k_minus_1, i_k
        raise NotImplementedError("Compressed level does not support coord_access")

    def locate(self, p_k_minus_1, i_k) -> tuple[int, bool]:
        del p_k_minus_1, i_k
        raise NotImplementedError(
            "Compressed level does not support locate; use a hash map or sorted search"
        )

    def append_coord(self, p_k, i_k):
        self.crd = self.crd.at[p_k].set(i_k)

    def append_edges(self, p_k_minus_1, pbegin_k, pend_k):
        self.pos = self.pos.at[p_k_minus_1 + 1].set(pend_k - pbegin_k)

    def append_init(self, sz_k_minus_1, sz_k):
        del sz_k
        for p_k_minus_1 in range(sz_k_minus_1 + 1):
            self.pos = self.pos.at[p_k_minus_1].set(0)

    def append_finalize(self, sz_k_minus_1, sz_k):
        del sz_k
        cumsum = self.pos[0]
        for p_k_minus_1 in range(1, sz_k_minus_1 + 1):
            cumsum += self.pos[p_k_minus_1]
            self.pos = self.pos.at[p_k_minus_1].set(cumsum)

    def insert_coord(self, p_k, i_k):
        del p_k, i_k
        raise NotImplementedError("Compressed level does not support insert")

    def insert_init(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        raise NotImplementedError("Compressed level does not support insert")

    def insert_finalize(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        raise NotImplementedError("Compressed level does not support insert")

    # --------------------------------------------------------------------------
    # PyTree
    # --------------------------------------------------------------------------

    def tree_flatten(self):
        children = (self.pos, self.crd)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        pos, crd = children
        return cls(pos, crd)


@jax.tree_util.register_pytree_node_class
class SingletonLevel(LevelType):
    crd: jax.Array

    def __init__(self, crd: jax.Array) -> None:
        self.crd = crd
        self.fmt = LevelFormat.SINGLETON

        self.supports_coord_pos_iter = True
        self.supports_append = True

        self.is_full = False
        self.is_unique = True
        self.is_ordered = True
        self.is_branchless = True
        self.is_compact = True

    def pos_bounds(self, p_k_minus_1) -> tuple[jax.Array, jax.Array]:
        return p_k_minus_1, p_k_minus_1 + 1

    def pos_access(self, p_k) -> tuple[jax.Array, bool]:
        i_k = self.crd[p_k]
        return i_k, True

    def coord_bounds(self, p_k_minus_1) -> tuple[int, int]:
        del p_k_minus_1
        raise NotImplementedError("Singleton level does not support coord_bounds")

    def coord_access(self, p_k_minus_1, i_k) -> tuple[int, bool]:
        del p_k_minus_1, i_k
        raise NotImplementedError("Singleton level does not support coord_access")

    def locate(self, p_k_minus_1, i_k) -> tuple[int, bool]:
        del p_k_minus_1, i_k
        raise NotImplementedError("Singleton level does not support locate")

    def append_coord(self, p_k, i_k):
        self.crd = self.crd.at[p_k].set(i_k)

    def append_edges(self, p_k_minus_1, pbegin_k, pend_k):
        del p_k_minus_1, pbegin_k, pend_k
        return

    def append_init(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        return

    def append_finalize(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        return

    def insert_coord(self, p_k, i_k):
        del p_k, i_k
        raise NotImplementedError("Singleton level does not support insert")

    def insert_init(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        raise NotImplementedError("Singleton level does not support insert")

    def insert_finalize(self, sz_k_minus_1, sz_k):
        del sz_k_minus_1, sz_k
        raise NotImplementedError("Singleton level does not support insert")

    # --------------------------------------------------------------------------
    # PyTree
    # --------------------------------------------------------------------------

    def tree_flatten(self):
        children = (self.crd,)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (crd,) = children
        return cls(crd)
