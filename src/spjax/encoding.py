import jax
from enum import Enum
from abc import ABC, abstractmethod


class LevelType(ABC):
    @abstractmethod
    def pos_bounds(self, p_k_minus_1) -> tuple[int, int]: ...

    @abstractmethod
    def pos_access(self, p_k) -> tuple[int, bool]: ...

    @abstractmethod
    def append_coord(self, p_k, i_k): ...

    @abstractmethod
    def append_edges(self, p_k_minus_1, pbegin_k, pend_k): ...

    @abstractmethod
    def append_init(self, sz_k_minus_1, sz_k): ...

    @abstractmethod
    def append_finalize(self, sz_k_minus_1, sz_k): ...


@jax.tree_util.register_pytree_node_class
class CompressedLevel(LevelType):
    pos: jax.Array
    crd: jax.Array

    def __init__(self, pos, crd) -> None:
        self.pos = pos
        self.crd = crd

    def pos_bounds(self, p_k_minus_1) -> tuple[int, int]:
        p_begin = int(self.pos[p_k_minus_1])
        p_end = int(self.pos[p_k_minus_1 + 1])
        return p_begin, p_end

    def pos_access(self, p_k) -> tuple[int, bool]:
        i_k = int(self.crd[p_k])
        return i_k, True

    def append_coord(self, p_k, i_k):
        self.crd = self.crd.at[p_k].set(i_k)

    def append_edges(self, p_k_minus_1, pbegin_k, pend_k):
        self.pos = self.pos.at[p_k_minus_1 + 1].set(pend_k - pbegin_k)

    def append_init(self, sz_k_minus_1, sz_k):
        del sz_k
        for p_k_minus_1 in range(sz_k_minus_1 + 1):
            self.pos = self.pos.at[p_k_minus_1].set(0)

    def append_finalize(self, sz_k_minus_1, sz_k):
        cumsum = self.pos[0]
        for p_k_minus_1 in range(1, sz_k_minus_1 + 1):
            cumsum += self.pos[p_k_minus_1]
            self.pos = self.pos.at[p_k_minus_1].set(cumsum)

    def tree_flatten(self):
        children = (self.pos, self.crd)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        pos, crd = children
        return cls(pos, crd)


@jax.tree_util.register_pytree_node_class
class SingletonLevel(LevelType):
    crd: jax.Array

    def __init__(self, crd) -> None:
        self.crd = crd

    def pos_bounds(self, p_k_minus_1) -> tuple[int, int]:
        return p_k_minus_1, p_k_minus_1 + 1

    def pos_access(self, p_k) -> tuple[int, bool]:
        i_k = int(self.crd[p_k])
        return i_k, True

    def tree_flatten(self):
        children = (self.crd,)
        aux_data = ()
        return children, aux_data

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

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        crd, = children
        return cls(crd)


class LevelFormat(str, Enum):
    DENSE = "dense"
    COMPRESSED = "compressed"
    SINGLETON = "singleton"
    BLOCKED = "blocked"


class Dimension:
    name: str

    def __init__(self, name):
        self.name = name


class SparseEncoding:
    lvl_format: tuple[LevelFormat, ...]
    dims: list[Dimension]
    lvls: dict[Dimension, LevelFormat]

    def __init__(self, dims: list[Dimension], lvls: dict[Dimension, LevelFormat]):
        self.dims = dims
        self.lvls = lvls
