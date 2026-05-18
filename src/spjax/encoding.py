import jax
from enum import Enum
from abc import ABC, abstractmethod


class LevelType(ABC):
    @abstractmethod
    def pos_bounds(self, p_k_minus_1) -> tuple[int, int]: ...

    @abstractmethod
    def pos_access(self, p_k) -> tuple[int, bool]: ...


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

class SingletonLevel(LevelType):
    crd: jax.Array

    def __init__(self, crd) -> None:
        self.crd = crd

    def pos_bounds(self, p_k_minus_1) -> tuple[int, int]:
        return p_k_minus_1, p_k_minus_1 + 1

    def pos_access(self, p_k) -> tuple[int, bool]:
        i_k = int(self.crd[p_k])
        return i_k, True


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
