from dataclasses import dataclass
from enum import Enum


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
