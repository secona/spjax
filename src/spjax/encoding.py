from dataclasses import dataclass
from enum import Enum


class LevelFormat(str, Enum):
    DENSE = "dense"
    COMPRESSED = "compressed"
    SINGLETON = "singleton"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class SparseEncoding:
    lvl_format: tuple[LevelFormat, ...]
