from dataclasses import dataclass
from enum import Enum


class DimLevelType(str, Enum):
    DENSE = "dense"
    COMPRESSED = "compressed"
    SINGLETON = "singleton"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class SparseEncoding:
    lvl_types: tuple[DimLevelType, ...]
    lvl_order: tuple[int, ...]
    ordered: tuple[bool, ...]
    unique: tuple[bool, ...]
    pos_width: int = 32
    crd_width: int = 32

    def __post_init__(self):
        nlevels = len(self.lvl_types)
        if nlevels == 0:
            raise ValueError("lvl_types must be non-empty")
        if len(self.lvl_order) != nlevels:
            raise ValueError("lvl_order length must match number of levels")
        if len(self.ordered) != nlevels:
            raise ValueError("ordered length must match number of levels")
        if len(self.unique) != nlevels:
            raise ValueError("unique length must match number of levels")

        expected_levels = tuple(range(nlevels))
        if tuple(sorted(self.lvl_order)) != expected_levels:
            raise ValueError("lvl_order must be a permutation of level indices")

        if self.pos_width not in (8, 16, 32, 64):
            raise ValueError("pos_width must be one of 8, 16, 32, 64")
        if self.crd_width not in (8, 16, 32, 64):
            raise ValueError("crd_width must be one of 8, 16, 32, 64")

    @property
    def nlevels(self) -> int:
        return len(self.lvl_types)

    def is_level_ordered(self, level: int) -> bool:
        return self.ordered[level]

    def is_level_unique(self, level: int) -> bool:
        return self.unique[level]
