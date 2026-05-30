from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto

from spjax.levels import LevelSpec
from spjax.tensor import TensorType

# ------------------------------------------------------------------------------
# Index Variables
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexVar:
    name: str

    def __repr__(self) -> str:
        return self.name


# ------------------------------------------------------------------------------
# Logical Tensor Dimension
# ------------------------------------------------------------------------------


@dataclass
class TensorDimension:
    iv: IndexVar

    def __repr__(self) -> str:
        return self.iv.name


# ------------------------------------------------------------------------------
# Tensor Access
# ------------------------------------------------------------------------------


@dataclass
class TensorAccess:
    name: str
    tensor_type: TensorType
    ivs: tuple[IndexVar, ...]

    def __repr__(self) -> str:
        ivs = ", ".join(iv.name for iv in self.ivs)
        return f"{self.name}({ivs})"


# ------------------------------------------------------------------------------
# Expression IR
# ------------------------------------------------------------------------------


class Expr(ABC):
    pass


@dataclass(frozen=True)
class AccessExpr(Expr):
    access: TensorAccess

    def __repr__(self) -> str:
        return repr(self.access)


@dataclass(frozen=True)
class AddExpr(Expr):
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass(frozen=True)
class MulExpr(Expr):
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"({self.left} * {self.right})"


# ------------------------------------------------------------------------------
# Iteration Graph
# ------------------------------------------------------------------------------


class IterationVarKind(Enum):
    SPATIAL = auto()
    REDUCTION = auto()


@dataclass(frozen=True)
class IterationVar:
    iv: IndexVar
    kind: IterationVarKind


@dataclass(frozen=True)
class IterationGraph:
    vars: tuple[IterationVar, ...]

    def __repr__(self) -> str:
        return " -> ".join(iv.iv.name for iv in self.vars)


# ------------------------------------------------------------------------------
# Iterator References
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class IteratorRef:
    tensor: str

    iv: IndexVar

    level: LevelSpec

    def __repr__(self) -> str:
        return f"{self.tensor}[{self.iv}]<{self.level.name}>"


# ------------------------------------------------------------------------------
# Merge Semantics
# ------------------------------------------------------------------------------


class MergeKind(Enum):
    UNION = auto()
    INTERSECTION = auto()
    LOCATE = auto()


@dataclass(frozen=True)
class LatticePoint:
    iterators: tuple[IteratorRef, ...]

    merge_kind: MergeKind

    expr: Expr

    def __repr__(self) -> str:
        iters = ", ".join(repr(it) for it in self.iterators)

        return (
            f"LatticePoint("
            f"merge={self.merge_kind.name}, "
            f"iters=[{iters}], "
            f"expr={self.expr}"
            f")"
        )


# ------------------------------------------------------------------------------
# Merge Lattice
# ------------------------------------------------------------------------------


class MergeLattice:
    def __init__(self, expr: Expr, iv: IndexVar) -> None:
        self.expr = expr
        self.iv = iv
        self.points: list[LatticePoint] = []

        self._build()

    def _build(self):
        self.points = self._build_recursive(self.expr)

    def _build_recursive(self, expr: Expr) -> list[LatticePoint]:
        if isinstance(expr, AccessExpr):
            iterator = self._make_iterator(expr.access)
            return [
                LatticePoint(
                    iterators=(iterator,),
                    merge_kind=MergeKind.UNION,
                    expr=expr,
                )
            ]

        if isinstance(expr, AddExpr):
            left_points = self._build_recursive(expr.left)
            right_points = self._build_recursive(expr.right)
            return self._union_lattices(left_points, right_points, expr)

        if isinstance(expr, MulExpr):
            left_points = self._build_recursive(expr.left)
            right_points = self._build_recursive(expr.right)
            return self._intersect_lattices(left_points, right_points, expr)

        return []

    def _union_lattices(
        self,
        left: list[LatticePoint],
        right: list[LatticePoint],
        expr: Expr,
    ) -> list[LatticePoint]:
        result: list[LatticePoint] = []

        for p1 in left:
            for p2 in right:
                result.append(
                    LatticePoint(
                        iterators=self._merge_iters(p1.iterators, p2.iterators),
                        merge_kind=MergeKind.UNION,
                        expr=expr,
                    )
                )

        for p1 in left:
            if not self._is_subset_of_any(p1.iterators, right):
                result.append(p1)

        for p2 in right:
            if not self._is_subset_of_any(p2.iterators, left):
                result.append(p2)

        return result

    def _intersect_lattices(
        self,
        left: list[LatticePoint],
        right: list[LatticePoint],
        expr: Expr,
    ) -> list[LatticePoint]:
        result: list[LatticePoint] = []

        for p1 in left:
            for p2 in right:
                result.append(
                    LatticePoint(
                        iterators=self._merge_iters(p1.iterators, p2.iterators),
                        merge_kind=MergeKind.INTERSECTION,
                        expr=expr,
                    )
                )

        return result

    def _merge_iters(
        self,
        a: tuple[IteratorRef, ...],
        b: tuple[IteratorRef, ...],
    ) -> tuple[IteratorRef, ...]:
        seen: set[tuple[str, str, str]] = set()
        merged: list[IteratorRef] = []
        for it in a + b:
            key = (it.tensor, it.iv.name, it.level.name)
            if key not in seen:
                seen.add(key)
                merged.append(it)
        return tuple(merged)

    def _is_subset_of_any(
        self,
        iters: tuple[IteratorRef, ...],
        points: list[LatticePoint],
    ) -> bool:
        iters_set = set((it.tensor, it.iv.name, it.level.name) for it in iters)
        for p in points:
            p_set = set((it.tensor, it.iv.name, it.level.name) for it in p.iterators)
            if iters_set.issubset(p_set):
                return True
        return False

    def _make_iterator(
        self,
        access: TensorAccess,
    ) -> IteratorRef:
        for iv, level in zip(
            access.ivs,
            access.tensor_type.level_specs,
        ):
            if iv == self.iv:
                return IteratorRef(
                    tensor=access.name,
                    iv=iv,
                    level=level,
                )

        raise RuntimeError(f"Tensor {access.name} does not participate in {self.iv}")

    def __repr__(self) -> str:
        body = "\n".join(f"  {point}" for point in self.points)

        return f"MergeLattice(iv={self.iv}): \n{body}"
