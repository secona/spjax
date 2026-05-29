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
        return (
            f"{self.tensor}[{self.iv}]"
            f"<{self.level.name}>"
        )

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
        expr = self.expr

        if isinstance(expr, AccessExpr):
            iterator = self._make_iterator(expr.access)

            self.points.append(
                LatticePoint(
                    iterators=(iterator,),
                    merge_kind=MergeKind.UNION,
                    expr=expr,
                )
            )

            return

        if isinstance(expr, AddExpr):
            left_iters = self._collect_iterators(expr.left)
            right_iters = self._collect_iterators(expr.right)

            self.points.append(
                LatticePoint(
                    iterators=tuple(left_iters + right_iters),
                    merge_kind=MergeKind.UNION,
                    expr=expr,
                )
            )

            return

        if isinstance(expr, MulExpr):
            left_iters = self._collect_iterators(expr.left)
            right_iters = self._collect_iterators(expr.right)

            self.points.append(
                LatticePoint(
                    iterators=tuple(left_iters + right_iters),
                    merge_kind=MergeKind.INTERSECTION,
                    expr=expr,
                )
            )

            return

    def _collect_iterators(self, expr: Expr) -> list[IteratorRef]:
        if isinstance(expr, AccessExpr):
            return [self._make_iterator(expr.access)]

        if isinstance(expr, AddExpr):
            return (
                self._collect_iterators(expr.left)
                + self._collect_iterators(expr.right)
            )

        if isinstance(expr, MulExpr):
            return (
                self._collect_iterators(expr.left)
                + self._collect_iterators(expr.right)
            )

        return []


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

        raise RuntimeError(
            f"Tensor {access.name} "
            f"does not participate in {self.iv}"
        )

    def __repr__(self) -> str:
        body = "\n".join(
            f"  {point}"
            for point in self.points
        )

        return (
            f"MergeLattice(iv={self.iv}) {{\n"
            f"{body}\n"
            f"}}"
        )
