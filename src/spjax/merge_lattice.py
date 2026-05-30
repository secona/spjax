from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

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
        lvl = ", ".join(spec.name for spec in self.tensor_type.level_specs)
        return f"{self.name}<{lvl}>({ivs})"


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
        return f"Expr({self.left} + {self.right})"


@dataclass(frozen=True)
class MulExpr(Expr):
    left: Expr
    right: Expr

    def __repr__(self) -> str:
        return f"Expr({self.left} * {self.right})"


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
        return "IterationGraph(" + " -> ".join(iv.iv.name for iv in self.vars) + ")"


# ------------------------------------------------------------------------------
# Iterator References
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class IteratorRef:
    tensor: str

    iv: IndexVar

    level: LevelSpec

    def is_full(self) -> bool:
        return self.level.properties.is_full

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
    expr: Optional[Expr]
    children: List["LatticePoint"] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return len(self.iterators) == 0 and self.expr is None

    def __repr__(self, level=1) -> str:
        iters = ", ".join(repr(it) for it in self.iterators)
        indent = "  " * level
        if self.is_terminal():
            return f"{indent}LatticePoint(∅)"

        rep = f"{indent}LatticePoint(iters=[{iters}], expr={self.expr})"
        for child in self.children:
            rep += f"\n{child.__repr__(level + 1)}"
        return rep


# ------------------------------------------------------------------------------
# Merge Lattice
# ------------------------------------------------------------------------------


class MergeLattice:
    def __init__(self, expr: Expr, iv: IndexVar) -> None:
        self.expr = expr
        self.iv = iv
        self.root = self._build_recursive(self.expr)

    def _build_recursive(self, expr: Expr) -> LatticePoint:
        if isinstance(expr, AccessExpr):
            iterator = self._make_iterator(expr.access)

            if iterator is None:
                node = LatticePoint(
                    iterators=(),
                    expr=expr,
                )
            else:
                node = LatticePoint(
                    iterators=(iterator,),
                    expr=expr,
                )

            terminal_node = LatticePoint(iterators=(), expr=None)
            node.children.append(terminal_node)

            return node

        if isinstance(expr, AddExpr):
            left_top = self._build_recursive(expr.left)
            right_top = self._build_recursive(expr.right)
            return self._union_lattices(left_top, right_top, expr)

        if isinstance(expr, MulExpr):
            left_top = self._build_recursive(expr.left)
            right_top = self._build_recursive(expr.right)
            return self._intersect_lattices(left_top, right_top, expr)

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _union_lattices(
        self,
        left_top: LatticePoint,
        right_top: LatticePoint,
        expr: Expr,
    ) -> LatticePoint:
        merged_iters = self._merge_iters(left_top.iterators, right_top.iterators)
        top_node = LatticePoint(iterators=merged_iters, expr=expr)

        full_iters = {it for it in merged_iters if it.is_full()}

        if full_iters.issubset(set(left_top.iterators)):
            top_node.children.append(left_top)

        if full_iters.issubset(set(right_top.iterators)):
            top_node.children.append(right_top)

        if not top_node.children:
            top_node.children.append(LatticePoint(iterators=(), expr=None))

        return top_node

    def _intersect_lattices(
        self,
        left_top: LatticePoint,
        right_top: LatticePoint,
        expr: Expr,
    ) -> LatticePoint:
        merged_iters = self._merge_iters(left_top.iterators, right_top.iterators)
        top_node = LatticePoint(iterators=merged_iters, expr=expr)

        terminal_node = LatticePoint(iterators=(), expr=None)
        top_node.children.append(terminal_node)

        return top_node

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
    ) -> Optional[IteratorRef]:
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

        return None

    def __repr__(self) -> str:
        return f"MergeLattice(iv={self.iv}): \n{self.root}"
