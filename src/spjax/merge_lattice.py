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
    """A symbolic representation of a loop index (e.g., 'i', 'j', 'k')"""

    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TensorDimension:
    """One index of a tensor"""

    name: str
    iv: IndexVar

    def __repr__(self) -> str:
        return f"{self.name}[{self.iv}]"


# ------------------------------------------------------------------------------
# Tensor Access
# ------------------------------------------------------------------------------


@dataclass
class TensorAccess:
    """Represents a tensor being accessed with a specific set of index variables"""

    name: str
    """The identifier of the tensor"""

    tensor_type: TensorType
    """The storage format and type information"""

    ivs: tuple[IndexVar, ...]
    """The index variables used for this access"""

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


@dataclass(frozen=True)
class Assignment(Expr):
    lhs: AccessExpr
    rhs: Expr

    def __repr__(self) -> str:
        return f"Expr({self.lhs} = {self.rhs})"


# ------------------------------------------------------------------------------
# Iteration Graph
# ------------------------------------------------------------------------------


class IterationVarKind(Enum):
    """Defines the role of an index variable in a loop nest"""

    SPATIAL = auto()
    """Dimensions that appear in the output tensor"""

    REDUCTION = auto()
    """Dimensions that are collapsed (e.g., summed over)"""


@dataclass(frozen=True)
class IterationVar:
    """Represents an index variable and its role (spatial or reduction)"""

    iv: IndexVar
    kind: IterationVarKind


@dataclass(frozen=True)
class IterationGraph:
    """Represents the ordered sequence of loops (the loop nest)"""

    vars: tuple[IterationVar, ...]

    def __repr__(self) -> str:
        return "IterationGraph(" + " -> ".join(iv.iv.name for iv in self.vars) + ")"


# ------------------------------------------------------------------------------
# Iterator References
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class IteratorRef:
    """A reference to a specific level of a tensor being iterated."""

    tensor: str
    """Name of the tensor"""

    iv: IndexVar
    """The index variable iterating over this level"""

    level: LevelSpec
    """The specific storage level (e.g., compressed, dense)"""

    def is_full(self) -> bool:
        return self.level.properties.is_full

    def __repr__(self) -> str:
        return f"{self.tensor}[{self.iv}]<{self.level.name}>"


# ------------------------------------------------------------------------------
# Merge Semantics
# ------------------------------------------------------------------------------


class MergeKind(Enum):
    """Specifies how multiple iteration spaces are merged together"""

    UNION = auto()
    """Iterate if ANY tensor has a value (e.g., addition)"""

    INTERSECTION = auto()
    """Iterate only if ALL tensors have values (e.g., multiplication)"""

    LOCATE = auto()
    """Access a specific coordinate (e.g., indexing into a dense dimension)"""


@dataclass(frozen=True)
class LatticePoint:
    """Represents a point in the MergeLattice"""

    tensor_dims: list[TensorDimension]
    """The set of input tensor dimensions"""

    expr: Optional[Expr]
    """The expression to be evaluated at this point"""

    children: List["LatticePoint"] = field(default_factory=list)
    """The successor points in the lattice"""

    def is_terminal(self) -> bool:
        return len(self.children) == 0 and self.expr is None

    def __repr__(self, level=1) -> str:
        indent = "  " * level
        if self.is_terminal():
            return f"{indent}LatticePoint(∅)"

        dims = "{" + ", ".join(str(d) for d in self.tensor_dims) + "}"

        rep = f"{indent}LatticePoint(dims={dims}, expr={self.expr})"
        for child in self.children:
            rep += f"\n{child.__repr__(level + 1)}"
        return rep


# ------------------------------------------------------------------------------
# Merge Lattice
# ------------------------------------------------------------------------------


class MergeLattice:
    "Represents the merge lattice for expression `expr` with index variable `iv`"

    def __init__(self, expr: Expr, iv: IndexVar) -> None:
        self.expr = expr
        self.iv = iv
        self.root = self._build_recursive(self.expr)

    def _build_recursive(self, expr: Expr) -> LatticePoint:
        if isinstance(expr, AccessExpr):
            tensor_dims = [TensorDimension(expr.access.name, self.iv)]
            node = LatticePoint(tensor_dims, expr=expr)
            terminal_node = LatticePoint(tensor_dims=[], expr=None)
            node.children.append(terminal_node)
            return node

        if isinstance(expr, AddExpr):
            left_top = self._build_recursive(expr.left)
            right_top = self._build_recursive(expr.right)

            # TODO: add tensor_dims
            top_node = LatticePoint([], expr=expr)
            top_node.children.append(left_top)
            top_node.children.append(right_top)

            return top_node

        if isinstance(expr, MulExpr):
            left_top = self._build_recursive(expr.left)
            right_top = self._build_recursive(expr.right)

            terminal_node = LatticePoint([], expr=None)

            # TODO: add tensor_dims
            top_node = LatticePoint([], expr=expr)
            top_node.children.append(terminal_node)

            return top_node

        if isinstance(expr, Assignment):
            right_top = self._build_recursive(expr.rhs)
            return self._map_assignment(right_top, expr.lhs)

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _map_assignment(self, point: LatticePoint, lhs: AccessExpr) -> LatticePoint:
        new_expr = Assignment(lhs, point.expr) if point.expr is not None else None
        new_children = [self._map_assignment(c, lhs) for c in point.children]
        return LatticePoint([], expr=new_expr, children=new_children)

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
