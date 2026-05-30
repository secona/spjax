from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from spjax.merge_lattice import (
    Expr,
    IndexVar,
    IterationGraph,
    IterationVarKind,
    IteratorRef,
    LatticePoint,
    MergeLattice,
)

# ------------------------------------------------------------------------------
# IR
# ------------------------------------------------------------------------------


class SparseIR:
    def __init__(
        self,
        expr: Expr,
        iteration_graph: IterationGraph,
    ) -> None:
        self.expr = expr
        self.iteration_graph = iteration_graph
        self.merge_lattices: dict[IndexVar, MergeLattice] = {}
        self.root: Optional[SparseIRNode] = None
        self._build()

    def _lattice_for(self, iv: IndexVar) -> MergeLattice:
        if iv not in self.merge_lattices:
            self.merge_lattices[iv] = MergeLattice(self.expr, iv)
        return self.merge_lattices[iv]

    def _build(self):
        if not self.iteration_graph.vars:
            return

        first_iv = self.iteration_graph.vars[0].iv
        self.root = self._build_level(self._lattice_for(first_iv).root, depth=0)

    def _build_level(self, root: LatticePoint, depth: int) -> Optional[SparseIRNode]:
        if root is None or root.is_terminal():
            return None

        body_node = self._lower_dag(root, depth)

        if body_node is None:
            return None

        iv = self.iteration_graph.vars[depth].iv
        kind = self.iteration_graph.vars[depth].kind
        return ForNode(iv=iv, kind=kind, body=body_node)

    def _lower_dag(self, point: LatticePoint, depth: int) -> Optional[SparseIRNode]:
        if point.is_terminal():
            return None

        body = self._descend(point.expr, depth)

        node = CoiterateNode(
            iterators=point.iterators,
            body=body,
        )

        return node

    def _descend(self, expr: Expr, depth: int) -> SparseIRNode:
        if depth == len(self.iteration_graph.vars) - 1:
            coord = tuple(v.iv for v in self.iteration_graph.vars)
            return EmitNode(coord=coord, expr=expr)

        next_depth = depth + 1
        next_iv = self.iteration_graph.vars[next_depth].iv
        next_lattice = self._lattice_for(next_iv)

        return self._build_level(next_lattice.root, next_depth)

    def __repr__(self) -> str:
        if self.root is None:
            return "SparseIR(empty)"
        return (
            f"SparseIR(expr={self.expr}, graph={self.iteration_graph}): \n{self.root}"
        )


# ------------------------------------------------------------------------------
# IR Nodes
# ------------------------------------------------------------------------------


class SparseIRNode:
    pass


@dataclass(frozen=True)
class CoiterateNode(SparseIRNode):
    iterators: tuple[IteratorRef, ...]
    body: SparseIRNode

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        iters = ", ".join(repr(it) for it in self.iterators)
        header = f"{prefix}CoIterateNode([{iters}])"

        body_str = self.body._format(indent + 1)
        return f"{header}: \n{body_str}"


@dataclass(frozen=True)
class ForNode(SparseIRNode):
    iv: IndexVar
    kind: IterationVarKind
    body: SparseIRNode

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        header = f"{prefix}ForNode({self.iv}, {self.kind.name})"
        body_str = self.body._format(indent + 1)
        return f"{header}: \n{body_str}"


@dataclass(frozen=True)
class EmitNode(SparseIRNode):
    coord: tuple[IndexVar, ...]
    expr: Expr

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        return f"{prefix}EmitNode({self.coord}, {self.expr})"


@dataclass(frozen=True)
class SequenceNode(SparseIRNode):
    children: tuple[SparseIRNode, ...]

    def __repr__(self) -> str:
        return self._format(0)

    def _format(self, indent: int) -> str:
        return "\n".join(child._format(indent) for child in self.children)
