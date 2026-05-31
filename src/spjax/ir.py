from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from spjax.merge_lattice import (
    AccessExpr,
    AddExpr,
    Assignment,
    Expr,
    IndexVar,
    IterationGraph,
    IterationVarKind,
    IteratorRef,
    MergeKind,
    MergeLattice,
    MulExpr,
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

    def _build(self):
        if not self.iteration_graph.vars:
            return

        if isinstance(self.expr, Assignment):
            output = self.expr.lhs
            rhs = self.expr.rhs
        else:
            output = None
            rhs = self.expr

        self.root = self._build_level(0, output, rhs)

    def _build_level(
        self, depth: int, output: Optional[AccessExpr], rhs: Expr
    ) -> SparseIRNode:
        if depth >= len(self.iteration_graph.vars):
            return EmitNode(output=output, expr=rhs)

        iv = self.iteration_graph.vars[depth].iv
        kind = self.iteration_graph.vars[depth].kind

        lattice = MergeLattice(rhs, iv)
        self.merge_lattices[iv] = lattice

        merge_kind = self._get_merge_kind(rhs)
        body = self._build_level(depth + 1, output, rhs)

        coiterate = CoiterateNode(
            iv=iv,
            iterators=lattice.root.iterators,
            merge=merge_kind,
            body=body,
        )

        return ForNode(iv=iv, kind=kind, body=coiterate)

    def _get_merge_kind(self, expr: Expr) -> MergeKind:
        if isinstance(expr, AddExpr):
            return MergeKind.UNION
        if isinstance(expr, MulExpr):
            return MergeKind.INTERSECTION
        return MergeKind.LOCATE

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
    iv: IndexVar
    iterators: tuple[IteratorRef, ...]
    merge: MergeKind
    body: SparseIRNode

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        iters = ", ".join(repr(it) for it in self.iterators)
        header = f"{prefix}CoIterateNode(iv={self.iv}, iterators=[{iters}], merge={self.merge.name})"

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
    output: Optional[AccessExpr]
    expr: Expr

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        return f"{prefix}EmitNode(output={self.output}, expr={self.expr})"
