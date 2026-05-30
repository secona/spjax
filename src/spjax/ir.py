from __future__ import annotations

from dataclasses import dataclass

from spjax.merge_lattice import (
    Expr,
    IndexVar,
    IterationGraph,
    IteratorRef,
    MergeKind,
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
        lattices: list[MergeLattice],
    ) -> None:
        self.expr = expr
        self.iteration_graph = iteration_graph
        self.lattices = lattices
        self.root: SparseIRNode | None = None
        self._build()

    def _build(self):
        if not self.lattices or not self.iteration_graph.vars:
            return

        self.root = self._build_level(self.lattices[0].points, depth=0)

    def _build_level(self, points, depth: int):
        if not points:
            return None

        if len(points) == 1:
            point = points[0]
            body = self._descend(point.expr, depth)
            if len(point.iterators) > 1:
                return CoiterateNode(
                    iterators=point.iterators,
                    merge_kind=point.merge_kind,
                    body=body,
                )
            else:
                return body

        sorted_points = sorted(points, key=lambda p: len(p.iterators), reverse=True)
        outer = sorted_points[0]
        inner = sorted_points[1:]

        outer_body = self._descend(outer.expr, depth)
        inner_bodies = [self._descend(p.expr, depth) for p in inner]

        children = [outer_body] + inner_bodies
        body = SequenceNode(tuple(children)) if len(children) > 1 else children[0]

        return CoiterateNode(
            iterators=outer.iterators,
            merge_kind=outer.merge_kind,
            body=body,
        )

    def _descend(self, expr: Expr, depth: int) -> SparseIRNode:
        if depth == len(self.iteration_graph.vars) - 1:
            iv = self.iteration_graph.vars[depth].iv
            return EmitNode(coord=iv, expr=expr)

        next_depth = depth + 1
        next_iv = self.iteration_graph.vars[next_depth].iv
        next_lattice = MergeLattice(expr, next_iv)

        return self._build_level(next_lattice.points, next_depth)

    def __repr__(self) -> str:
        if self.root is None:
            return "SparseIR(empty)"
        return f"SparseIR(expr={self.expr}, graph={self.iteration_graph}) {{\n{self.root}\n}}"


# ------------------------------------------------------------------------------
# IR Nodes
# ------------------------------------------------------------------------------


class SparseIRNode:
    pass


@dataclass(frozen=True)
class CoiterateNode(SparseIRNode):
    iterators: tuple[IteratorRef, ...]
    merge_kind: MergeKind
    body: SparseIRNode

    def __repr__(self) -> str:
        return self._format(0)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        iters = ", ".join(repr(it) for it in self.iterators)
        header = f"{prefix}coiterate([{iters}], {self.merge_kind.name})"

        if isinstance(self.body, (CoiterateNode, SequenceNode)):
            body_str = self.body._format(indent + 1)
            return f"{header} {{\n{body_str}\n{prefix}}}"
        else:
            body_str = self.body._format(indent + 1)
            return f"{header} {{\n{body_str}\n{prefix}}}"


@dataclass(frozen=True)
class EmitNode(SparseIRNode):
    coord: IndexVar
    expr: Expr

    def __repr__(self) -> str:
        return self._format(0)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        return f"{prefix}emit({self.coord}, {self.expr})"


@dataclass(frozen=True)
class SequenceNode(SparseIRNode):
    children: tuple[SparseIRNode, ...]

    def __repr__(self) -> str:
        return self._format(0)

    def _format(self, indent: int) -> str:
        return "\n".join(child._format(indent) for child in self.children)
