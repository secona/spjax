from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from spjax.merge_lattice import (
    AccessExpr,
    Expr,
    IndexVar,
    IteratorRef,
    MergeKind,
)
from spjax.ir import (
    SparseIR,
    SparseIRNode,
    ForNode,
    CoiterateNode,
    EmitNode as SparseEmitNode,
)


# ------------------------------------------------------------------------------
# Lowering IR Nodes
# ------------------------------------------------------------------------------


class LoweringIRNode:
    def _format(self, indent: int) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class DenseTraverseNode(LoweringIRNode):
    iv: IndexVar
    iterator: IteratorRef
    body: LoweringIRNode

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        header = f"{prefix}DenseTraverseNode(iv={self.iv}, iterator={self.iterator})"
        body_str = self.body._format(indent + 1)
        return f"{header}: \n{body_str}"


@dataclass(frozen=True)
class UnionMergeNode(LoweringIRNode):
    iv: IndexVar
    iterators: tuple[IteratorRef, ...]
    body: LoweringIRNode

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        iters = ", ".join(repr(it) for it in self.iterators)
        header = f"{prefix}UnionMergeNode(iv={self.iv}, iterators=[{iters}])"
        body_str = self.body._format(indent + 1)
        return f"{header}: \n{body_str}"


@dataclass(frozen=True)
class IntersectionMergeNode(LoweringIRNode):
    iv: IndexVar
    iterators: tuple[IteratorRef, ...]
    body: LoweringIRNode

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        iters = ", ".join(repr(it) for it in self.iterators)
        header = f"{prefix}IntersectionMergeNode(iv={self.iv}, iterators=[{iters}])"
        body_str = self.body._format(indent + 1)
        return f"{header}: \n{body_str}"


@dataclass(frozen=True)
class LocateNode(LoweringIRNode):
    iv: IndexVar
    iterators: tuple[IteratorRef, ...]
    body: LoweringIRNode

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        iters = ", ".join(repr(it) for it in self.iterators)
        header = f"{prefix}LocateNode(iv={self.iv}, iterators=[{iters}])"
        body_str = self.body._format(indent + 1)
        return f"{header}: \n{body_str}"


@dataclass(frozen=True)
class ReduceNode(LoweringIRNode):
    output: AccessExpr
    expr: Expr

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        return f"{prefix}ReduceNode(output={self.output}, expr={self.expr})"


@dataclass(frozen=True)
class EmitNode(LoweringIRNode):
    expr: Expr

    def __repr__(self) -> str:
        return self._format(1)

    def _format(self, indent: int) -> str:
        prefix = "  " * indent
        return f"{prefix}EmitNode(expr={self.expr})"


# ------------------------------------------------------------------------------
# Lowering
# ------------------------------------------------------------------------------


class LoweringIR:
    def __init__(self, sparse_ir: SparseIR) -> None:
        self.sparse_ir = sparse_ir
        self.root: Optional[LoweringIRNode] = None
        self._lower()

    def _lower(self):
        if self.sparse_ir.root is None:
            return
        self.root = self._lower_node(self.sparse_ir.root)

    def _lower_node(self, node: SparseIRNode) -> LoweringIRNode:
        if isinstance(node, ForNode):
            if not isinstance(node.body, CoiterateNode):
                raise ValueError(
                    f"Expected CoiterateNode as body of ForNode, got {type(node.body)}"
                )

            coiterate = node.body
            iv = node.iv
            iterators = coiterate.iterators
            body = self._lower_node(coiterate.body)

            if len(iterators) == 1 and iterators[0].is_full():
                return DenseTraverseNode(iv=iv, iterator=iterators[0], body=body)

            if coiterate.merge == MergeKind.UNION:
                return UnionMergeNode(iv=iv, iterators=iterators, body=body)

            if coiterate.merge == MergeKind.INTERSECTION:
                any_full = any(it.is_full() for it in iterators)
                any_sparse = any(not it.is_full() for it in iterators)

                if any_full and any_sparse:
                    return LocateNode(iv=iv, iterators=iterators, body=body)

                return IntersectionMergeNode(iv=iv, iterators=iterators, body=body)

            if coiterate.merge == MergeKind.LOCATE:
                return LocateNode(iv=iv, iterators=iterators, body=body)

            raise ValueError(f"Unsupported merge kind: {coiterate.merge}")

        if isinstance(node, SparseEmitNode):
            if node.output is not None:
                return ReduceNode(output=node.output, expr=node.expr)
            else:
                return EmitNode(expr=node.expr)

        raise ValueError(f"Unsupported SparseIRNode: {type(node)}")

    def __repr__(self) -> str:
        if self.root is None:
            return "LoweringIR(empty)"
        return f"LoweringIR: \n{self.root}"


def lower(sparse_ir: SparseIR) -> LoweringIR:
    return LoweringIR(sparse_ir)
