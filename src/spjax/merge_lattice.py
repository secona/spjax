from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from spjax.levels import (
    CompressedSpec,
    DenseSpec,
    IterationKind,
    LevelSpec,
    SingletonSpec,
)


@dataclass(frozen=True)
class IndexVar:
    name: str


@dataclass
class Dimension:
    name: str
    iv: IndexVar
    spec: LevelSpec


@dataclass
class TensorAccess:
    name: str
    dims: list[Dimension]


class OpKind(Enum):
    ADD = "+"
    MUL = "*"


@dataclass
class Expression:
    op: Optional[OpKind] = None
    left: Optional[Expression] = None
    right: Optional[Expression] = None
    access: Optional[TensorAccess] = None

    @staticmethod
    def leaf(access: TensorAccess) -> Expression:
        return Expression(access=access)

    @staticmethod
    def add(left: Expression, right: Expression) -> Expression:
        return Expression(left=left, right=right, op=OpKind.ADD)

    @staticmethod
    def mul(left: Expression, right: Expression) -> Expression:
        return Expression(left=left, right=right, op=OpKind.MUL)

    @property
    def is_leaf(self):
        return self.access is not None


class IterationGraph:
    def __init__(
        self, ivs: list[IndexVar], reduction: Optional[set[IndexVar]] = None
    ) -> None:
        self.ivs = ivs
        self.reduction = reduction


class OutputTensor:
    def __init__(self, name: str, dims: list[Dimension]) -> None:
        self.name = name
        self.dims = dims


class CodeGen:
    def __init__(
        self, expr: Expression, iter_graph: IterationGraph, out: OutputTensor
    ) -> None:
        self.expr = expr
        self.iter_graph = iter_graph
        self.out = out

    def generate(self):
        for iv in self.iter_graph.ivs:
            self.__codegen(self.expr, iv)

    def __codegen(self, expr, iv: IndexVar):
        lattice = MergeLattice(expr, iv)
        del lattice


class LatticePoint:
    def __init__(self, expr: Expression) -> None:
        self.expr = expr


class MergeLattice:
    def __init__(self, expr: Expression, iv: IndexVar) -> None:
        self.expr = expr
        self.iv = iv
        self.points: list[LatticePoint] = []

    def __coiter_and_locate(self, expr: Expression, iv: IndexVar):
        if expr.is_leaf:
            dims = [d for d in expr.access.dims if d.iv == iv]
            coiter = [d for d in dims if d.spec.supports(IterationKind.POSITION)]
            locate = [d for d in dims if d.spec.supports(IterationKind.LOCATE)]
            return coiter, locate

        return


def __example():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")

    A_i = Dimension("A", i, CompressedSpec())
    A_k = Dimension("A", k, SingletonSpec())
    B_k = Dimension("B", k, CompressedSpec())
    B_j = Dimension("B", j, DenseSpec(10))
    C_i = Dimension("C", i, DenseSpec(10))
    C_j = Dimension("C", j, DenseSpec(10))

    expr = Expression.mul(
        Expression.leaf(TensorAccess("A", [A_i, A_k])),
        Expression.leaf(TensorAccess("B", [B_k, B_j])),
    )

    cg = CodeGen(
        expr, IterationGraph([i, k, j], reduction={k}), OutputTensor("C", [C_i, C_j])
    )
    cg.generate()


del __example
