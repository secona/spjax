import jax.numpy as jnp

from spjax.ir import SparseIR
from spjax.merge_lattice import (
    AccessExpr,
    Assignment,
    IndexVar,
    IterationGraph,
    IterationVar,
    IterationVarKind,
    MulExpr,
    TensorAccess,
)
from spjax.tensor import SparseTensor, TensorType


def main() -> None:
    a = SparseTensor.from_1d(
        (jnp.array([1, 2, 3, 4]), jnp.array([0, 3, 5, 7])),
        mode="sparse",
    )
    b = SparseTensor.from_1d(
        (jnp.array([1, 2, 3, 5]), jnp.array([1, 3, 4, 9])),
        mode="sparse",
    )

    i = IndexVar("i")

    A = TensorAccess("A", a.tensor_type, (i,))
    B = TensorAccess("B", b.tensor_type, (i,))
    out_type = TensorType(shape=(), level_specs=())
    out = TensorAccess("out", out_type, ())

    expr = Assignment(AccessExpr(out), MulExpr(AccessExpr(A), AccessExpr(B)))
    print(expr)
    print()

    graph = IterationGraph(vars=(IterationVar(i, IterationVarKind.REDUCTION),))
    print(graph)
    print()

    ir = SparseIR(expr, graph)

    for iv, lattice in ir.merge_lattices.items():
        print(lattice)
    print()

    print(ir)


if __name__ == "__main__":
    main()
