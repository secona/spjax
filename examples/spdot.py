import jax.numpy as jnp

from spjax.ir import SparseIR
from spjax.merge_lattice import (
    AccessExpr,
    IndexVar,
    IterationGraph,
    IterationVar,
    IterationVarKind,
    MergeLattice,
    MulExpr,
    TensorAccess,
)
from spjax.tensor import SparseTensor


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

    print(expr := MulExpr(AccessExpr(A), AccessExpr(B)))
    print()

    print(graph := IterationGraph(vars=(IterationVar(i, IterationVarKind.REDUCTION),)))
    print()

    print(Li := MergeLattice(expr, i))
    print()

    print(SparseIR(expr, graph, [Li]))
    print()


if __name__ == "__main__":
    main()
