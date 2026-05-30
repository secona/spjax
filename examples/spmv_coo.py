import jax
from spjax import SparseTensor
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


@jax.jit
def spmv(X, y):
    return X.dot(y)


def main() -> None:
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_1d(jax.numpy.ones(32), mode="dense")

    i, j = IndexVar("i"), IndexVar("j")

    A = TensorAccess("A", a.tensor_type, (i, j))
    B = TensorAccess("B", b.tensor_type, (j,))

    expr = MulExpr(AccessExpr(A), AccessExpr(B))

    print(
        graph := IterationGraph(
            vars=(
                IterationVar(i, IterationVarKind.SPATIAL),
                IterationVar(j, IterationVarKind.REDUCTION),
            )
        )
    )
    print()

    print(Li := MergeLattice(expr, i))
    print()

    print(Lk := MergeLattice(expr, k))
    print()

    print(Lj := MergeLattice(expr, j))
    print()

    print(SparseIR(expr, graph, [Li, Lk, Lj]))
    print()


if __name__ == "__main__":
    main()
