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


def main() -> None:
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_file("./matrix/Hamrle1.mtx")

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")

    A = TensorAccess("A", a.tensor_type, (i, k))
    B = TensorAccess("B", b.tensor_type, (k, j))

    print(expr := MulExpr(AccessExpr(A), AccessExpr(B)))
    print()

    print(
        graph := IterationGraph(
            vars=(
                IterationVar(i, IterationVarKind.SPATIAL),
                IterationVar(k, IterationVarKind.REDUCTION),
                IterationVar(j, IterationVarKind.SPATIAL),
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
