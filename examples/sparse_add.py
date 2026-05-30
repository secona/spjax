from spjax import SparseTensor
from spjax.ir import SparseIR
from spjax.merge_lattice import (
    AccessExpr,
    AddExpr,
    IndexVar,
    IterationGraph,
    IterationVar,
    IterationVarKind,
    MergeLattice,
    TensorAccess,
)


def main() -> None:
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_file("./matrix/Hamrle1.mtx")

    i, j = IndexVar("i"), IndexVar("j")

    A = TensorAccess("A", a.tensor_type, (i, j))
    B = TensorAccess("B", b.tensor_type, (i, j))

    print(
        expr := AddExpr(
            AccessExpr(A),
            AccessExpr(B),
        )
    )
    print()

    print(
        graph := IterationGraph(
            vars=(
                IterationVar(i, IterationVarKind.SPATIAL),
                IterationVar(j, IterationVarKind.SPATIAL),
            )
        )
    )
    print()

    print(Li := MergeLattice(expr, i))
    print()

    print(Lj := MergeLattice(expr, j))
    print()

    print(SparseIR(expr, graph, [Li, Lj]))
    print()


if __name__ == "__main__":
    main()
