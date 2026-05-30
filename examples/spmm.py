from spjax import SparseTensor
from spjax.ir import SparseIR
from spjax.merge_lattice import (
    AccessExpr,
    IndexVar,
    IterationGraph,
    IterationVar,
    IterationVarKind,
    MulExpr,
    TensorAccess,
)


def main() -> None:
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_file("./matrix/Hamrle1.mtx")

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")

    A = TensorAccess("A", a.tensor_type, (i, k))
    B = TensorAccess("B", b.tensor_type, (k, j))

    expr = MulExpr(AccessExpr(A), AccessExpr(B))
    print(expr)
    print()

    graph = IterationGraph(
        vars=(
            IterationVar(i, IterationVarKind.SPATIAL),
            IterationVar(k, IterationVarKind.REDUCTION),
            IterationVar(j, IterationVarKind.SPATIAL),
        )
    )
    print(graph)
    print()

    ir = SparseIR(expr, graph)

    for iv, lattice in ir.merge_lattices.items():
        print(lattice)
    print()

    print(ir)


if __name__ == "__main__":
    main()
