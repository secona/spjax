from spjax import SparseTensor
from spjax.ir import SparseIR
from spjax.lowering_ir import lower
from spjax.merge_lattice import (
    AccessExpr,
    AddExpr,
    Assignment,
    IndexVar,
    IterationGraph,
    IterationVar,
    IterationVarKind,
    TensorAccess,
)


def main() -> None:
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_file("./matrix/Hamrle1.mtx")

    i, j = IndexVar("i"), IndexVar("j")

    A = TensorAccess("A", a.tensor_type, (i, j))
    B = TensorAccess("B", b.tensor_type, (i, j))
    C = TensorAccess("C", a.tensor_type, (i, j))

    expr = Assignment(AccessExpr(C), AddExpr(AccessExpr(A), AccessExpr(B)))
    print(expr)
    print()

    graph = IterationGraph(
        vars=(
            IterationVar(i, IterationVarKind.SPATIAL),
            IterationVar(j, IterationVarKind.SPATIAL),
        )
    )
    print(graph)
    print()

    ir = SparseIR(expr, graph)

    for iv, lattice in ir.merge_lattices.items():
        print(lattice)
        print()
    print()

    print(ir)
    print()

    lowering_ir = lower(ir)
    print(lowering_ir)


if __name__ == "__main__":
    main()
