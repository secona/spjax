from spjax import SparseTensor
from spjax.ir import SparseIR
from spjax.lowering_ir import lower
from spjax.levels import DenseSpec
from spjax.tensor import TensorType
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


def main() -> None:
    a = SparseTensor.from_file("./matrix/ibm32.mtx")

    b_type = TensorType(shape=(32, 32), level_specs=(DenseSpec(32), DenseSpec(32)))

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")

    A = TensorAccess("A", a.tensor_type, (i, k))
    B = TensorAccess("B", b_type, (k, j))
    C = TensorAccess("C", b_type, (i, j))

    expr = Assignment(AccessExpr(C), MulExpr(AccessExpr(A), AccessExpr(B)))

    graph = IterationGraph(
        vars=(
            IterationVar(i, IterationVarKind.SPATIAL),
            IterationVar(j, IterationVarKind.SPATIAL),
            IterationVar(k, IterationVarKind.REDUCTION),
        )
    )
    print(graph)
    print()

    ir = SparseIR(expr, graph)

    for iv, lattice in ir.merge_lattices.items():
        print(lattice)
    print()

    print(ir)
    print()

    lowering_ir = lower(ir)
    print(lowering_ir)


if __name__ == "__main__":
    main()
