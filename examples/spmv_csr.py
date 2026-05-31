import jax
from spjax import SparseTensor
from spjax.ir import SparseIR
from spjax.lowering_ir import lower
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
    a = SparseTensor.from_file("./matrix/ibm32.mtx", fmt="csr")
    b = SparseTensor.from_1d(jax.numpy.ones(32), mode="dense")

    i, j = IndexVar("i"), IndexVar("j")

    A = TensorAccess("A", a.tensor_type, (i, j))
    B = TensorAccess("B", b.tensor_type, (j,))
    y_out = TensorAccess("y_out", b.tensor_type, (i,))

    expr = Assignment(AccessExpr(y_out), MulExpr(AccessExpr(A), AccessExpr(B)))

    graph = IterationGraph(
        vars=(
            IterationVar(i, IterationVarKind.SPATIAL),
            IterationVar(j, IterationVarKind.REDUCTION),
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
