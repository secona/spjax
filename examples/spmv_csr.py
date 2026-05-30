import jax
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
    a = SparseTensor.from_file("./matrix/ibm32.mtx", fmt="csr")
    b = SparseTensor.from_1d(jax.numpy.ones(32), mode="dense")

    i, j = IndexVar("i"), IndexVar("j")

    A = TensorAccess("A", a.tensor_type, (i, j))
    B = TensorAccess("B", b.tensor_type, (j,))

    expr = MulExpr(AccessExpr(A), AccessExpr(B))

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


if __name__ == "__main__":
    main()
