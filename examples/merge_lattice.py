from spjax.ir import SparseIR
from spjax.merge_lattice import *
from spjax.levels import *
from spjax.tensor import SparseTensor

a = SparseTensor.from_file("./matrix/ibm32.mtx", fmt="csr")
b = SparseTensor.from_file("./matrix/ibm32.mtx", fmt="coo")

i = IndexVar("i")
j = IndexVar("j")

A = TensorAccess(
    name="A",
    tensor_type=a.tensor_type,
    ivs=(i, j),
)

B = TensorAccess(
    name="B",
    tensor_type=b.tensor_type,
    ivs=(i, j),
)

print(expr := AddExpr(
    AccessExpr(A),
    AccessExpr(B),
))
print()

print(graph := IterationGraph(
    vars=(
        IterationVar(i, IterationVarKind.SPATIAL),
        IterationVar(j, IterationVarKind.SPATIAL),
    )
))
print()

print(Li := MergeLattice(expr, i))
print()

print(Lj := MergeLattice(expr, j))
print()

print(ir := SparseIR(expr, graph, [Li, Lj]))
print()
