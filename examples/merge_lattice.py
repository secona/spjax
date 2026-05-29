from spjax.merge_lattice import *
from spjax.levels import *
from spjax.tensor import SparseTensor

a = SparseTensor.from_file("./matrix/ibm32.mtx")
b = SparseTensor.from_file("./matrix/ibm32.mtx")

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

expr = AddExpr(
    AccessExpr(A),
    AccessExpr(B),
)

graph = IterationGraph(
    vars=(
        IterationVar(i, IterationVarKind.SPATIAL),
        IterationVar(j, IterationVarKind.SPATIAL),
    )
)

print("Expression:")
print(expr)

print("\nIteration Graph:")
print(graph)

print("\nMerge Lattice for i:")
print(MergeLattice(expr, i))

print("\nMerge Lattice for j:")
print(MergeLattice(expr, j))
