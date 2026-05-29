from spjax.merge_lattice import *
from spjax.levels import *

i = IndexVar("i")
j = IndexVar("j")

csr = TensorType(
    shape=(10, 10),
    level_specs=(
        CompressedSpec(),
        SingletonSpec(),
    ),
)

dense = TensorType(
    shape=(10, 10),
    level_specs=(
        DenseSpec(10),
        DenseSpec(10),
    ),
)

A = TensorAccess(
    name="A",
    tensor_type=csr,
    ivs=(i, j),
)

B = TensorAccess(
    name="B",
    tensor_type=dense,
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
