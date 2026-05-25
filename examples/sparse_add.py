import jax
from spjax import SparseTensor
from spjax.primitive import sparse_add


def main() -> None:
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_file("./matrix/Hamrle1.mtx")
    print(jax.jit(sparse_add).lower(a, b).as_text())

    c = sparse_add(a, b)
    c = SparseTensor.from_dense(c)
    print(c)


if __name__ == "__main__":
    main()
