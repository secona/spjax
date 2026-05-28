import jax
from spjax import SparseTensor


def main() -> None:
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_file("./matrix/Hamrle1.mtx")
    print(jax.jit(lambda a, b: a.add(b)).lower(a, b).as_text())

    c = a.add(b)
    c = SparseTensor.from_dense(c)
    print(c)


if __name__ == "__main__":
    main()
