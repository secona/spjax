import jax
from spjax import SparseTensor


@jax.jit
def spmm(X, Y):
    return X.dot(Y)


def main() -> None:
    print(jax.devices())
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_file("./matrix/Hamrle1.mtx")
    c = spmm(a, b)
    print(c)
    print(spmm.lower(a, b).as_text())


if __name__ == "__main__":
    main()
