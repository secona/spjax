import jax
from spjax import SparseTensor


@jax.jit
def spmv(X, y):
    return X.dot(y)


def main() -> None:
    print(jax.devices())
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = jax.numpy.ones(32)
    c = spmv(a, b)
    print(c)
    print(spmv.lower(a, b).as_text())

    print(a)


if __name__ == "__main__":
    main()
