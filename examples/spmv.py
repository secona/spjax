import jax
from spjax import SparseTensor, sparse_dot


@jax.jit
def spmv(X, y):
    return sparse_dot(X, y)


def main() -> None:
    print(jax.devices())
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = jax.numpy.ones(32)
    c = spmv(a, b)
    print(c)
    print(spmv.lower(a, b).as_text())


if __name__ == "__main__":
    main()
