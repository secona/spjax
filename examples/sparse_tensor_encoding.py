import jax
from spjax import SparseTensor


def add(x, y):
    return x + y


def main() -> None:
    print(jax.devices())
    a = SparseTensor.from_file("./matrix/ibm32.mtx")
    b = SparseTensor.from_file("./matrix/Hamrle1.mtx")
    c = jax.jit(add)(a, b)
    print(c.to_dense_str())
    print(jax.jit(add).lower(a, b).as_text())


if __name__ == "__main__":
    main()
