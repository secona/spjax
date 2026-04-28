from spjax import SparseTensor


def main() -> None:
    SparseTensor.from_file("./matrix/ibm32.mtx")


if __name__ == "__main__":
    main()
