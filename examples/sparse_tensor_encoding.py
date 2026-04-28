from spjax import SparseTensor


def main() -> None:
    m = SparseTensor.from_file("./matrix/ibm32.mtx")
    print("nnz:", m.nnz)


if __name__ == "__main__":
    main()
