import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.io
import scipy.sparse

from spjax import SparseTensor, sparse_dot


class TestSpMMIdentity:
    def test_identity_right(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        I_dense = np.eye(3, dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        I = SparseTensor.from_dense(I_dense)

        result = sparse_dot(A, I)
        expected = A_dense @ I_dense

        assert jnp.allclose(result, expected)

    def test_identity_left(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        I_dense = np.eye(3, dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        I = SparseTensor.from_dense(I_dense)

        result = sparse_dot(I, A)
        expected = I_dense @ A_dense

        assert jnp.allclose(result, expected)


class TestSpMMZero:
    def test_zero_right(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        Z_dense = np.zeros((3, 3), dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        Z = SparseTensor.from_dense(Z_dense)

        result = sparse_dot(A, Z)
        expected = A_dense @ Z_dense

        assert jnp.allclose(result, expected)
        assert jnp.all(result == 0)

    def test_zero_left(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        Z_dense = np.zeros((3, 3), dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        Z = SparseTensor.from_dense(Z_dense)

        result = sparse_dot(Z, A)
        expected = Z_dense @ A_dense

        assert jnp.allclose(result, expected)
        assert jnp.all(result == 0)


class TestSpMMHandCrafted:
    def test_2x2_exact(self):
        A_dense = np.array([[1, 0], [2, 3]], dtype=np.float32)
        B_dense = np.array([[0, 4], [5, 0]], dtype=np.float32)
        expected = np.array([[0, 4], [15, 8]], dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_dot(A, B)
        assert jnp.allclose(result, expected)

    def test_3x3_exact(self):
        A_dense = np.array(
            [[1, 0, 2], [0, 3, 0], [4, 0, 5]],
            dtype=np.float32,
        )
        B_dense = np.array(
            [[0, 6, 0], [7, 0, 8], [0, 9, 0]],
            dtype=np.float32,
        )
        expected = np.array(
            [[0, 24, 0], [21, 0, 24], [0, 69, 0]],
            dtype=np.float32,
        )

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_dot(A, B)
        assert jnp.allclose(result, expected)

    def test_non_square(self):
        A_dense = np.array([[1, 0, 3], [0, 2, 0]], dtype=np.float32)  # 2x3
        B_dense = np.array([[0, 4], [5, 0], [0, 6]], dtype=np.float32)  # 3x2
        expected = np.array([[0, 22], [10, 0]], dtype=np.float32)  # 2x2

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_dot(A, B)
        assert jnp.allclose(result, expected)
        assert result.shape == (2, 2)


class TestSpMMShape:
    def test_shape_invariant(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32)  # 2x3
        B_dense = np.array([[0, 4], [5, 0], [0, 6]], dtype=np.float32)  # 3x2

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_dot(A, B)
        assert result.shape == (2, 2)


class TestSpMMFinite:
    def test_no_nan_or_inf(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        B_dense = np.array([[0, 6, 0], [7, 0, 8], [0, 9, 0]], dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_dot(A, B)
        assert jnp.all(jnp.isfinite(result))


class TestSpMMAgainstSciPy:
    def test_ibm32_hamrle1(self):
        a = SparseTensor.from_file("./matrix/ibm32.mtx")
        b = SparseTensor.from_file("./matrix/Hamrle1.mtx")

        result = sparse_dot(a, b)

        a_scipy = scipy.io.mmread("./matrix/ibm32.mtx").tocsr()
        b_scipy = scipy.io.mmread("./matrix/Hamrle1.mtx").tocsr()
        expected = (a_scipy @ b_scipy).toarray()

        assert jnp.allclose(result, expected, atol=1e-6)

    def test_random_small(self):
        rng = np.random.default_rng(42)

        for _ in range(5):
            A_dense = rng.random((8, 12)).astype(np.float32)
            B_dense = rng.random((12, 10)).astype(np.float32)

            # Zero out 70% of entries to make them sparse
            A_dense[rng.random(A_dense.shape) < 0.7] = 0
            B_dense[rng.random(B_dense.shape) < 0.7] = 0

            A = SparseTensor.from_dense(A_dense)
            B = SparseTensor.from_dense(B_dense)

            result = sparse_dot(A, B)
            expected = A_dense @ B_dense

            assert jnp.allclose(result, expected, atol=1e-5)

    def test_random_rectangular(self):
        rng = np.random.default_rng(123)

        A_dense = rng.random((5, 20)).astype(np.float32)
        B_dense = rng.random((20, 7)).astype(np.float32)

        A_dense[rng.random(A_dense.shape) < 0.8] = 0
        B_dense[rng.random(B_dense.shape) < 0.8] = 0

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_dot(A, B)
        expected = A_dense @ B_dense

        assert jnp.allclose(result, expected, atol=1e-5)
        assert result.shape == (5, 7)


class TestSpMMJit:
    def test_jitted_spmm_matches_eager(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        B_dense = np.array([[0, 6, 0], [7, 0, 8], [0, 9, 0]], dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        eager = sparse_dot(A, B)
        jitted = jax.jit(sparse_dot)(A, B)

        assert jnp.allclose(eager, jitted)

    def test_jitted_spmm_against_scipy(self):
        a = SparseTensor.from_file("./matrix/ibm32.mtx")
        b = SparseTensor.from_file("./matrix/Hamrle1.mtx")

        jitted = jax.jit(sparse_dot)(a, b)

        a_scipy = scipy.io.mmread("./matrix/ibm32.mtx").tocsr()
        b_scipy = scipy.io.mmread("./matrix/Hamrle1.mtx").tocsr()
        expected = (a_scipy @ b_scipy).toarray()

        assert jnp.allclose(jitted, expected, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
