import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spjax import SparseTensor, sparse_add


class TestSparseAddBasic:
    def test_2x2_exact(self):
        A_dense = np.array([[1, 0], [2, 3]], dtype=np.float32)
        B_dense = np.array([[0, 4], [5, 0]], dtype=np.float32)
        expected = np.array([[1, 4], [7, 3]], dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_add(A, B)
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
            [[1, 6, 2], [7, 3, 8], [4, 9, 5]],
            dtype=np.float32,
        )

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_add(A, B)
        assert jnp.allclose(result, expected)

    def test_random_sparse(self):
        rng = np.random.default_rng(42)

        A_dense = rng.random((5, 7)).astype(np.float32)
        B_dense = rng.random((5, 7)).astype(np.float32)

        # Zero out entries to make them sparse
        A_dense[rng.random(A_dense.shape) < 0.7] = 0
        B_dense[rng.random(B_dense.shape) < 0.7] = 0

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_add(A, B)
        expected = A_dense + B_dense

        assert jnp.allclose(result, expected, atol=1e-5)


class TestSparseAddZero:
    def test_zero_right(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        Z_dense = np.zeros((3, 3), dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        Z = SparseTensor.from_dense(Z_dense)

        result = sparse_add(A, Z)
        expected = A_dense + Z_dense

        assert jnp.allclose(result, expected)
        assert jnp.allclose(result, A_dense)

    def test_zero_left(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        Z_dense = np.zeros((3, 3), dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        Z = SparseTensor.from_dense(Z_dense)

        result = sparse_add(Z, A)
        expected = Z_dense + A_dense

        assert jnp.allclose(result, expected)
        assert jnp.allclose(result, A_dense)

    def test_both_zero(self):
        Z_dense = np.zeros((3, 3), dtype=np.float32)

        Z1 = SparseTensor.from_dense(Z_dense)
        Z2 = SparseTensor.from_dense(Z_dense)

        result = sparse_add(Z1, Z2)
        expected = Z_dense + Z_dense

        assert jnp.allclose(result, expected)
        assert jnp.all(result == 0)


class TestSparseAddJit:
    def test_jitted_matches_eager(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        B_dense = np.array([[0, 6, 0], [7, 0, 8], [0, 9, 0]], dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        eager = sparse_add(A, B)
        jitted = jax.jit(sparse_add)(A, B)

        assert jnp.allclose(eager, jitted)

    def test_jitted_random(self):
        rng = np.random.default_rng(123)

        A_dense = rng.random((8, 10)).astype(np.float32)
        B_dense = rng.random((8, 10)).astype(np.float32)

        A_dense[rng.random(A_dense.shape) < 0.6] = 0
        B_dense[rng.random(B_dense.shape) < 0.6] = 0

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        eager = sparse_add(A, B)
        jitted = jax.jit(sparse_add)(A, B)

        assert jnp.allclose(eager, jitted)


class TestSparseAddShape:
    def test_shape_invariant(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32)
        B_dense = np.array([[0, 4, 0], [5, 0, 6]], dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_add(A, B)
        assert result.shape == (2, 3)


class TestSparseAddFinite:
    def test_no_nan_or_inf(self):
        A_dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=np.float32)
        B_dense = np.array([[0, 6, 0], [7, 0, 8], [0, 9, 0]], dtype=np.float32)

        A = SparseTensor.from_dense(A_dense)
        B = SparseTensor.from_dense(B_dense)

        result = sparse_add(A, B)
        assert jnp.all(jnp.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
