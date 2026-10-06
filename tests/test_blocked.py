import jax.numpy as jnp
import numpy as np
import pytest
from numpy import testing as npt

from smoothlp._blocked import block_solve, blocked_cholesky, fill


CHOLESKY_CASES = [
    pytest.param(2, 5, 3, id="2x2-blocks-3-bands"),
    pytest.param(3, 4, 3, id="3x3-blocks-3-bands"),
    pytest.param(2, 6, 4, id="2x2-blocks-4-bands"),
    pytest.param(3, 5, 4, id="3x3-blocks-4-bands"),
]


SOLVE_CASES = [
    pytest.param(
        n,
        k,
        off_diagonals + 1,
        id=f"{n}x{n}-blocks-{off_diagonals}-off-diagonals",
    )
    for off_diagonals in range(1, 5)
    for n, k in [(2, 5), (3, 6)]
]


def _fill_bands(data, lower=False):
    block_count, band_count, block_size, _ = data.shape
    matrix = np.zeros((block_count * block_size, block_count * block_size))
    for column in range(block_count):
        column_slice = slice(column * block_size, (column + 1) * block_size)
        for offset in range(min(band_count, block_count - column)):
            row = column + offset
            row_slice = slice(row * block_size, (row + 1) * block_size)
            matrix[row_slice, column_slice] = data[column, offset]
            if offset and not lower:
                matrix[column_slice, row_slice] = data[column, offset].T
    return matrix


def _compress_bands(matrix, k, n, band_count=3):
    """Store the requested lower block diagonals."""
    bands = np.zeros((k, band_count, n, n))
    for column in range(k):
        column_slice = slice(column * n, (column + 1) * n)
        for offset in range(min(band_count, k - column)):
            row = column + offset
            row_slice = slice(row * n, (row + 1) * n)
            bands[column, offset] = matrix[row_slice, column_slice]
    return jnp.asarray(bands)


def _assert_blocked_cholesky(bands):
    matrix = _fill_bands(np.asarray(bands))
    actual = _fill_bands(np.asarray(blocked_cholesky(bands)), lower=True)
    expected = np.linalg.cholesky(matrix)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    npt.assert_allclose(actual @ actual.T, matrix, rtol=1e-5, atol=1e-6)


def _matrix_from_factor_blocks(diagonal, lower_bands):
    k, n = len(diagonal), diagonal[0].shape[0]
    factor = np.zeros((k * n, k * n))
    for i, block in enumerate(diagonal):
        factor[i * n : (i + 1) * n, i * n : (i + 1) * n] = block
        for offset, blocks in enumerate(lower_bands, start=1):
            if i + offset < k:
                factor[
                    (i + offset) * n : (i + offset + 1) * n,
                    i * n : (i + 1) * n,
                ] = blocks[i]
    return factor @ factor.T


def _nonsymmetric_bands(n, k, band_count=3):
    diagonal = [
        np.diag(1.1 + 0.1 * i + 0.05 * np.arange(n))
        + np.tril(np.full((n, n), 0.03 * (i + 1)), k=-1)
        for i in range(k)
    ]
    lower_bands = []
    for offset in range(1, band_count):
        pattern = np.arange(1, n * n + 1).reshape(n, n)
        if offset % 2 == 0:
            pattern = pattern[::-1, ::-1]
        lower_bands.append(
            [0.01 * (i + 1) * pattern / offset for i in range(k - offset)]
        )
    matrix = _matrix_from_factor_blocks(diagonal, lower_bands)
    return matrix, _compress_bands(matrix, k, n, band_count)


@pytest.mark.parametrize(("n", "k"), [(2, 5), (3, 4)])
@pytest.mark.parametrize("band_count", [2, 3, 4, 5])
def test_fill_uses_stored_lower_blocks(n, k, band_count):
    bands = np.zeros((k, band_count, n, n))
    expected = np.zeros((k * n, k * n))
    expected_lower = np.zeros_like(expected)
    for i in range(k):
        diagonal = np.arange(n * n).reshape(n, n) + 100 * (i + 1)
        bands[i, 0] = diagonal
        expected[i * n : (i + 1) * n, i * n : (i + 1) * n] = diagonal
        expected_lower[i * n : (i + 1) * n, i * n : (i + 1) * n] = diagonal
        for offset in range(1, band_count):
            if i + offset >= k:
                continue
            lower = np.arange(n * n).reshape(n, n) + 10 * offset + i
            bands[i, offset] = lower
            row = slice((i + offset) * n, (i + offset + 1) * n)
            column = slice(i * n, (i + 1) * n)
            expected[row, column] = lower
            expected[column, row] = lower.T
            expected_lower[row, column] = lower

    npt.assert_array_equal(fill(jnp.asarray(bands)), expected)
    npt.assert_array_equal(fill(jnp.asarray(bands), lower=True), expected_lower)


@pytest.mark.parametrize(("n", "k", "band_count"), CHOLESKY_CASES)
def test_blocked_cholesky_constant_symmetric_blocks(n, k, band_count):
    block = np.eye(n) + np.ones((n, n))
    bands = np.zeros((k, band_count, n, n))
    bands[:, 0] = block
    for offset in range(1, band_count):
        bands[:-offset, offset] = 0.5**offset * block

    _assert_blocked_cholesky(jnp.asarray(bands))


@pytest.mark.parametrize(("n", "k", "band_count"), CHOLESKY_CASES)
def test_blocked_cholesky_varying_symmetric_blocks(n, k, band_count):
    diagonal = [
        np.diag(1.0 + 0.08 * i + 0.05 * np.arange(n))
        + np.tril(np.full((n, n), 0.02 * (i + 1)), k=-1)
        for i in range(k)
    ]
    lower_bands = [
        [
            np.diag(0.08 / 4 ** (offset - 1) + 0.005 * i + 0.003 * np.arange(n))
            for i in range(k - offset)
        ]
        for offset in range(1, band_count)
    ]
    matrix = _matrix_from_factor_blocks(diagonal, lower_bands)

    _assert_blocked_cholesky(_compress_bands(matrix, k, n, band_count))


@pytest.mark.parametrize(("n", "k", "band_count"), CHOLESKY_CASES)
def test_blocked_cholesky_nonsymmetric_off_diagonal_blocks(n, k, band_count):
    matrix, bands = _nonsymmetric_bands(n, k, band_count)

    assert np.linalg.eigvalsh(matrix).min() > 0
    assert not np.allclose(bands[0, 1], bands[0, 1].T)
    _assert_blocked_cholesky(bands)


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("rhs_count", [1, 4], ids=["single-rhs", "multiple-rhs"])
@pytest.mark.parametrize(("n", "k", "band_count"), SOLVE_CASES)
def test_block_solve_matches_dense_triangular_solve(lower, rhs_count, n, k, band_count):
    # Construct the factor directly to test the solver independently of Cholesky.
    random = np.random.default_rng(123)
    bands = np.zeros((k, band_count, n, n))
    for column in range(k):
        diagonal = np.tril(random.normal(scale=0.1, size=(n, n)))
        np.fill_diagonal(diagonal, 1.1 + 0.1 * column + 0.05 * np.arange(n))
        bands[column, 0] = diagonal
        for offset in range(1, min(band_count, k - column)):
            bands[column, offset] = random.normal(scale=0.1, size=(n, n))

    chol = jnp.asarray(bands)
    dense_chol = _fill_bands(bands, lower=True)
    triangular = dense_chol if lower else dense_chol.T

    for a in random.normal(size=(3, k, n, rhs_count)):
        actual = np.asarray(block_solve(chol, jnp.asarray(a), lower=lower))
        expected = np.linalg.solve(triangular, a.reshape(k * n, rhs_count)).reshape(
            k, n, rhs_count
        )

        assert actual.shape == a.shape
        npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        npt.assert_allclose(
            triangular @ actual.reshape(k * n, rhs_count),
            a.reshape(k * n, rhs_count),
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize(("n", "k", "band_count"), SOLVE_CASES)
def test_block_solve_solves_cholesky_system(n, k, band_count):
    matrix, bands = _nonsymmetric_bands(n, k, band_count)
    chol = blocked_cholesky(bands)
    rhs = np.random.default_rng(789).normal(size=(k, n, 3))

    intermediate = block_solve(chol, jnp.asarray(rhs))
    actual = np.asarray(block_solve(chol, intermediate, lower=False))
    expected = np.linalg.solve(matrix, rhs.reshape(k * n, 3)).reshape(k, n, 3)

    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    npt.assert_allclose(
        matrix @ actual.reshape(k * n, 3),
        rhs.reshape(k * n, 3),
        rtol=1e-5,
        atol=1e-6,
    )
