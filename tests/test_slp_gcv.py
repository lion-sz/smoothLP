"""Check spline tuning against the full stacked penalized regression."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from smoothlp.slp import SmoothLocalProjection
from tests.slp_reference import (
    assert_estimate_matches,
    fit_reference,
    hac_meat,
    stacked_reference,
)


@pytest.mark.parametrize("H,n_shocks", [(5, 1), (5, 2), (19, 2)])
def test_gcv_matches_full_design(H, n_shocks):
    rng = np.random.default_rng(71)
    T = 80
    controls = np.column_stack((np.ones(T), rng.normal(size=T)))
    shocks = rng.normal(size=(T, n_shocks)) + controls[:, 1:2]
    y = 0.7 * shocks[:, 0] + controls @ np.array([1.0, 0.4]) + rng.normal(size=T)
    model = SmoothLocalProjection(y, shocks, controls, H, target_poly_order=1)
    result = model.fit("gcv", lam_grid_size=31)

    # Construct explicit horizon-specific controls, without Python's partialling
    # out or orthogonalization. Each shock has a consecutive spline block.
    rows, outcomes = [], []
    for t in range(T):
        for h in range(min(H + 1, T - t)):
            rows.append(
                np.concatenate(
                    (
                        np.kron(shocks[t], model.B[h]),
                        np.kron(np.eye(H + 1)[h], controls[t]),
                    )
                )
            )
            outcomes.append(y[t + h])
    X, Y = np.array(rows), np.array(outcomes)
    n_spline_coef = model.B.shape[1] * n_shocks
    assert np.linalg.matrix_rank(X[:, :n_spline_coef]) < n_spline_coef
    P = np.zeros((X.shape[1], X.shape[1]))
    D = np.diff(np.eye(model.B.shape[1]), n=2, axis=0)
    P[:n_spline_coef, :n_spline_coef] = np.kron(np.eye(n_shocks), D.T @ D)
    XX, XY = X.T @ X, X.T @ Y
    grid = len(Y) * 10 ** np.linspace(-3, 1, 31)
    trace, coefs = [], []
    for penalty in grid:
        A = XX + penalty * P
        coef = np.linalg.solve(A, XY)
        rss = np.sum((Y - X @ coef) ** 2)
        df = np.trace(np.linalg.solve(A, XX))
        trace.append(
            [rss, rss / (1 - df / len(Y)) ** 2, np.log(rss) + 2 * df / len(Y), df]
        )
        coefs.append(coef[:n_spline_coef])
    trace = np.array(trace)
    selected = np.argmin(trace[:, 1])
    assert_allclose(result.tune_trace, trace, rtol=1e-6, atol=1e-6)
    assert_allclose(result.lam, grid[selected], rtol=1e-12)
    expected_coef = coefs[selected].reshape(n_shocks, model.B.shape[1]).T
    assert_allclose(result.coef, expected_coef, rtol=1e-6, atol=1e-7)

    reference = fit_reference(
        stacked_reference(y, shocks, controls, model.B, model.n_diff), grid[selected]
    )
    assert_estimate_matches(result, reference)

    # Retain all nuisance coefficients for a second covariance derivation.
    full_coef = np.linalg.solve(XX + grid[selected] * P, XY)
    row_scores = X * (Y - X @ full_coef)[:, None]
    scores = np.zeros((T, X.shape[1]))
    row = 0
    for t in range(T):
        count = min(H + 1, T - t)
        scores[t] = row_scores[row : row + count].sum(axis=0)
        row += count
    bread = np.linalg.solve(XX + 0.1 * grid[selected] * P, np.eye(X.shape[1]))
    covariance = bread @ hac_meat(scores, H) @ bread.T
    assert_allclose(
        reference.vcov_nw,
        covariance[:n_spline_coef, :n_spline_coef],
        rtol=1e-6,
        atol=1e-8,
    )
