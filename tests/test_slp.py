"""SLP estimation, GCV/CV, and equivalent public fitting paths.

The reference shares only the public spline basis with production. Joint
estimation is referenced by promoting controls to penalized shocks.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from smoothlp.slp import SmoothLocalProjection
from tests.slp_reference import (
    assert_estimate_matches,
    fit_reference,
    hac_meat,
    simulated_data,
    stacked_reference,
)


HORIZONS = [4]
SHOCK_COUNTS = [1, 3]
CONTROL_COUNTS = [0, 2]
POLYNOMIAL_ORDERS = [2, 3]
FIXED_LAMBDAS = [1.0, 10.0]
GRID_SIZE = 11


def _select_shocks(result, n_shocks):
    """Select original shocks, including all their cross-covariance blocks."""
    n_coef = result.coef.shape[0] * n_shocks
    return SimpleNamespace(
        coef=result.coef[:, :n_shocks],
        irf=result.irf[:, :n_shocks],
        se=result.se[:, :n_shocks],
        vcov_nw=result.vcov_nw[:n_coef, :n_coef],
    )


def _reference_inputs(shocks, controls, partial_out):
    if partial_out:
        return shocks, controls
    return np.column_stack((shocks, controls)), np.empty((len(shocks), 0))


@pytest.mark.parametrize("H", HORIZONS)
@pytest.mark.parametrize("n_shocks", SHOCK_COUNTS)
@pytest.mark.parametrize("n_controls", CONTROL_COUNTS)
@pytest.mark.parametrize("target_poly_order", POLYNOMIAL_ORDERS)
@pytest.mark.parametrize("masked", [False, True], ids=["complete", "gaps"])
@pytest.mark.parametrize("partial_out", [True, False], ids=["partial", "joint"])
@pytest.mark.parametrize("lam", FIXED_LAMBDAS)
def test_fixed_lambda_matches_reference(
    H, n_shocks, n_controls, target_poly_order, masked, partial_out, lam
):
    y, shocks, controls, mask = simulated_data(
        T=320 if H == 100 else 80,
        n_shocks=n_shocks,
        n_controls=n_controls,
        masked=masked,
    )
    # Exercise the public 1D single-shock input while the reference stays 2D.
    model = SmoothLocalProjection(
        y,
        shocks[:, 0] if n_shocks == 1 else shocks,
        controls,
        H=H,
        target_poly_order=target_poly_order,
        mask=mask,
    )
    reference_shocks, reference_controls = _reference_inputs(
        shocks, controls, partial_out
    )
    design = stacked_reference(
        y, reference_shocks, reference_controls, model.B, model.n_diff, mask
    )
    reference = _select_shocks(fit_reference(design, lam), n_shocks)
    result = model.fit(lam, partial_out=partial_out)
    assert_estimate_matches(result, reference)
    assert_allclose(result.lam, lam, rtol=0, atol=0)
    assert result.tune_trace.item() is None


def _full_design_gcv_reference(y, shocks, controls, B, n_diff, mask):
    """Dense fits with explicit, unpenalized horizon-specific controls.

    Compute every tuning statistic and derive the selected fit's covariance
    with nuisance coefficients retained. No production estimation helpers.
    """
    T, n_shocks = shocks.shape
    horizons, n_splines = B.shape
    included = np.ones(T, dtype=bool) if mask is None else ~mask
    rows, outcomes, dates = [], [], []
    for t in range(T):
        for h in range(min(horizons, T - t)):
            if not included[t : t + h + 1].all():
                continue
            rows.append(
                np.concatenate(
                    (
                        np.kron(shocks[t], B[h]),
                        np.kron(np.eye(horizons)[h], controls[t]),
                    )
                )
            )
            outcomes.append(y[t + h])
            dates.append(t)
    X, Y = np.array(rows), np.array(outcomes)
    n_spline_coef = n_splines * n_shocks
    # Positive penalties identify the overcomplete spline representation.
    assert np.linalg.matrix_rank(X[:, :n_spline_coef]) < n_spline_coef
    D = np.diff(np.eye(n_splines), n=n_diff, axis=0)
    P = np.zeros((X.shape[1], X.shape[1]))
    P[:n_spline_coef, :n_spline_coef] = np.kron(np.eye(n_shocks), D.T @ D)
    XX, XY = X.T @ X, X.T @ Y
    grid = len(Y) * 10 ** np.linspace(-3, 1, GRID_SIZE)
    trace, coefs = [], []
    for lam in grid:
        A = XX + lam * P
        coef = np.linalg.solve(A, XY)
        rss = np.sum((Y - X @ coef) ** 2)
        df = np.trace(np.linalg.solve(A, XX))
        trace.append(
            [rss, rss / (1 - df / len(Y)) ** 2, np.log(rss) + 2 * df / len(Y), df]
        )
        coefs.append(coef)
    trace = np.array(trace)
    selected = np.argmin(trace[:, 1])
    coef, lam = coefs[selected], grid[selected]
    scores = np.zeros((T, X.shape[1]))
    np.add.at(scores, dates, X * (Y - X @ coef)[:, None])
    bread = np.linalg.solve(XX + 0.1 * lam * P, np.eye(X.shape[1]))
    covariance = bread @ hac_meat(scores, horizons - 1) @ bread.T
    return SimpleNamespace(
        lam=lam,
        tune_trace=trace,
        coef=coef[:n_spline_coef].reshape(n_shocks, n_splines).T,
        vcov_nw=covariance[:n_spline_coef, :n_spline_coef],
    )


@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("n_shocks", [3])
@pytest.mark.parametrize("n_controls", [2])
@pytest.mark.parametrize("target_poly_order", [3])
@pytest.mark.parametrize("masked", [False, True], ids=["complete", "gaps"])
@pytest.mark.parametrize("partial_out", [True, False], ids=["partial", "joint"])
def test_gcv_matches_full_design(
    H, n_shocks, n_controls, target_poly_order, masked, partial_out
):
    y, shocks, controls, mask = simulated_data(
        T=80,
        n_shocks=n_shocks,
        n_controls=n_controls,
        seed=71,
        masked=masked,
    )
    model = SmoothLocalProjection(
        y,
        shocks,
        controls,
        H,
        target_poly_order=target_poly_order,
        mask=mask,
    )
    result = model.fit("gcv", lam_grid_size=GRID_SIZE, partial_out=partial_out)
    reference_shocks, reference_controls = _reference_inputs(
        shocks, controls, partial_out
    )
    full = _full_design_gcv_reference(
        y, reference_shocks, reference_controls, model.B, model.n_diff, mask
    )
    assert result.tune_trace.shape == (GRID_SIZE, 4)
    assert np.isfinite(result.tune_trace).all()
    assert_allclose(result.tune_trace, full.tune_trace, rtol=1e-6, atol=1e-6)
    assert_allclose(result.lam, full.lam, rtol=1e-12)
    assert_allclose(result.coef, full.coef[:, :n_shocks], rtol=1e-6, atol=1e-7)
    reference = fit_reference(
        stacked_reference(
            y, reference_shocks, reference_controls, model.B, model.n_diff, mask
        ),
        full.lam,
    )
    assert_allclose(reference.vcov_nw, full.vcov_nw, rtol=1e-6, atol=1e-8)
    assert_estimate_matches(result, _select_shocks(reference, n_shocks))


@pytest.mark.parametrize("H", HORIZONS)
@pytest.mark.parametrize("n_shocks", SHOCK_COUNTS)
@pytest.mark.parametrize("n_controls", CONTROL_COUNTS)
@pytest.mark.parametrize("target_poly_order", POLYNOMIAL_ORDERS)
@pytest.mark.parametrize("masked", [False, True], ids=["complete", "gaps"])
@pytest.mark.parametrize("lam", [10, "gcv", "cv"])
def test_joint_matches_promoted_shocks(
    H, n_shocks, n_controls, target_poly_order, masked, lam
):
    y, shocks, controls, mask = simulated_data(
        T=320 if H == 100 else 80,
        n_shocks=n_shocks,
        n_controls=n_controls,
        masked=masked,
    )
    options = dict(H=H, target_poly_order=target_poly_order, mask=mask)
    joint = SmoothLocalProjection(y, shocks, controls, **options).fit(
        lam,
        partial_out=False,
        lam_grid_size=GRID_SIZE,
    )
    promoted = SmoothLocalProjection(
        y,
        np.column_stack((shocks, controls)),
        np.empty((len(y), 0)),
        **options,
    ).fit(lam, partial_out=True, lam_grid_size=GRID_SIZE)
    assert_estimate_matches(joint, _select_shocks(promoted, n_shocks))
    assert_allclose(joint.lam, promoted.lam, rtol=1e-12, atol=0)
    # These fits have the same complete penalized design and residuals.
    for name in ("P", "_mask", "_Ys", "_us"):
        actual, expected = (
            np.asarray(getattr(joint, name)),
            np.asarray(getattr(promoted, name)),
        )
        assert actual.shape == expected.shape, name
        assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-8, equal_nan=False, err_msg=name
        )
    if lam in ("gcv", "cv"):
        expected_shape = (GRID_SIZE, 4) if lam == "gcv" else (GRID_SIZE,)
        assert joint.tune_trace.shape == promoted.tune_trace.shape == expected_shape
        assert np.isfinite(joint.tune_trace).all()
        assert np.isfinite(promoted.tune_trace).all()
        assert_allclose(joint.tune_trace, promoted.tune_trace, rtol=1e-6, atol=1e-6)
    else:
        assert joint.tune_trace.item() is promoted.tune_trace.item() is None


def _full_design_cv_reference(y, shocks, controls, B, n_diff, mask):
    """Five chronological folds per horizon, with dense training-only fits.

    Assign zero-based valid-observation ranks to five equal-width intervals.
    Every valid row is held out exactly once. Nuisance controls are explicit
    horizon-specific regressors, so their coefficients use only training rows.
    """
    T, n_shocks = shocks.shape
    horizons, n_splines = B.shape
    included = np.ones(T, dtype=bool) if mask is None else ~mask
    rows, outcomes, folds = [], [], []
    for h in range(horizons):
        times = [t for t in range(T - h) if included[t : t + h + 1].all()]
        assert len(times) >= 5
        for rank, t in enumerate(times):
            rows.append(
                np.concatenate(
                    (
                        np.kron(shocks[t], B[h]),
                        np.kron(np.eye(horizons)[h], controls[t]),
                    )
                )
            )
            outcomes.append(y[t + h])
            folds.append(rank * 5 // len(times))
    X, Y, folds = np.array(rows), np.array(outcomes), np.array(folds)
    assert np.all((folds >= 0) & (folds < 5))
    n_spline_coef = n_splines * n_shocks
    D = np.diff(np.eye(n_splines), n=n_diff, axis=0)
    P = np.zeros((X.shape[1], X.shape[1]))
    P[:n_spline_coef, :n_spline_coef] = np.kron(np.eye(n_shocks), D.T @ D)
    grid = len(Y) * 10 ** np.linspace(-3, 1, GRID_SIZE)
    fold_mse = []
    held_out_count = np.zeros(len(Y), dtype=int)
    for fold in range(5):
        test = folds == fold
        train = ~test
        assert test.any()
        held_out_count += test
        X_train, Y_train = X[train], Y[train]
        XX, XY = X_train.T @ X_train, X_train.T @ Y_train
        errors = []
        for lam in grid:
            coef = np.linalg.solve(XX + lam * P, XY)
            errors.append(np.mean((Y[test] - X[test] @ coef) ** 2))
        fold_mse.append(errors)
    assert np.all(held_out_count == 1)
    # Match the estimator's convention: equal weight per fold, not per row.
    trace = np.mean(fold_mse, axis=0)
    return SimpleNamespace(lam=grid[np.argmin(trace)], tune_trace=trace)


@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("n_shocks", [3])
@pytest.mark.parametrize("n_controls", [2])
@pytest.mark.parametrize("target_poly_order", [3])
@pytest.mark.parametrize("masked", [False, True], ids=["complete", "gaps"])
@pytest.mark.parametrize("partial_out", [True, False], ids=["partial", "joint"])
def test_cv_matches_full_design(
    H, n_shocks, n_controls, target_poly_order, masked, partial_out
):
    y, shocks, controls, mask = simulated_data(
        T=80,
        n_shocks=n_shocks,
        n_controls=n_controls,
        seed=217,
        masked=masked,
    )
    model = SmoothLocalProjection(
        y,
        shocks,
        controls,
        H,
        target_poly_order=target_poly_order,
        mask=mask,
    )
    result = model.fit("cv", lam_grid_size=GRID_SIZE, partial_out=partial_out)
    reference_shocks, reference_controls = _reference_inputs(
        shocks, controls, partial_out
    )
    full = _full_design_cv_reference(
        y,
        reference_shocks,
        reference_controls,
        model.B,
        model.n_diff,
        mask,
    )
    assert result.tune_trace.shape == (GRID_SIZE,)
    assert np.isfinite(result.tune_trace).all()
    assert_allclose(result.tune_trace, full.tune_trace, rtol=1e-6, atol=1e-6)
    assert_allclose(result.lam, full.lam, rtol=1e-12)
    reference = fit_reference(
        stacked_reference(
            y,
            reference_shocks,
            reference_controls,
            model.B,
            model.n_diff,
            mask,
        ),
        full.lam,
    )
    assert_estimate_matches(result, _select_shocks(reference, n_shocks))
