"""Compare partial_out settings with the same jointly penalized regressors.

Controls supplied with partial_out=False become additional shocks with
partial_out=True. Compare estimates and covariance for the original shocks.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from smoothlp.slp import SmoothLocalProjection


@pytest.fixture(params=[0, 2], ids=["two-shocks", "two-shocks-two-controls"])
def simulated_data(request):
    """Generate a reproducible sample for each test, with correlated regressors."""
    n_controls = request.param
    n_regressors = 2 + n_controls
    T = 90
    rng = np.random.default_rng(418)
    common = rng.normal(size=(T, 1))
    regressors = (
        rng.normal(size=(T, n_regressors))
        + common @ np.array([[0.7, -0.4, 0.25, 0.5]])[:, :n_regressors]
    )
    errors = rng.normal(scale=0.5, size=T)
    for t in range(1, T):
        errors[t] += 0.35 * errors[t - 1]
    endog = regressors @ np.array([0.8, -0.6, 0.35, 0.2])[:n_regressors] + errors
    endog[1:] += 0.25 * regressors[:-1, 0]
    return endog, regressors[:, :2], regressors[:, 2:]


@pytest.fixture(
    params=[False, True],
    ids=["complete", "gaps"],
)
def mask(simulated_data, request):
    if not request.param:
        return None
    omission_mask = np.zeros(simulated_data[0].size, dtype=bool)
    omission_mask[[0, 11, -1]] = True
    omission_mask[42:46] = True
    return omission_mask


@pytest.fixture(
    params=[1, 2, 3],
    ids=["target-poly-order-1", "target-poly-order-2", "target-poly-order-3"],
)
def models(simulated_data, mask, request):
    endog, shocks, controls = simulated_data
    options = {"H": 5, "target_poly_order": request.param, "mask": mask}
    joint_model = SmoothLocalProjection(endog, shocks, controls, **options)
    partial_model = SmoothLocalProjection(
        endog,
        np.column_stack((shocks, controls)),
        np.empty((endog.size, 0)),
        **options,
    )
    return joint_model, partial_model


def _assert_shock_estimates_match(joint_result, partial_result):
    """Select original shocks, including their cross-shock covariance blocks."""
    n_splines = joint_result.model.B.shape[1]
    n_shocks = joint_result.model.shock.shape[1]
    n_coef = n_splines * n_shocks
    assert joint_result.coef.shape == (n_splines, n_shocks)
    assert partial_result.coef.shape == (n_splines, partial_result.model.shock.shape[1])
    expected = {
        "coef": partial_result.coef[:, :n_shocks],
        "irf": partial_result.irf[:, :n_shocks],
        "vcov_nw": partial_result.vcov_nw[:n_coef, :n_coef],
        "se": partial_result.se[:, :n_shocks],
    }
    for name, target in expected.items():
        actual = np.asarray(getattr(joint_result, name))
        assert actual.shape == target.shape, name
        assert np.isfinite(actual).all(), name
        assert np.isfinite(target).all(), name
        assert_allclose(
            actual, target, rtol=1e-6, atol=1e-8, equal_nan=False, err_msg=name
        )


@pytest.mark.parametrize("lam", [1.0, 10.0, 1000.0], ids=["weak", "medium", "strong"])
def test_fixed_lambda_estimates_match(models, lam):
    joint_model, partial_model = models
    joint_result = joint_model.fit(lam=lam, partial_out=False)
    partial_result = partial_model.fit(lam=lam, partial_out=True)

    _assert_shock_estimates_match(joint_result, partial_result)


def test_gcv_lambda_and_estimates_match(models):
    joint_model, partial_model = models
    tuning_options = {
        "lam": "gcv",
        "lam_grid_lower": -3.0,
        "lam_grid_upper": 1.0,
        "lam_grid_size": 31,
    }
    joint_result = joint_model.fit(partial_out=False, **tuning_options)
    partial_result = partial_model.fit(partial_out=True, **tuning_options)

    # Both models fit the same penalized design and should select the same grid point.
    assert_allclose(
        joint_result.lam, partial_result.lam, rtol=1e-12, atol=0.0, equal_nan=False
    )
    _assert_shock_estimates_match(joint_result, partial_result)


def test_cv_rejected_without_partial_out(simulated_data):
    endog, shocks, controls = simulated_data
    model = SmoothLocalProjection(endog, shocks, controls, H=5)
    with pytest.raises(AssertionError, match="cv is not yet supported"):
        model.fit(lam="cv", partial_out=False)


@pytest.mark.parametrize("p", [0, 1, 2, 4, 5])
def test_non_cubic_splines_rejected_without_partial_out(simulated_data, p):
    endog, shocks, controls = simulated_data
    model = SmoothLocalProjection(endog, shocks, controls, H=5, p=p)
    with pytest.raises(AssertionError, match="Non-cubic splines"):
        model.fit(lam=10.0, partial_out=False)
