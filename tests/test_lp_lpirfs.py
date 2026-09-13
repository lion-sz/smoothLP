import pytest
import pathlib
import polars as pl
import numpy as np
from numpy import testing as npt

from smoothlp.lp import LocalProjection


@pytest.fixture(scope="module", params=[0, 4], ids=["intercept", "four_lags"])
def two_shock_data(request):
    rng = np.random.default_rng(812)
    n_obs, n_lags = 240, request.param
    shocks = rng.standard_normal((n_obs + n_lags, 2))
    noise = rng.standard_normal(n_obs + n_lags)
    y = np.zeros(n_obs + n_lags)
    for t in range(1, len(y)):
        y[t] = 0.6 * y[t - 1] + shocks[t, 0] - 0.4 * shocks[t, 1] + noise[t]
    controls = np.column_stack(
        [np.ones(n_obs)]
        + [y[n_lags - lag : len(y) - lag] for lag in range(1, n_lags + 1)]
    )
    return y[n_lags:], shocks[n_lags:], controls


def assert_lp_results_equal(actual, expected):
    npt.assert_allclose(actual.coef, expected.coef, rtol=1e-7, atol=1e-10)
    npt.assert_allclose(actual.vcov, expected.vcov, rtol=1e-7, atol=1e-10)
    assert actual.irf.shape == expected.irf.shape == (13, 2)
    assert actual.se.shape == expected.se.shape == (13, 2)
    npt.assert_allclose(actual.irf, expected.irf, rtol=1e-7, atol=1e-10)
    npt.assert_allclose(actual.se, expected.se, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("vcov", ["non-robust", "HC1", "HC3", "HAC"])
def test_two_shocks_engines(two_shock_data, vcov):
    y, shocks, controls = two_shock_data
    model = LocalProjection(y, shocks, controls, H=12)
    result_np = model.fit(vcov=vcov, engine="np")
    assert_lp_results_equal(model.fit(vcov=vcov, engine="jax"), result_np)
    # The statsmodels engine currently exposes only robust estimators.
    if vcov != "non-robust":
        assert_lp_results_equal(model.fit(vcov=vcov, engine="sm"), result_np)


@pytest.mark.parametrize("vcov", ["HC1", "HC3", "HAC"])
@pytest.mark.parametrize("mask_type", ["x", "y", "both"])
def test_two_shocks_masked_engines(two_shock_data, vcov, mask_type):
    y, shocks, controls = two_shock_data
    mask = np.zeros(len(y), dtype=bool)
    # Exercise isolated, adjacent, and sample-boundary exclusions.
    mask[[0, 1, 35, 36, 93, len(y) - 2, len(y) - 1]] = True
    original_mask = mask.copy()
    model = LocalProjection(y, shocks, controls, H=12, mask=mask, mask_type=mask_type)
    result_np = model.fit(vcov=vcov, engine="np")
    npt.assert_array_equal(mask, original_mask)
    result_sm = model.fit(vcov=vcov, engine="sm")
    npt.assert_array_equal(mask, original_mask)
    assert_lp_results_equal(result_np, result_sm)


@pytest.mark.parametrize("mask_type", ["x", "y", "both"])
def test_two_shocks_jax_rejects_mask(two_shock_data, mask_type):
    y, shocks, controls = two_shock_data
    mask = np.zeros(len(y), dtype=bool)
    mask[35] = True
    model = LocalProjection(y, shocks, controls, H=12, mask=mask, mask_type=mask_type)
    with pytest.raises(AssertionError, match="Mask is not yet supported"):
        model.fit(engine="jax")


@pytest.fixture(scope="module")
def interest_data_no_controls():
    p = pathlib.Path(__file__).parent
    dat = pl.read_csv(p / "test_data/interest_rules/interest_rules_var_data.csv")
    Y = dat["GDP_gap"].to_numpy()
    shock = dat["Infl"].to_numpy().flatten()
    X = np.ones((shock.shape[0], 1))
    return Y, shock, X


def get_lpirfs_data(file: str):
    p = pathlib.Path(__file__).parent
    dat = pl.read_csv(p / file)
    coef = dat[["mean"]].to_numpy().flatten()
    se = (dat["upper"] - dat["lower"]).to_numpy().flatten() / 2
    return coef, se


@pytest.fixture(scope="module")
def interest_results():
    return get_lpirfs_data("test_data/interest_rules/interest_rules_no_control.csv")


@pytest.fixture(scope="module")
def interest_results_nw():
    return get_lpirfs_data("test_data/interest_rules/interest_rules_no_control_nw.csv")


@pytest.fixture(scope="module")
def interest_results_lags4():
    return get_lpirfs_data(
        "test_data/interest_rules/interest_rules_no_control_lags4.csv"
    )


@pytest.fixture(scope="module")
def interest_results_nw_lags4():
    return get_lpirfs_data(
        "test_data/interest_rules/interest_rules_no_control_nw_lags4.csv"
    )


def test_no_control_non_robust(interest_data_no_controls, interest_results):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="non-robust", engine="np")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="non-robust", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_np.se, se)
    npt.assert_allclose(res_jax.coef, res_np.coef, rtol=1e-5)
    npt.assert_allclose(res_jax.vcov, res_np.vcov, rtol=1e-5)


def test_no_control_hc1(interest_data_no_controls, interest_results):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="HC1", engine="np")
    res_sm = LocalProjection(Y, shock, X, 12).fit(vcov="HC1", engine="sm")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="HC1", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_sm.coef, res_np.coef)
    npt.assert_allclose(res_sm.vcov, res_np.vcov)
    npt.assert_allclose(res_jax.coef, res_np.coef, rtol=1e-5)
    npt.assert_allclose(res_jax.vcov, res_np.vcov, rtol=1e-5)


def test_no_control_hc3(interest_data_no_controls, interest_results):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="HC3", engine="np")
    res_sm = LocalProjection(Y, shock, X, 12).fit(vcov="HC3", engine="sm")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="HC3", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_sm.coef, res_np.coef)
    npt.assert_allclose(res_sm.vcov, res_np.vcov)
    npt.assert_allclose(res_jax.coef, res_np.coef, rtol=1e-5)
    npt.assert_allclose(res_jax.vcov, res_np.vcov, rtol=1e-5)


def test_no_control_hac(interest_data_no_controls, interest_results_nw):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results_nw
    # Apply the small sample correction, since this is not done in lpirfs
    T = X.shape[0] - np.arange(13)
    se = se * np.sqrt(T / (T - 2))

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="HAC", engine="np")
    res_sm = LocalProjection(Y, shock, X, 12).fit(vcov="HAC", engine="sm")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="HAC", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_np.se, se)
    npt.assert_allclose(res_sm.coef, res_np.coef)
    npt.assert_allclose(res_np.vcov, res_sm.vcov)
    npt.assert_allclose(res_jax.coef, res_np.coef, rtol=1e-5)
    npt.assert_allclose(res_jax.vcov, res_np.vcov, rtol=1e-5)


def test_no_control_non_robust_l4(interest_data_no_controls, interest_results_lags4):
    Y, shock, X = interest_data_no_controls
    X = np.concat([X, np.stack([np.roll(Y, i) for i in range(1, 5)], axis=1)], axis=1)
    Y, shock, X = Y[4:], shock[4:], X[4:, :]
    coef, se = interest_results_lags4

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="non-robust", engine="np")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="non-robust", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_np.se, se)
    npt.assert_allclose(res_jax.coef, res_np.coef, atol=1e-5, rtol=1e-5)
    npt.assert_allclose(res_jax.vcov, res_np.vcov, atol=1e-5, rtol=1e-5)


def test_no_control_hc1_l4(interest_data_no_controls, interest_results_lags4):
    Y, shock, X = interest_data_no_controls
    X = np.concat([X, np.stack([np.roll(Y, i) for i in range(1, 5)], axis=1)], axis=1)
    Y, shock, X = Y[4:], shock[4:], X[4:, :]
    coef, se = interest_results_lags4

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="HC1", engine="np")
    res_sm = LocalProjection(Y, shock, X, 12).fit(vcov="HC1", engine="sm")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="HC1", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_sm.coef, res_np.coef)
    npt.assert_allclose(res_sm.vcov, res_np.vcov)
    npt.assert_allclose(res_jax.coef, res_np.coef, atol=1e-5, rtol=1e-5)
    npt.assert_allclose(res_jax.vcov, res_np.vcov, atol=1e-5, rtol=1e-5)


def test_no_control_hc3_l4(interest_data_no_controls, interest_results_lags4):
    Y, shock, X = interest_data_no_controls
    X = np.concat([X, np.stack([np.roll(Y, i) for i in range(1, 5)], axis=1)], axis=1)
    Y, shock, X = Y[4:], shock[4:], X[4:, :]
    coef, se = interest_results_lags4

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="HC3", engine="np")
    res_sm = LocalProjection(Y, shock, X, 12).fit(vcov="HC3", engine="sm")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="HC3", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_sm.coef, res_np.coef)
    npt.assert_allclose(res_sm.vcov, res_np.vcov)
    npt.assert_allclose(res_jax.coef, res_np.coef, atol=1e-5, rtol=1e-5)
    npt.assert_allclose(res_jax.vcov, res_np.vcov, atol=1e-5, rtol=1e-5)


def test_no_control_hac_l4(interest_data_no_controls, interest_results_nw_lags4):
    Y, shock, X = interest_data_no_controls
    X = np.concat([X, np.stack([np.roll(Y, i) for i in range(1, 5)], axis=1)], axis=1)
    Y, shock, X = Y[4:], shock[4:], X[4:, :]
    coef, se = interest_results_nw_lags4
    # Apply the small sample correction, since this is not done in lpirfs
    T = X.shape[0] - np.arange(13)
    se = se * np.sqrt(T / (T - 6))

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="HAC", engine="np")
    res_sm = LocalProjection(Y, shock, X, 12).fit(vcov="HAC", engine="sm")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="HAC", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_np.se, se)
    npt.assert_allclose(res_sm.coef, res_np.coef)
    npt.assert_allclose(res_np.vcov, res_sm.vcov)
    npt.assert_allclose(res_jax.coef, res_np.coef, atol=1e-5, rtol=1e-5)
    npt.assert_allclose(res_jax.vcov, res_np.vcov, atol=1e-5, rtol=1e-5)
