import pytest
import pathlib
import polars as pl
import numpy as np
from numpy import testing as npt
from statsmodels import api as sm
from statsmodels.stats.sandwich_covariance import cov_hac_simple

from smoothlp.lp import LocalProjection


def assert_lp_shocks_equal(actual, expected, *, rtol=1e-7, atol=1e-10):
    n_shocks = expected.model.shock.shape[1]
    n_horizons = expected.model.H + 1
    assert actual.coef.shape == expected.coef.shape == (n_horizons, n_shocks)
    assert actual.vcov.shape == expected.vcov.shape == (n_horizons, n_shocks, n_shocks)
    npt.assert_allclose(actual.coef, expected.coef, rtol=rtol, atol=atol)
    npt.assert_allclose(actual.vcov, expected.vcov, rtol=rtol, atol=atol)


# Compare every supported engine/covariance pair to independent statsmodels OLS.
# The sm engine does not expose non-robust covariance; direct OLS still provides
# the reference for that configuration of the NumPy and JAX engines.
_ENGINE_VCOV = [
    (engine, vcov)
    for engine in ("np", "jax", "sm")
    for vcov in ("non-robust", "HC1", "HC3", "HAC")
    if (engine, vcov) != ("sm", "non-robust")
]


def statsmodels_reference(model, vcov, *, preserve_gaps):
    """Fit full regressions without using smoothlp stacking or masking helpers."""
    shocks = model.shock
    design = shocks if model.exog is None else np.column_stack((shocks, model.exog))
    n_obs, n_params = design.shape
    n_shocks = shocks.shape[1]
    coefficients, covariances = [], []
    for h in range(model.H + 1):
        times = np.arange(n_obs - h)
        keep = np.ones(len(times), dtype=bool)
        if model.mask is not None:
            if model.mask_type == "x":
                keep = ~model.mask[times]
            elif model.mask_type == "y":
                keep = ~model.mask[times + h]
            else:
                keep = np.array([not model.mask[t : t + h + 1].any() for t in times])
        outcome = model.endog[times + h]
        if model.diff is not None:
            outcome = outcome - model.diff[times]
        fit = sm.OLS(outcome[keep], design[times[keep]]).fit()
        if vcov == "non-robust":
            covariance = fit.cov_params()
        elif vcov in ("HC1", "HC3"):
            covariance = fit.get_robustcov_results(cov_type=vcov).cov_params()
        elif preserve_gaps and model.mask is not None:
            # Preserve original time distances, using full-design OLS scores.
            scores = np.zeros((len(times), n_params))
            scores[keep] = fit.model.exog * fit.resid[:, None]
            covariance = cov_hac_simple(
                (scores, fit.normalized_cov_params),
                nlags=h + 1,
                use_correction=False,
            )
            covariance *= fit.nobs / (fit.nobs - n_params)
        else:
            covariance = cov_hac_simple(fit, nlags=h + 1)
        coefficients.append(fit.params[:n_shocks])
        covariances.append(covariance[:n_shocks, :n_shocks])
    return np.array(coefficients), np.array(covariances)


@pytest.fixture(scope="module", params=["none", "intercept", "four_lags"])
def reference_data(request):
    rng = np.random.default_rng(812)
    n_obs = 96
    shocks = rng.standard_normal((n_obs + 4, 2))
    shocks[:, 1] += 0.5 * shocks[:, 0]
    noise = rng.standard_normal(n_obs + 4) * np.linspace(0.5, 2, n_obs + 4)
    y = np.zeros(n_obs + 4)
    for t in range(1, len(y)):
        y[t] = 0.6 * y[t - 1] + shocks[t, 0] - 0.4 * shocks[t, 1] + noise[t]
    controls = None
    if request.param == "intercept":
        controls = np.ones((n_obs, 1))
    elif request.param == "four_lags":
        controls = np.column_stack(
            [np.ones(n_obs)] + [y[4 - lag : len(y) - lag] for lag in range(1, 5)]
        )
    return y[4:], shocks[4:], controls, y[3:-1].copy()


@pytest.mark.parametrize("engine,vcov", _ENGINE_VCOV)
@pytest.mark.parametrize("shock_shape", ["1d", "one_column", "two_columns"])
@pytest.mark.parametrize("use_diff", [False, True], ids=["levels", "differences"])
@pytest.mark.parametrize("mask_type", [None, "x", "y", "both"])
def test_configurations_against_statsmodels(
    reference_data,
    engine,
    vcov,
    shock_shape,
    use_diff,
    mask_type,
):
    y, shocks, controls, baseline = reference_data
    shock = {"1d": shocks[:, 0], "one_column": shocks[:, :1], "two_columns": shocks}[
        shock_shape
    ]
    mask = None
    if mask_type is not None:
        mask = np.zeros(len(y), dtype=bool)
        mask[[0, 19, len(y) - 1]] = True
        mask[40:50] = True
    model = LocalProjection(
        y,
        shock,
        controls,
        H=5,
        mask=mask,
        mask_type=mask_type or "both",
        diff=baseline if use_diff else None,
    )
    originals = [
        None if x is None else x.copy() for x in (y, shock, controls, mask, model.diff)
    ]
    expected_coef, expected_vcov = statsmodels_reference(
        model,
        vcov,
        preserve_gaps=engine != "sm",
    )
    actual = model.fit(vcov=vcov, engine=engine)
    assert actual.coef.shape == expected_coef.shape
    assert actual.vcov.shape == expected_vcov.shape
    npt.assert_allclose(actual.coef, expected_coef, rtol=1e-7, atol=1e-10)
    npt.assert_allclose(actual.vcov, expected_vcov, rtol=1e-7, atol=1e-10)
    expected_se = np.sqrt(np.diagonal(expected_vcov, axis1=1, axis2=2))
    expected_irf = expected_coef
    if model.shock.shape[1] == 1:
        expected_irf, expected_se = expected_irf[:, 0], expected_se[:, 0]
    assert actual.irf.shape == expected_irf.shape
    assert actual.se.shape == expected_se.shape
    npt.assert_allclose(actual.irf, expected_irf, rtol=1e-7, atol=1e-10)
    npt.assert_allclose(actual.se, expected_se, rtol=1e-7, atol=1e-10)
    for value, original in zip((y, shock, controls, mask, model.diff), originals):
        if original is not None:
            npt.assert_array_equal(value, original)


@pytest.mark.parametrize("engine,vcov", _ENGINE_VCOV)
@pytest.mark.parametrize("mask_type", ["x", "y", "both"])
@pytest.mark.parametrize("horizon", [0, 5])
def test_empty_mask(reference_data, engine, vcov, mask_type, horizon):
    y, shocks, controls, _ = reference_data
    model = LocalProjection(
        y,
        shocks,
        controls,
        H=horizon,
        mask=np.zeros(len(y), dtype=bool),
        mask_type=mask_type,
    )
    expected_coef, expected_vcov = statsmodels_reference(
        model, vcov, preserve_gaps=True
    )
    actual = model.fit(vcov=vcov, engine=engine)
    assert actual.coef.shape == (horizon + 1, 2)
    assert actual.vcov.shape == (horizon + 1, 2, 2)
    npt.assert_allclose(actual.coef, expected_coef, rtol=1e-7, atol=1e-10)
    npt.assert_allclose(actual.vcov, expected_vcov, rtol=1e-7, atol=1e-10)


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
    assert_lp_shocks_equal(res_jax, res_np, rtol=1e-5, atol=0)


def test_no_control_hc1(interest_data_no_controls, interest_results):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="HC1", engine="np")
    res_sm = LocalProjection(Y, shock, X, 12).fit(vcov="HC1", engine="sm")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="HC1", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_sm.coef, res_np.coef)
    npt.assert_allclose(res_sm.vcov, res_np.vcov)
    assert_lp_shocks_equal(res_jax, res_np, rtol=1e-5, atol=0)


def test_no_control_hc3(interest_data_no_controls, interest_results):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="HC3", engine="np")
    res_sm = LocalProjection(Y, shock, X, 12).fit(vcov="HC3", engine="sm")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="HC3", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_sm.coef, res_np.coef)
    npt.assert_allclose(res_sm.vcov, res_np.vcov)
    assert_lp_shocks_equal(res_jax, res_np, rtol=1e-5, atol=0)


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
    assert_lp_shocks_equal(res_jax, res_np, rtol=1e-5, atol=0)


def test_no_control_non_robust_l4(interest_data_no_controls, interest_results_lags4):
    Y, shock, X = interest_data_no_controls
    X = np.concat([X, np.stack([np.roll(Y, i) for i in range(1, 5)], axis=1)], axis=1)
    Y, shock, X = Y[4:], shock[4:], X[4:, :]
    coef, se = interest_results_lags4

    res_np = LocalProjection(Y, shock, X, 12).fit(vcov="non-robust", engine="np")
    res_jax = LocalProjection(Y, shock, X, 12).fit(vcov="non-robust", engine="jax")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_np.se, se)
    assert_lp_shocks_equal(res_jax, res_np, atol=1e-5, rtol=1e-5)


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
    assert_lp_shocks_equal(res_jax, res_np, atol=1e-5, rtol=1e-5)


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
    assert_lp_shocks_equal(res_jax, res_np, atol=1e-5, rtol=1e-5)


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
    assert_lp_shocks_equal(res_jax, res_np, atol=1e-5, rtol=1e-5)
