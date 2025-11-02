import pytest
import pathlib
import polars as pl
import numpy as np
from numpy import testing as npt

from smoothlp.lp import LocalProjection


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
