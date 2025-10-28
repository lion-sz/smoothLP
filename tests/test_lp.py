import pytest
import pathlib
import polars as pl
import numpy as np
from numpy import testing as npt

from src.lp import LocalProjection


@pytest.fixture(scope="module")
def interest_data_no_controls():
    p = pathlib.Path(__file__).parent
    dat = pl.read_csv(p / "test_data/interest_rules_var_data.csv")
    Y = dat["GDP_gap"].to_numpy()
    shock = dat["Infl"].to_numpy()
    X = np.ones((shock.shape[0], 1))
    return Y, shock, X


@pytest.fixture(scope="module")
def interest_results():
    p = pathlib.Path(__file__).parent
    dat = pl.read_csv(p / "test_data/interest_rules_no_control.csv")
    coef = dat[["mean"]].to_numpy().flatten()
    se = (dat["upper"] - dat["lower"]).to_numpy().flatten() / 2
    return coef, se


@pytest.fixture(scope="module")
def interest_results_nw():
    p = pathlib.Path(__file__).parent
    dat = pl.read_csv(p / "test_data/interest_rules_no_control_nw.csv")
    coef = dat[["mean"]].to_numpy().flatten()
    se = (dat["upper"] - dat["lower"]).to_numpy().flatten() / 2
    return coef, se


def test_no_control_non_robust(interest_data_no_controls, interest_results):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results

    res_np = LocalProjection(Y, X, shock, 12).fit(vcov="non-robust", engine="np")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_np.se, se)


def test_no_control_hc1(interest_data_no_controls, interest_results):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results

    res_np = LocalProjection(Y, X, shock, 12).fit(vcov="HC1", engine="np")
    res_sm = LocalProjection(Y, X, shock, 12).fit(vcov="HC1", engine="sm")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_sm.coefs, res_np.coefs)
    npt.assert_allclose(res_sm.vcov, res_np.vcov)


def test_no_control_hc3(interest_data_no_controls, interest_results):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results

    res_np = LocalProjection(Y, X, shock, 12).fit(vcov="HC3", engine="np")
    res_sm = LocalProjection(Y, X, shock, 12).fit(vcov="HC3", engine="sm")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_sm.coefs, res_np.coefs)
    npt.assert_allclose(res_sm.vcov, res_np.vcov)


def test_no_control_hac(interest_data_no_controls, interest_results_nw):
    Y, shock, X = interest_data_no_controls
    coef, se = interest_results_nw
    # Apply the small sample correction, since this is not done in lpirfs
    T = X.shape[0] - np.arange(13)
    se = se * np.sqrt(T / (T - 2))

    res_np = LocalProjection(Y, X, shock, 12).fit(vcov="HAC", engine="np")
    res_sm = LocalProjection(Y, X, shock, 12).fit(vcov="HAC", engine="sm")

    npt.assert_allclose(res_np.irf, coef)
    npt.assert_allclose(res_np.se, se)
    npt.assert_allclose(res_sm.coefs, res_np.coefs)
    npt.assert_allclose(res_np.vcov, res_sm.vcov)
