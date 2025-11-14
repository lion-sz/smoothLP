import pathlib

import pytest
from box import Box
import polars as pl
import numpy as np
from numpy import testing as npt
from jax import random as jrng

from smoothlp import slp
from simulator import simulation

RTOL = 1e-5


def make_simulation(simulation_name):
    conf_file = pathlib.Path(__file__).parent / "configs.yaml"
    conf = Box.from_yaml(filename=conf_file)
    assert simulation_name in conf, "Simulation not found."
    sim = simulation.Simulation(conf[simulation_name])
    return sim


@pytest.fixture(scope="module")
def sales():
    sales_file = pathlib.Path(__file__).parent / "test_data/spanish_sales/sales.parquet"
    sales = pl.read_parquet(sales_file)
    Y = sales["corrected"].to_numpy()
    shock = sales["MP_median"].to_numpy()
    X = sales.select(pl.lit(1.0), "New_cases", "StringencyIndex_Average").to_numpy()
    return Y, shock, X


def test_slp_no_mask():
    sim = make_simulation("zero_dense")
    Y, shock, irf = sim.get_data(0, 1, jrng.key(12))
    X = np.ones_like(Y)[:, np.newaxis]

    # First with a fixed lambda
    res_np = slp.SmoothLocalProjection(Y, shock, X, H=20).fit(10.0, engine="np")
    res_jax = slp.SmoothLocalProjection(Y, shock, X, H=20).fit(10.0, engine="jax")
    npt.assert_allclose(res_jax.irf, res_np.irf, rtol=RTOL)
    npt.assert_allclose(res_jax.coef, res_np.coef, rtol=10 * RTOL)
    npt.assert_allclose(res_jax.vcov_nw, res_np.vcov_nw, rtol=10 * RTOL)

    # Now tuning the lambda
    res_np = slp.SmoothLocalProjection(Y, shock, X, H=20).fit(None, engine="np")
    res_jax = slp.SmoothLocalProjection(Y, shock, X, H=20).fit(None, engine="jax")
    npt.assert_allclose(res_jax.lam, res_np.lam, rtol=1e-2)
