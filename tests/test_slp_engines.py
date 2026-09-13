"""End-to-end public API checks against an independent expanded-design oracle."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from smoothlp.slp import SmoothLocalProjection
from tests.slp_reference import (
    assert_estimate_matches,
    fit_reference,
    simulated_data,
    stacked_reference,
)


@pytest.mark.parametrize("H,n_shocks", [(100, 1), (100, 2), (100, 4), (19, 2)])
@pytest.mark.parametrize("masked", [False, True], ids=["complete", "gaps"])
@pytest.mark.parametrize("lam", [1.0, 1000.0], ids=["weak", "strong"])
def test_slp_matches_expanded_reference(H, n_shocks, masked, lam):
    y, shocks, controls, mask = simulated_data(n_shocks=n_shocks, masked=masked)
    shock_input = shocks[:, 0] if n_shocks == 1 else shocks
    model = SmoothLocalProjection(y, shock_input, controls, H=H, mask=mask)
    design = stacked_reference(y, shocks, controls, model.B, model.n_diff, mask)
    assert_estimate_matches(model.fit(lam), fit_reference(design, lam))


def test_slp_shock_permutation():
    y, shocks, controls, mask = simulated_data(n_shocks=4, masked=True)
    model = SmoothLocalProjection(y, shocks, controls, H=100, mask=mask)
    original = model.fit(10.0)
    order = np.array([2, 0, 3, 1])
    permuted = SmoothLocalProjection(
        y, shocks[:, order], controls, H=100, mask=mask
    ).fit(10.0)
    q = model.B.shape[1]
    indices = np.concatenate([np.arange(a * q, (a + 1) * q) for a in order])
    for actual, expected in (
        (permuted.coef, original.coef[indices]),
        (permuted.irf, original.irf[:, order]),
        (permuted.se, original.se[:, order]),
        (permuted.vcov_nw, original.vcov_nw[np.ix_(indices, indices)]),
    ):
        assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_slp_horizon_100():
    y, shocks, controls, mask = simulated_data(T=320, n_shocks=3, masked=True)
    model = SmoothLocalProjection(y, shocks, controls, H=100, mask=mask)
    assert model.B.shape[1] > 100
    design = stacked_reference(y, shocks, controls, model.B, model.n_diff, mask)
    assert_estimate_matches(model.fit(10.0), fit_reference(design, 10.0))
