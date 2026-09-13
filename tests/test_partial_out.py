import jax
import numpy as np
from numpy import testing as npt

from smoothlp.utils import build_stacked, partial_out


def test_partial_out_numpy_jax_agree():
    rng = np.random.default_rng(42)
    y = rng.normal(size=40)
    shocks = rng.normal(size=(40, 2))
    controls = np.column_stack((np.ones(40), rng.normal(size=(40, 3))))
    inclusion_mask = np.ones(40)
    inclusion_mask[7] = 0
    inclusion_mask[18:22] = 0
    Ys, mask_s = build_stacked("np", y, inclusion_mask, H=5)

    expected = partial_out("np", Ys, shocks, controls, mask_s, True)
    actual = jax.jit(partial_out, static_argnames=["engine", "leverage"])(
        "jax", Ys, shocks, controls, mask_s, True
    )
    for actual_array, expected_array in zip(actual, expected):
        npt.assert_allclose(actual_array, expected_array, rtol=1e-7, atol=1e-10)
