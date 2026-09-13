import jax
import numpy as np
import pytest
from numpy import testing as npt

from smoothlp.utils import _build_stacked_jax, _build_stacked_np


def stacked_reference(endog, omitted, H, mask_type):
    """Direct date-based reference using public True-means-omit masks."""
    T = len(endog)
    outcomes = np.empty((T, H + 1), dtype=endog.dtype)
    included = np.zeros((T, H + 1), dtype=bool)
    for t in range(T):
        for h in range(H + 1):
            # The builders retain wrapped outcomes even where rows are unused.
            outcomes[t, h] = endog[(t + h) % T]
            if t + h >= T:
                continue
            if mask_type == "x":
                included[t, h] = not omitted[t]
            elif mask_type == "y":
                included[t, h] = not omitted[t + h]
            else:
                included[t, h] = not any(omitted[t : t + h + 1])
    return outcomes, included


@pytest.fixture(params=["np", "jax"])
def builder(request):
    if request.param == "np":
        return _build_stacked_np
    return jax.jit(_build_stacked_jax, static_argnames=["H", "mask_type"])


@pytest.mark.parametrize("mask_type", ["x", "y", "both"])
@pytest.mark.parametrize("H", [0, 1, 5, 11, 14])
@pytest.mark.parametrize(
    "omitted_times",
    [[], [4], [2, 6, 7, 8], [0, 11], list(range(12))],
    ids=["complete", "isolated", "isolated_and_chunk", "boundaries", "all_omitted"],
)
def test_build_stacked_matches_reference(builder, mask_type, H, omitted_times):
    endog = np.array([3, -2, 8, 1, 7, -5, 4, 9, 6, -1, 2, 10], dtype=float)
    omitted = np.zeros(len(endog), dtype=bool)
    omitted[omitted_times] = True
    inclusion_mask = (~omitted).astype(float)
    original_endog = endog.copy()
    original_mask = inclusion_mask.copy()

    expected_y, expected_mask = stacked_reference(endog, omitted, H, mask_type)
    actual_y, actual_mask = builder(endog, inclusion_mask, H, mask_type)

    npt.assert_array_equal(actual_y, expected_y)
    npt.assert_array_equal(actual_mask, expected_mask)
    npt.assert_array_equal(endog, original_endog)
    npt.assert_array_equal(inclusion_mask, original_mask)
