"""Independent float64 SLP reference: expand splines BEFORE QR residualization.

Do not reuse production stacking, residualization, penalty, tuning, or covariance
helpers. The supplied public B is shared: these tests target estimation rather
than spline construction. Binary masks exclude any interval containing an
omitted observation, including its endpoints.
"""

from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose


def simulated_data(T=320, n_shocks=2, n_controls=3, seed=812, masked=False):
    rng = np.random.default_rng(seed)
    controls = (
        np.column_stack((np.ones(T), rng.normal(size=(T, n_controls - 1))))
        if n_controls
        else np.empty((T, 0))
    )
    common = rng.normal(size=(T, 1))
    shocks = rng.normal(size=(T, n_shocks)) + 0.65 * common
    shocks += controls @ rng.normal(scale=0.4, size=(n_controls, n_shocks))
    innovations = rng.normal(size=T) * (0.5 + 0.3 * np.abs(common[:, 0]))
    errors = innovations.copy()
    for t in range(1, T):
        errors[t] += 0.55 * errors[t - 1]
    weights = np.linspace(0.3, 0.9, n_shocks) * (-1.0) ** np.arange(n_shocks)
    signal = shocks @ weights
    y = controls @ np.linspace(0.2, 0.7, n_controls) + signal + errors
    y[1:] += 0.4 * signal[:-1]
    mask = None
    if masked:
        mask = np.zeros(T, dtype=bool)
        # Retain long uninterrupted stretches for the H=100 reference checks.
        mask[11] = True
        mask[T // 3 : T // 3 + 5] = True
    return y, shocks, controls, mask


def stacked_reference(y, shocks, controls, B, n_diff, mask=None):
    """Invalid rows stay zero at their original times, including internal gaps."""
    T, s = shocks.shape
    if controls is None:
        controls = np.empty((T, 0))
    horizons, q = B.shape
    mask = np.ones(T, dtype=bool) if mask is None else ~np.asarray(mask, dtype=bool)
    X = np.zeros((T, horizons, s * q))
    Y = np.zeros((T, horizons))
    n = 0
    for h in range(horizons):
        times = np.arange(T - h)
        times = times[np.array([mask[t : t + h + 1].all() for t in times], dtype=bool)]
        if not len(times):
            continue
        W = controls[times]
        if W.shape[1]:
            assert np.linalg.matrix_rank(W) == W.shape[1]
        Q, _ = np.linalg.qr(W, mode="reduced")
        Z = np.array([np.kron(shocks[t], B[h]) for t in times])
        outcome = y[times + h]
        X[times, h] = Z - Q @ (Q.T @ Z)
        Y[times, h] = outcome - Q @ (Q.T @ outcome)
        n += len(times)
    difference = np.diff(np.eye(q), n=n_diff, axis=0)
    P = np.kron(np.eye(s), difference.T @ difference)
    flat = X.reshape(-1, s * q)
    return SimpleNamespace(
        X=X,
        Y=Y,
        B=B,
        P=P,
        gram=flat.T @ flat,
        rhs=flat.T @ Y.ravel(),
        n=n,
        n_controls=controls.shape[1] * horizons,
    )


def hac_meat(scores, H):
    """Time-clustered scores, Bartlett lags 0..H; no sample-size rescaling."""
    meat = scores.T @ scores
    for lag in range(1, H + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (H + 1)) * (cross + cross.T)
    return meat


def fit_reference(design, lam):
    X, Y, B, P = design.X, design.Y, design.B, design.P
    coef = np.linalg.solve(design.gram + lam * P, design.rhs)
    errors = Y - (X @ coef)
    # Aggregate horizons within original time BEFORE calculating HAC lags.
    scores = np.sum(X * errors[..., None], axis=1)
    meat = hac_meat(scores, B.shape[0] - 1)
    # Preserve SLP's inference convention, distinct from fitting lambda.
    bread = np.linalg.solve(design.gram + 0.1 * lam * P, np.eye(len(coef)))
    vcov = bread @ meat @ bread.T
    s, q = len(coef) // B.shape[1], B.shape[1]
    irf = np.column_stack([B @ coef[a * q : (a + 1) * q] for a in range(s)])
    se = np.column_stack(
        [
            np.sqrt(np.diag(B @ vcov[a * q : (a + 1) * q, a * q : (a + 1) * q] @ B.T))
            for a in range(s)
        ]
    )
    coef = coef.reshape(s, q).T
    return SimpleNamespace(coef=coef, irf=irf, vcov_nw=vcov, se=se, lam=lam)


def assert_estimate_matches(actual, expected):
    for name in ("coef", "irf", "vcov_nw", "se"):
        value, target = np.asarray(getattr(actual, name)), getattr(expected, name)
        assert value.shape == target.shape, name
        assert np.isfinite(value).all(), name
        assert_allclose(value, target, rtol=1e-6, atol=1e-8, err_msg=name)
