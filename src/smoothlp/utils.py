from typing import Literal

import numpy as np
from scipy import linalg as sla
import jax
from jax import numpy as jnp, scipy as jsp, Array, lax
from jaxtyping import Float


def bspline(p, knots, x):
    """B-Splines defined through the interior knots on unit interval.

    This function returns k B-Splines, where k is the size of the knots array.
    Note, that the knots must be equally spaced.

    The return matrix has dimension i x k, where i is the size of x
    and k the number of splines.
    """
    np.testing.assert_allclose(np.diff(knots), 1.0)
    k = knots.shape[0]
    # Add boundaries
    if p > 0:
        knots = np.concat(
            (
                np.arange(knots[0] - p, knots[0]),
                knots,
                np.arange(knots[-1] + 1, knots[-1] + p + 2),
            )
        )

    # Reshape
    knots = knots.reshape(-1, 1)
    x = x.reshape(1, -1)

    # Compute the 0th order bspline
    temp = np.roll(knots, -1)
    temp[-1] = knots[-1] + 1.0
    b = ((x >= knots) & (x < temp)).astype(float)

    # Compute the higher order bsplines
    for i in range(1, p + 1):
        x1 = (x - knots) / (np.roll(knots, -i) - knots) * b
        x2 = (
            (np.roll(knots, -(i + 1)) - x)
            / (np.roll(knots, -(i + 1)) - np.roll(knots, -1))
            * np.roll(b, -1, axis=0)
        )
        b = x1 + x2

    return b[p : (p + k), :].T


def demean(
    fe: list[np.ndarray],
    data: list[np.ndarray],
    maxiter: int = 10,
    norm: float = 1e-6,
    mask: np.ndarray = None,
):
    """Iteratively demean the provided data

    If a mask was provided, ignore masked elements during the mean computation step.
    """
    # Prep the arrays to compute the mean.
    sort_arr, g_id, g_pos, g_count, inv_map = [], [], [], [], []
    for f in fe:
        t_sort_arr = np.argsort(f)
        t_g_id, t_g_pos, t_g_count = np.unique(
            f[t_sort_arr], return_index=True, return_counts=True
        )
        if mask is not None:
            t_g_count = np.add.reduceat((~mask)[t_sort_arr], t_g_pos)
        _, t_inv_map = np.unique(f, return_inverse=True)
        sort_arr.append(t_sort_arr)
        g_id.append(t_g_id)
        g_pos.append(t_g_pos)
        g_count.append(t_g_count)
        inv_map.append(t_inv_map)

    iter_norm = 1e6
    iter = 0
    while iter_norm >= norm and iter < maxiter:
        iter_norm = 0.0
        iter += 1
        for j in range(len(fe)):
            for k in range(len(data)):
                temp = data[k]
                if mask is None:
                    temp_comp_mean = temp
                else:
                    if len(temp.shape) == 1:
                        temp_comp_mean = np.where(mask, 0.0, temp)
                    else:
                        temp_comp_mean = np.where(mask.reshape((-1, 1)), 0.0, temp)
                if len(temp.shape) == 1:
                    mean = (
                        np.add.reduceat(temp_comp_mean[sort_arr[j]], g_pos[j])
                        / g_count[j]
                    )
                    temp = temp - mean[inv_map[j]]
                    iter_norm = max(iter_norm, np.abs(mean).max())
                else:
                    for d in range(temp.shape[1]):
                        mean = (
                            np.add.reduceat(temp_comp_mean[sort_arr[j], d], g_pos[j])
                            / g_count[j]
                        )
                        temp[:, d] = temp[:, d] - mean[inv_map[j]]
                        iter_norm = max(iter_norm, np.abs(mean).max())
                data[k] = temp
        if len(fe) == 1:
            break

    print(f"Exited demeaning after {iter} iterations with norm {iter_norm}.")
    return data


def _lrv_np(Xu: Float[np.ndarray, "T k"], lags: int) -> Float[np.ndarray, "k k"]:
    cross = 0.5 * Xu
    temp = Xu.copy()
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        temp = np.roll(temp, -1, axis=0)
        temp[-1, :] = 0.0
        cross += w * temp
    cross = Xu.T @ cross
    lrv = cross + cross.T
    return lrv


def _lrv_jax(Xu: Float[Array, "T k"], lags: int) -> Float[Array, "k k"]:
    def accumulate(carry, lag):
        shifted, weighted = carry

        # Advance one time step, zero-padding instead of wrapping.
        shifted = jnp.roll(shifted, -1, axis=0).at[-1].set(0.0)
        weight = 1.0 - lag / (lags + 1)
        return (shifted, weighted + weight * shifted), None

    (_, weighted), _ = lax.scan(
        accumulate,
        (Xu, 0.5 * Xu),
        jnp.arange(1, lags + 1),
    )

    cross = Xu.T @ weighted
    lrv = cross + cross.T
    return lrv


def comp_lrv(Xu, lags: int):
    if isinstance(Xu, np.ndarray):
        lrv = _lrv_np(Xu, lags)
    elif isinstance(Xu, jnp.ndarray):
        lrv = _lrv_jax(Xu, lags)
    else:
        raise ValueError("Xu must be a numpy or jax array")
    return lrv


def np_vmap(f, in_axes=0, out_axes=0):
    def mapped(x):
        temp = np.moveaxis(x, in_axes, 0)
        return np.stack(
            [f(temp[_i, ...]) for _i in range(temp.shape[0])], axis=out_axes
        )

    return mapped


def _build_stacked_np(
    endog: Float[np.ndarray, " T"],
    mask: Float[np.ndarray, " T"],
    H: int,
    mask_type: str = "both",
) -> tuple[Float[np.ndarray, "T H+1"], Float[np.ndarray, "T H+1"]]:
    """Stack outcomes and binary inclusion masks (1 keeps, 0 omits).

    For 'both', every date from t through t+h must be included. Outcome
    values wrap beyond the sample, but their inclusion flags are always zero.
    """
    T = len(endog)
    Ys = np.empty((T, H + 1), dtype=endog.dtype)
    mask_s = np.empty((T, H + 1), dtype=mask.dtype)
    current_mask = mask.copy() if mask_type in ("both", "y") else np.ones_like(mask)
    for h in range(H + 1):
        Ys[:, h] = np.roll(endog, -h)
        mask_s[:, h] = current_mask
        next_mask = np.roll(current_mask, -1)
        if mask_type == "both":
            next_mask[current_mask == 0] = 0
        next_mask[-1] = 0
        current_mask = next_mask
    if mask_type in ("both", "x"):
        mask_s *= mask[:, np.newaxis]
    return Ys, mask_s


def _build_stacked_jax(
    endog: Float[Array, " T"],
    mask: Float[Array, " T"],
    H: int,
    mask_type: str = "both",
) -> tuple[Float[Array, "T H+1"], Float[Array, "T H+1"]]:
    """JAX equivalent of build_stacked_np; mask uses 1=include, 0=omit."""

    def fy(ty, _):
        typ = jnp.roll(ty, -1)
        return typ, ty

    Ys = lax.scan(fy, endog, length=H + 1)[1]
    Ys = Ys.swapaxes(0, 1)

    def fm(tm, _):
        tmp = jnp.roll(tm, -1)
        if mask_type == "both":
            tmp = jnp.where(tm == 0, 0, tmp)
        tmp = tmp.at[-1].set(0)
        return tmp, tm

    if mask_type in ("both", "y"):
        start = mask
    else:
        start = jnp.ones_like(mask)
    mask_s = lax.scan(fm, start, length=H + 1)[1]
    mask_s = mask_s.swapaxes(0, 1)
    if mask_type in ("both", "x"):
        mask_s = mask_s * mask[:, np.newaxis]

    return Ys, mask_s


def build_stacked(
    engine: Literal["jax", "np"],
    endog: Float[Array, " T"],
    mask: Float[Array, " T"],
    H: int,
    mask_type: str = "both",
) -> tuple[Float[Array, "T H+1"], Float[Array, "T H+1"]]:
    if engine == "jax":
        return _build_stacked_jax(endog, mask, H, mask_type)
    elif engine == "np":
        return _build_stacked_np(endog, mask, H, mask_type)
    else:
        raise ValueError(f"Can only use jax or np as engine, not '{engine}'.")


def _partial_out_jax(
    Ys: Float[Array, "T H+1"],
    shock: Float[Array, "T s"],
    exog: Float[Array, "T k"],
    mask_s: Float[Array, "T H+1"],
    leverage: bool,
    unmasked: bool = False,
):
    # Partial out the regressors
    XpX = jnp.einsum("ti,tj,th->ijh", exog, exog, mask_s)
    XpX_inv = jax.vmap(_invert_jax, in_axes=-1, out_axes=-1)(XpX)
    if leverage:
        X_leverage = jnp.einsum("ti,ijh,tj->th", exog, XpX_inv, exog) * mask_s
    else:
        X_leverage = None

    Ys_partial: Float[Array, "T H+1"] = jnp.einsum(
        "tj,jkh,qk,qh, qh->th",
        exog,
        XpX_inv,
        exog,
        mask_s,
        Ys,
        optimize="greedy",
    )
    Ys_r = Ys - Ys_partial

    shock_partial: Float[Array, "T H+1"] = jnp.einsum(
        "tj,jkh,qk,qh,qs->tsh",
        exog,
        XpX_inv,
        exog,
        mask_s,
        shock,
        optimize="greedy",
    )
    shock_r = shock[..., jnp.newaxis] - shock_partial

    if not unmasked:
        Ys_r = Ys_r * mask_s
        shock_r = shock_r * mask_s[:, jnp.newaxis, :]

    return Ys_r, shock_r, X_leverage


def _partial_out_np(
    Ys: Float[np.ndarray, "T H+1"],
    shock: Float[np.ndarray, "T s"],
    exog: Float[np.ndarray, "T k"],
    mask_s: Float[np.ndarray, "T H+1"],
    leverage: bool = False,
):
    """Return masked outcome/shock residuals and control leverage per horizon."""
    # XpX = np.einsum("ti,tj,th->ijh", exog, exog, mask_s)  # This does not use blas...
    XpX = np.stack(
        [exog.T @ (exog * mask_s[:, h, None]) for h in range(mask_s.shape[1])],
        axis=-1,
    )
    XpX_inv = np_vmap(np.linalg.inv, in_axes=-1, out_axes=-1)(XpX)
    if leverage:
        # X_leverage = np.einsum("ti,ijh,tj->th", exog, XpX_inv, exog, optimize=["einsum_path", (0, 1), (0, 1)]) * mask_s
        X_leverage = np.empty_like(mask_s, dtype=XpX_inv.dtype)

        for h in range(mask_s.shape[1]):
            A_h = np.ascontiguousarray(XpX_inv[:, :, h])
            XA = exog @ A_h
            X_leverage[:, h] = np.einsum("ti,ti->t", XA, exog) * mask_s[:, h]
    else:
        X_leverage = None

    Ys_partial = np.einsum(
        "tj,jkh,qk,qh,qh->th",
        exog,
        XpX_inv,
        exog,
        mask_s,
        Ys,
        optimize="greedy",
    )
    Ys_r_m = (Ys - Ys_partial) * mask_s

    shock_partial = np.einsum(
        "tj,jkh,qk,qh,qs->tsh",
        exog,
        XpX_inv,
        exog,
        mask_s,
        shock,
        optimize="greedy",
    )
    shock_r_m = (shock[..., np.newaxis] - shock_partial) * mask_s[:, np.newaxis, :]
    return Ys_r_m, shock_r_m, X_leverage


def partial_out(
    engine: Literal["jax", "np"],
    Ys,
    shock,
    exog,
    mask_s,
    leverage: bool,
    unmasked: bool = False,
):
    """Partial out controls using binary inclusion masks (1 keeps, 0 omits)."""
    if exog is None and engine in ("jax", "np"):
        assert unmasked is False, "Unmasked without exog not yet supported."
        lib = jnp if engine == "jax" else np
        return (
            Ys * mask_s,
            shock[:, :, None] * mask_s[:, None, :],
            lib.zeros_like(mask_s) if leverage else None,
        )
    if engine == "jax":
        return _partial_out_jax(Ys, shock, exog, mask_s, leverage, unmasked)
    elif engine == "np":
        return _partial_out_np(Ys, shock, exog, mask_s, leverage)
    else:
        raise ValueError(f"Can only use jax or np as engine, not '{engine}'.")


def _invert_np(X):
    U = sla.cholesky(X)
    U_inv = sla.solve_triangular(U, np.eye(X.shape[0]))
    return U_inv @ U_inv.T


def _invert_jax(X):
    U = jsp.linalg.cholesky(X)
    U_inv = jsp.linalg.solve_triangular(U, jnp.eye(X.shape[0]))
    return U_inv @ U_inv.T


def invert(engine: Literal["jax", "np"], X, in_axes: int | None = None):
    if engine == "jax":
        if in_axes is None:
            return _invert_jax(X)
        else:
            return jax.vmap(_invert_jax, in_axes=2, out_axes=2)(X)
    elif engine == "np":
        if in_axes is None:
            return _invert_np(X)
        else:
            if in_axes != len(X.shape) - 1:
                raise ValueError("Currently can only map along last axis")
            res = [_invert_np(X[..., h]) for h in range(X.shape[-1])]
            return np.stack(res, axis=in_axes)
    else:
        raise ValueError(f"Can only use jax or np as engine, not '{engine}'")
