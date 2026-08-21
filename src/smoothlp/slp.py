from dataclasses import dataclass
from functools import cached_property, partial
from datetime import datetime

import jax
import numpy as np
from jax import lax, Array
from jax import numpy as jnp, scipy as jsp
from jaxtyping import Float

from smoothlp.models import AbstractLocalProjection
from smoothlp.utils import bspline, demean, invert_jax


def build_stacked(
    endog: Float[Array, "T"],
    mask: Float[Array, "T"],
    H: int,
    mask_type: str = "both",
) -> tuple[Float[Array, "T H+1"], Float[Array, "T H+1"]]:
    def fy(ty, _):
        typ = jnp.roll(ty, -1)
        return typ, ty

    Ys = lax.scan(fy, endog, length=H + 1)[1]
    Ys = Ys.swapaxes(0, 1)

    def fm(tm, _):
        tmp = jnp.roll(tm, -1)
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


@dataclass
class SmoothLocalProjection(AbstractLocalProjection):
    """Run a smoothed local projection"""

    fe: tuple[np.ndarray] | tuple

    p: int
    n_diff: int

    mask_any_prev_h: bool

    def __init__(
        self,
        endog: np.ndarray,
        shock: np.ndarray,
        exog: np.ndarray | None,
        H: int,
        p: int = 3,
        target_poly_order: int = 2,
        fe: tuple[np.ndarray] | np.ndarray | None = None,
        mask: np.ndarray | None = None,
        mask_type: str = "both",
        mask_any_prev_h: bool = True,
    ):
        super().__init__(endog, shock, exog, H, mask, mask_type)
        if fe is None:
            self.fe = tuple()
        else:
            if not isinstance(fe, (list, tuple)):
                self.fe = (fe,)
            else:
                self.fe = fe
        self.p = p
        self.n_diff = target_poly_order + 1

        self.mask_any_prev_h = mask_any_prev_h

        self._Xs = None
        self._Ys = None
        self._omit = None
        self._us = None

    @property
    def knots(self):
        return np.arange(-2, self.H - 1).astype(np.float64)

    @property
    def B(self):
        B = bspline(self.p + 1, self.knots, np.arange(self.H + 1))
        # B = B[:, :-2]
        B = B / np.sum(B, axis=1)[:, np.newaxis]
        return B
        # TODO: Potential issue here.
        #   I believe there is a strange interaction between penalty and the splines
        #   hidden here. Specifically, If I use too few splines, then the sums
        #   over all splines are not 1 for each horizon. Then the coefficients scale,
        #   to account for this. But since the penalty applies to the coefficients,
        #   if is effectively not scale invariant.
        #   But when I use more than H+1 splines, I run into singularities.
        #   Therefore I rescale the splines.
        #   Note: replaced normalization with boundary trimming — drop p-1 columns
        #   from each end so all retained splines have complete support on [0, H],
        #   avoiding the scale interaction with the penalty (matches Julia).

        # New Version:
        # knots = np.arange(-(self.p + 1), self.H + self.p - 1).astype(np.float64)
        # B = bspline(self.p + 1, knots, np.arange(self.H + 1))
        # B = B[:, self.p: B.shape[1] - self.p - 2]
        # print(B.shape)

    def fit(
        self,
        lam: float | None,
        lam_grid_lower: float = -3.0,
        lam_grid_upper: float = 4.0,
        lam_grid_size: int = 1_000,
    ):
        # First create the splines. Indexed by horizon x basis function

        if len(self.fe) > 0:
            res = demean(self.fe, [self.endog, self.shock, self.exog], mask=self.mask)
            self.endog, self.shock, self.exog = res

        params = {
            "mask_type": "both",
            "mask_any_prev_h": self.mask_any_prev_h,
            "H": self.H,
            "n_diff": self.n_diff,
            "lam": lam,
            "lam_grid_lower": lam_grid_lower,
            "lam_grid_upper": lam_grid_upper,
            "lam_grid_size": lam_grid_size,
        }
        engine_fn = jax.jit(partial(SmoothLocalProjection._fit, **params))

        s = datetime.now()
        res = jax.block_until_ready(
            engine_fn(self.B, self.endog, self.shock, self.exog, self.mask)
        )
        X, Y, P, mask, u, theta_hat, vcov_nw, lam, tune_trace = (
            np.array(r) for r in res
        )
        print(f"Runtime of {(datetime.now() - s).total_seconds()}s")

        return SmoothLocalProjectionResult(
            self,
            self.knots,
            P,
            lam,
            theta_hat,
            vcov_nw,
            tune_trace,
            _Xs=X,
            _Ys=Y,
            _mask=mask,
            _us=u,
        )

    @staticmethod
    def _fit(
        B,
        endog,
        shock,
        exog,
        mask,
        mask_type: str,
        mask_any_prev_h: bool,
        H: int,
        n_diff: int,
        lam: float | None,
        lam_grid_lower: float,
        lam_grid_upper: float,
        lam_grid_size: int,
    ):
        # Build the stacked arrays
        T, k = exog.shape
        n_shocks = shock.shape[1]
        n_splines = B.shape[1]

        Xs = exog
        if mask is None:
            mask = jnp.ones(T)
        Ys, mask_s = build_stacked(endog, mask, H, mask_type)
        N = jnp.sum(mask_s)

        # Partial out the exogenous variables
        XpX = jnp.einsum("ti,tj,th->ijh", Xs, Xs, mask_s)
        XpX_inv = jax.vmap(invert_jax, in_axes=-1, out_axes=-1)(XpX)

        Ys_partial = jnp.einsum(
            "tj,jkh,qk,qh, qh->th", Xs, XpX_inv, Xs, mask_s, Ys, optimize="greedy"
        )
        Ys_r = Ys - Ys_partial
        Ys_r_m = Ys_r * mask_s

        Bs = jnp.einsum("hk,ts->thsk", B, shock)
        Bs = Bs.reshape(T, H + 1, n_splines * shock.shape[1])
        Bs_partial = jnp.einsum(
            "ti, ikh, qk, qh, qhj -> thj",
            Xs,
            XpX_inv,
            Xs,
            mask_s,
            Bs,
            optimize="greedy",
        )
        Bs_r_m = Bs - Bs_partial
        Bs_r_m = Bs_r_m * mask_s[..., jnp.newaxis]

        Y_r_m = Ys_r_m.reshape(T * (H + 1))
        B_r_m = Bs_r_m.reshape(T * (H + 1), n_splines)

        # Build the penalty matrix
        P = jnp.zeros((n_splines * n_shocks, n_splines * n_shocks))
        D = jnp.eye(n_splines)
        for i in range(n_diff):
            D = jnp.diff(D, axis=0)
        temp = D.T @ D
        for i in range(n_shocks):
            P = P.at[
                (i * n_splines) : (i + 1) * n_splines,
                (i * n_splines) : (i + 1) * n_splines,
            ].set(temp)
        if lam is None:
            lam, tune_trace = SmoothLocalProjection._tune_lambda(
                B_r_m,
                Y_r_m,
                P,
                N,
                n_splines * n_shocks,
                lam_grid_lower,
                lam_grid_upper,
                lam_grid_size,
            )
        else:
            tune_trace = None

        XtX_lam = B_r_m.T @ B_r_m + lam * P
        XtX_lam_inv = jnp.linalg.pinv(XtX_lam)
        theta_hat = XtX_lam_inv @ (B_r_m.T @ Y_r_m)
        u = Ys_r - jnp.einsum("thk,k->th", Bs_r_m, theta_hat)
        u = u

        # Compute the covariance matrix for the splines.
        Xu = jnp.einsum("thk,th->tk", Bs_r_m, u)
        Xu_past = Xu
        d_vcov = Bs_r_m.shape[2]

        def comp_gamma(carry, l):
            Xu_past, lrv = carry
            gamma = Xu.T @ Xu_past
            Xu_past = jnp.roll(Xu_past.at[0, ...].set(0.0), -1, axis=0)
            lrv = lrv + (1 - l / (H + 1)) * (gamma + gamma.T * (l > 0))
            return (Xu_past, lrv), gamma

        (_, lrv), gammas = lax.scan(
            comp_gamma, (Xu_past, jnp.zeros((d_vcov, d_vcov))), np.arange(H + 1)
        )

        XpX = jnp.einsum("thi,thj->ij", Bs_r_m, Bs_r_m) + 0.1 * lam * P
        XpX_inv = jnp.linalg.pinv(XpX)
        vcov_nw = XpX_inv @ lrv @ XpX_inv

        return Bs_r_m, Ys, P, mask_s, u, theta_hat, vcov_nw, lam, tune_trace

    @staticmethod
    def _tune_lambda(
        X,
        y,
        P,
        n: int,
        k_shock: int,
        lam_grid_lower: float,
        lam_grid_upper: float,
        lam_grid_size: int,
    ):
        """Tune lambda using Demmler-Reinsch Orthogonalization

        This procedure was adopted from the LocalProjections Julia package.
        See Rupper (2003), appendix B.1.1 for a reference.
        """

        # Set up the orthogonalization
        XpX = X.T @ X + 1e-9 * P
        R = jsp.linalg.cholesky(XpX, lower=False)
        Rinv = jsp.linalg.solve_triangular(R, jnp.eye(k_shock))
        RTinv_P_Rinv = Rinv.T @ P @ Rinv
        s, U = jnp.linalg.eigh(RTinv_P_Rinv)
        A = X @ Rinv @ U
        b = A.T @ y

        ypy = jnp.sum(jnp.square(y))
        ApA = A.T @ A

        def eval_f(lam):
            d = 1 / (1 + lam * s)
            bd = b * d
            rss = ypy - 2 * b.T @ bd + bd.T @ ApA @ bd
            dfit = jnp.sum(d)
            gcv = rss / (1 - dfit / n) ** 2
            aic = jnp.log(rss) + 2 * dfit / n
            return jnp.array([rss, gcv, aic, dfit])

        # Build the grid and map the step function over it
        lam_grid = n * jnp.pow(
            10, jnp.linspace(lam_grid_lower, lam_grid_upper, lam_grid_size)
        )
        values = jax.vmap(eval_f)(lam_grid)

        argmin_gcv = jnp.argmin(values[:, 1])
        min_gcv = lam_grid[argmin_gcv]

        return min_gcv, values


@dataclass
class SmoothLocalProjectionResult:
    model: SmoothLocalProjection
    knots: np.ndarray
    P: np.ndarray

    lam: float
    coef: np.ndarray
    vcov_nw: np.ndarray
    tune_trace: np.ndarray | None

    _Xs: np.ndarray | None
    _Ys: np.ndarray | None
    _mask: np.ndarray | None
    _us: np.ndarray | None
    _engine: str = "np"

    @property
    def H(self):
        return self.model.H

    @cached_property
    def B(self):
        return self.model.B

    @cached_property
    def irf(self):
        B = self.model.B
        n_splines = B.shape[1]
        n_shocks = self.model.shock.shape[1]
        irfs = [
            (
                B @ self.coef[i * n_splines : (i + 1) * n_splines].reshape((-1, 1))
            ).flatten()
            for i in range(n_shocks)
        ]
        irf = np.stack(irfs, axis=1)
        if n_shocks == 1:
            irf = irf.squeeze()
        return irf

    @cached_property
    def se(self):
        B = self.model.B
        n_splines = B.shape[1]
        n_shocks = self.model.shock.shape[1]
        lib = jnp if self._engine == "jax" else np
        ses = []
        for i in range(n_shocks):
            shock_slice = slice(i * n_splines, (i + 1) * n_splines)
            ses.append(
                lib.sqrt(
                    lib.diag(B @ self.vcov_nw[shock_slice, shock_slice] @ B.transpose())
                )
            )
        se = np.stack(ses, axis=1)
        if n_shocks == 1:
            se = se.squeeze(axis=1)
        return se
