from dataclasses import dataclass
from functools import cached_property, partial
from datetime import datetime

import jax
import numpy as np
from jax import lax, Array
from jax import numpy as jnp, scipy as jsp
from jaxtyping import Float

from smoothlp.models import AbstractLocalProjection
from smoothlp.utils import bspline, demean, invert, build_stacked, partial_out


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
        assert mask_type == "both", "Only mask_type='both' is supported"
        assert mask_any_prev_h, "Only mask_any_prev_h=True is supported"
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
        """Compute the knots.

        Note, that I'm deviating from the paper here and agree with the R code.
        The left most knot is at H_min -3 not -2.
        Without this I'm running into this peculiarity that the splines do not
        always sum up to 1, which then messes with the penalty.
        """
        return np.arange(-3, self.H).astype(np.float64)

    @property
    def B(self):
        B = bspline(self.p, self.knots, np.arange(self.H + 1))
        return B

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
            "H": self.H,
            "n_diff": self.n_diff,
            "lam": lam,
            "lam_grid_lower": lam_grid_lower,
            "lam_grid_upper": lam_grid_upper,
            "lam_grid_size": lam_grid_size,
        }
        engine_fn = jax.jit(partial(SmoothLocalProjection._fit, **params))

        s = datetime.now()
        # Demeaning uses omission masks; stacked estimation uses inclusion weights.
        inclusion_mask = (
            None if self.mask is None else jnp.logical_not(self.mask).astype(float)
        )
        res = jax.block_until_ready(
            engine_fn(self.B, self.endog, self.shock, self.exog, inclusion_mask)
        )
        shock, Y, P, mask, u, theta_hat, vcov_nw, lam, tune_trace = (
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
            _Xs=shock,
            _Ys=Y,
            _mask=mask,
            _us=u,
        )

    @staticmethod
    def _fit(
        B: Float[Array, "{H}+1 b"],
        endog: Float[Array, " T"],
        shock: Float[Array, "T s"],
        exog: Float[Array, "T k"],
        mask: Float[Array, " T"] | None,
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

        if mask is None:
            mask = jnp.ones(T)
        Ys, mask_s = build_stacked("jax", endog, mask, H, "both")
        N = jnp.sum(mask_s)

        # Partial out the exogenous variables
        Ys_r_m, shock_r_m, _ = partial_out(
            "jax", Ys, shock, exog, mask_s, leverage=False
        )
        Y_r_m = Ys_r_m.reshape(T * (H + 1))

        # Set up the main matrix
        q = n_shocks * n_splines
        XpX = jnp.einsum("tih,tjh,hk,hl->ikjl", shock_r_m, shock_r_m, B, B).reshape(
            q, q
        )

        # Build the penalty matrix
        P = jnp.zeros((q, q))
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
            Xpy = jnp.einsum("tsh, th, hk -> sk", shock_r_m, Ys_r_m, B).reshape(q)
            ypy = jnp.sum(jnp.square(Y_r_m))
            n_fitted = k * (H + 1)
            lam, tune_trace = SmoothLocalProjection._tune_lambda(
                XpX,
                Xpy,
                ypy,
                P,
                N,
                lam_grid_lower,
                lam_grid_upper,
                lam_grid_size,
                n_fitted,
            )
        else:
            tune_trace = None

        XpX_lam = XpX + lam * P
        XpX_lam_inv = invert("jax", XpX_lam)
        t_Xpy = jnp.einsum("tsh, hk, th -> sk", shock_r_m, B, Ys_r_m).flatten()
        theta_hat = XpX_lam_inv @ t_Xpy
        theta_hat_s = theta_hat.reshape(n_shocks, n_splines)
        ys_hat = jnp.einsum("tsh, hk, sk -> th", shock_r_m, B, theta_hat_s)
        u = Ys_r_m - ys_hat

        # Compute the covariance matrix for the splines.
        # I first accumulate the weighted Xu values,
        # before multiplying them once with the Xu to get the cross
        # product. I basically first accumulate and only then multiply.
        # This makes the computation much faster.
        def accumulate(carry, lag):
            shifted, weighted = carry

            # Advance one time step, zero-padding instead of wrapping.
            shifted = jnp.roll(shifted, -1, axis=0).at[-1].set(0.0)
            weight = 1.0 - lag / (H + 1)
            return (shifted, weighted + weight * shifted), None

        Xu = jnp.einsum("tsh, th, hk -> tsk", shock_r_m, u, B)
        Xu = Xu.reshape(T, n_shocks * n_splines)

        (_, weighted), _ = lax.scan(
            accumulate,
            (Xu, 0.5 * Xu),
            jnp.arange(1, H + 1),
        )

        cross = Xu.T @ weighted
        lrv = cross + cross.T

        XpX_small_P = XpX + 0.1 * lam * P
        XpX_inv = invert("jax", XpX_small_P)
        vcov_nw = XpX_inv @ lrv @ XpX_inv

        return shock_r_m, Ys, P, mask_s, u, theta_hat, vcov_nw, lam, tune_trace

    @staticmethod
    def _tune_lambda(
        XpX,
        Xpy,
        ypy,
        P,
        n: int,
        lam_grid_lower: float,
        lam_grid_upper: float,
        lam_grid_size: int,
        n_fitted: int,
    ):
        """Tune lambda using Demmler-Reinsch Orthogonalization

        This procedure was adopted from the LocalProjections Julia package.
        See Rupper (2003), appendix B.1.1 for a reference.
        """
        # This matrix is rank deficint (more splines than horizons).
        # To make it well behaved, I add a rather strong penalty to it.
        # Note, that I need to subtract this later on.
        penalty_scale = jnp.minimum(jnp.max(jnp.abs(P)), 1.0)
        reference_lam = jnp.max(jnp.abs(XpX)) / penalty_scale
        R = jsp.linalg.cholesky(XpX + reference_lam * P, lower=False)
        Rinv = jsp.linalg.solve_triangular(R, jnp.eye(XpX.shape[0]))
        RTinv_P_Rinv = Rinv.T @ P @ Rinv
        s, U = jnp.linalg.eigh(RTinv_P_Rinv)
        s = jnp.maximum(s, 0.0)  # P is positive semidefinite
        jax.debug.print("Condition number: {cond}", cond=jnp.max(s) / jnp.min(s + 1e-9))
        Z = Rinv @ U
        b = Z.T @ Xpy
        ApA = Z.T @ XpX @ Z

        def eval_f(lam):
            d = 1 / (1 + (lam - reference_lam) * s)
            bd = b * d
            rss = ypy - 2 * b.T @ bd + bd.T @ ApA @ bd
            # A is whitened at reference_lam, so A.T @ A is not identity.
            dfit = jnp.sum(jnp.diag(ApA) * d) + n_fitted
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
