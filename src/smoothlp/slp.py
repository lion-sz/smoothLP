from dataclasses import dataclass
from functools import cached_property, partial
from datetime import datetime
from numbers import Real
from typing import Literal

import jax
import numpy as np
from jax import lax, Array
from jax import numpy as jnp
from jaxtyping import Float
from jax.numpy import newaxis as nax

from smoothlp._blocked import blocked_cholesky, block_solve, fill
from smoothlp.models import AbstractLocalProjection
from smoothlp.utils import bspline, demean, build_stacked, partial_out


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
        self.p = 3  # Only cubic splines implemented.
        self.n_diff = target_poly_order + 1  # Check how other packages handle this!

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
        lam: float | Literal["cv", "gcv"] = "gcv",
        lam_grid_lower: float = -3.0,
        lam_grid_upper: float = 1.0,
        lam_grid_size: int = 1_000,
        partial_out: bool = True,
    ):
        if isinstance(lam, Real):
            lam_selection = None
            lam = float(lam)
        elif lam in (None, "gcv", "cv"):
            # Keep None as the backwards-compatible spelling of GCV selection.
            lam_selection = "gcv" if lam is None else lam
            lam = None
        else:
            raise ValueError("lam must be a real number, None, 'gcv', or 'cv'")

        if len(self.fe) > 0:
            res = demean(self.fe, [self.endog, self.shock, self.exog], mask=self.mask)
            self.endog, self.shock, self.exog = res

        params = {
            "H": self.H,
            "n_diff": self.n_diff,
            "lam_selection": lam_selection,
            "lam": lam,
            "lam_grid_lower": lam_grid_lower,
            "lam_grid_upper": lam_grid_upper,
            "lam_grid_size": lam_grid_size,
        }

        s = datetime.now()
        # Demeaning uses omission masks; stacked estimation uses inclusion weights.
        inclusion_mask = (
            None if self.mask is None else jnp.logical_not(self.mask).astype(float)
        )
        res = jax.block_until_ready(
            self._fit(
                self.B,
                self.endog,
                self.shock,
                self.exog,
                inclusion_mask,
                partial_out_exog=partial_out,
                **params,
            )
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
    def _comp_XTX(shock, mask_s, B, n_bands, partial_out_exog):
        n = shock.shape[1]
        temp = B[0, :3].reshape((-1, 1)) @ B[0, :3].reshape((1, -1))

        def acc(carry, data):
            if partial_out_exog:
                mask_h, shock_h = data
            else:
                mask_h, shock_h = data, shock
            xxT = jnp.einsum("ti,tj,t->ij", shock_h, shock_h, mask_h)
            res = temp[:, 0][:, jnp.newaxis, jnp.newaxis] * xxT[jnp.newaxis, :, :]
            res = res.at[:2, :, :].add(carry[[0, 1], :, :])
            carryp = temp[[1, 1, 2], [1, 2, 2]].reshape((3, 1, 1)) * xxT
            carryp = carryp.at[0, :, :].add(carry[2, :, :])
            return carryp, res

        # Residualized shocks have shape (T, n, H+1). Scan horizons together
        # with their masks, so each iteration receives a (T, n) shock slice.
        # Joint estimation keeps one 2D shock array shared by all horizons.
        scan_data = jnp.swapaxes(mask_s, 0, 1)
        if partial_out_exog:
            scan_data = (scan_data, jnp.moveaxis(shock, -1, 0))
        final_carry, XTX_stacked = lax.scan(acc, jnp.zeros((3, n, n)), scan_data)
        XTX_stacked = jnp.concat(
            [
                XTX_stacked,
                jnp.concat([final_carry[[0, 1], :, :], jnp.zeros((1, n, n))]).reshape(
                    (1, 3, n, n)
                ),
                jnp.concat([final_carry[[2], :, :], jnp.zeros((2, n, n))]).reshape(
                    (1, 3, n, n)
                ),
            ]
        )
        # Higher difference orders widen the penalty, not the spline Gram.
        XTX_stacked = jnp.pad(XTX_stacked, ((0, 0), (0, n_bands - 3), (0, 0), (0, 0)))
        return XTX_stacked

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "H",
            "n_diff",
            "partial_out_exog",
            "lam_selection",
            "lam_grid_size",
        ),
    )
    def _fit(
        B: Float[Array, "{H}+1 b"],
        endog: Float[Array, " T"],
        shock: Float[Array, "T s"],
        exog: Float[Array, "T k"],
        mask: Float[Array, " T"] | None,
        H: int,
        n_diff: int,
        lam_selection: None | Literal["cv", "gcv"],
        lam: None | float,
        lam_grid_lower: float,
        lam_grid_upper: float,
        lam_grid_size: int,
        partial_out_exog: bool = False,
    ):
        """Fit with block solves, optionally partialling out controls first."""
        T = endog.shape[0]
        k = B.shape[1]
        n_shocks = shock.shape[1]

        # Build the stacked mask and outcome array.
        if mask is None:
            mask = jnp.ones(T)
        Ys, mask_s = build_stacked("jax", endog, mask, H, "both")
        N = jnp.sum(mask_s)

        if partial_out_exog:
            # Partial out the exogenous variables
            Ys_r, shock_r, _ = partial_out(
                "jax", Ys, shock, exog, mask_s, leverage=False
            )
        else:
            Ys_r = Ys
            shock_r = jnp.concat([shock, exog], axis=1)
        n = shock_r.shape[1]

        # Build the penalty matrix
        q = n * k
        D = jnp.eye(k)
        for i in range(n_diff):
            D = jnp.diff(D, axis=0)
        # Coefficients are basis-major: (b1*x1, ..., b1*xn, b2*x1, ...).
        # Therefore every entry of the basis penalty is an n-by-n diagonal
        # block, preserving the block-banded structure in the basis dimension.
        basis_penalty = D.T @ D
        P = jnp.kron(basis_penalty, jnp.eye(n))

        # Slot i stores the block at (column + i, column). The Gram
        # matrix needs three bands; the penalty needs n_diff + 1.
        n_bands = max(3, n_diff + 1)
        Ps = jnp.zeros((k, n_bands, n, n))
        for i in range(min(n_diff + 1, k)):
            d = jnp.diag(basis_penalty, k=-i)
            Ps = Ps.at[: (k - i), i].set(d[:, nax, nax] * jnp.eye(n))

        XTX_stacked = SmoothLocalProjection._comp_XTX(
            shock_r, mask_s, B, n_bands, partial_out_exog
        )
        if partial_out_exog:
            XTy = jnp.einsum("tih,hk,th->ki", shock_r, B, mask_s * Ys_r)
        else:
            XTy = jnp.einsum("ti,hk,th->ki", shock_r, B, mask_s * Ys_r)
        XTy = XTy.reshape((q, 1))

        if lam_selection is not None:
            lam_grid = N * jnp.pow(
                10, jnp.linspace(lam_grid_lower, lam_grid_upper, lam_grid_size)
            )
            n_fitted = exog.shape[1] * (H + 1) if partial_out_exog else 0
            if lam_selection == "gcv":
                ypy = jnp.sum(jnp.square(Ys_r * mask_s))
                lam, tune_trace = SmoothLocalProjection._tune_lambda(
                    lam_grid,
                    XTX_stacked,
                    XTy,
                    ypy,
                    Ps,
                    N,
                    n_fitted,
                )
            elif lam_selection == "cv":
                # lam, tune_trace = 10.0, None
                lam, tune_trace = SmoothLocalProjection._tune_lambda_cv(
                    lam_grid,
                    B,
                    Ps,
                    Ys,
                    shock,
                    exog,
                    mask_s,
                    n_bands,
                    partial_out_exog,
                )
        else:
            tune_trace = None

        # Here I need to tune lambda
        design = XTX_stacked + lam * Ps
        L = blocked_cholesky(design)
        theta = block_solve(L, XTy.reshape((k, n, 1)))
        theta = block_solve(L, theta, lower=False).reshape((q, 1))
        theta_s = theta.reshape((k, n))

        if partial_out_exog:
            ys_hat = jnp.einsum("tsh,hk,ks->th", shock_r, B, theta_s)
        else:
            ys_hat = jnp.einsum("ts,hk,ks->th", shock_r, B, theta_s)
        us = (Ys_r - ys_hat) * mask_s

        # Select shock-major spline blocks for the covariance matrix,
        # while retaining basis-major ordering internally.
        shock_indices = (
            jnp.arange(n_shocks)[:, jnp.newaxis] + n * jnp.arange(k)[jnp.newaxis, :]
        ).reshape(-1)
        selector = jnp.eye(q)[shock_indices]

        # Compute W = bread^-1 selector.T.
        bread_vcov = XTX_stacked + 0.1 * lam * Ps
        bread_L = blocked_cholesky(bread_vcov)
        W = block_solve(bread_L, selector.T.reshape((k, n, n_shocks * k)), lower=True)
        W = block_solve(bread_L, W, lower=False)
        if partial_out_exog:
            g = jnp.einsum("th,tsh,hk,ksl->tl", us, shock_r, B, W)
        else:
            g = jnp.einsum("tk,ts,ksl->tl", us @ B, shock_r, W)

        def accumulate(carry, lag):
            shifted, weighted = carry

            # Advance one time step, zero-padding instead of wrapping.
            shifted = jnp.roll(shifted, -1, axis=0).at[-1].set(0.0)
            weight = 1.0 - lag / (H + 1)
            return (shifted, weighted + weight * shifted), None

        (_, weighted), _ = lax.scan(
            accumulate,
            (g, 0.5 * g),
            jnp.arange(1, H + 1),
        )

        cross = g.T @ weighted
        lrv = cross + cross.T
        coef = theta[shock_indices].reshape((n_shocks, k)).T

        shocks_partial = shock if partial_out_exog else None
        return shocks_partial, Ys, P, mask_s, us, coef, lrv, lam, tune_trace

    @staticmethod
    def _tune_lambda(
        lam_grid,
        XTX_stacked,
        XTy,
        ypy,
        Ps,
        N: int,
        n_fitted: int,
    ):
        k, _, n, _ = Ps.shape
        q = k * n
        # Here I need to tune lambda
        reference_lam = 10.0
        sl = XTX_stacked + reference_lam * Ps

        L = blocked_cholesky(sl)
        res = jnp.eye(q).reshape((k, n, q))
        Linv = block_solve(L, res).reshape((q, q))

        RPR = Linv @ fill(Ps) @ Linv.T
        s, U = jnp.linalg.eigh(RPR)
        s = jnp.maximum(s, 0.0)  # P is positive semidefinite
        # b is UT (R-T CTy), I can compute the right part cheaply
        RinvT_CT_y = block_solve(L, XTy.reshape((k, n, 1))).reshape((q, 1))
        b = (U.T @ RinvT_CT_y).flatten()
        # Apa is a diagonal matrix
        ApA = jnp.ones_like(s) - reference_lam * s

        def eval_f(lam):
            d = 1 / (1 + (lam - reference_lam) * s)
            bd = b * d
            rss = ypy - 2 * jnp.dot(b, bd) + jnp.dot(bd * ApA, bd)
            # A is whitened at reference_lam, so A.T @ A is not identity.
            dfit = jnp.sum(ApA * d) + n_fitted
            gcv = rss / (1 - dfit / N) ** 2
            aic = jnp.log(rss) + 2 * dfit / N
            return jnp.array([rss, gcv, aic, dfit])

        values = jax.vmap(eval_f)(lam_grid)
        argmin_gcv = jnp.argmin(values[:, 1])
        min_gcv = lam_grid[argmin_gcv]
        lam = min_gcv
        jax.debug.print(
            "Picking lambda {lam}, position {pos}.", lam=lam, pos=argmin_gcv
        )
        return lam, values

    @staticmethod
    def _tune_lambda_cv(
        lam_grid,
        B,
        Ps,
        Ys,
        shock,
        exog,
        mask_s,
        n_bands: int,
        partial_out_exog: bool,
    ):
        n_folds = 5
        k, _, n, _ = Ps.shape
        # n_shocks = shock.shape[1]
        n_splines = B.shape[1]
        Th = jnp.sum(mask_s, axis=0)

        index = (jnp.cumsum(mask_s, axis=0) - 1) / (Th[jnp.newaxis, :] / n_folds)
        index = jnp.floor(index)

        def fold(_, ind: float):
            """
            This function uses a slightly tuned version of the Demmler-Reisch orthogonalization.
            It makes sure that I'm computing the out of sample Rss by slightly adjusting which y
            is used in the regression.
            """
            # Set up the masks
            m_train = jnp.where(index == ind, 0.0, mask_s)
            m_test = jnp.where(index != ind, 0.0, mask_s)
            # Partial out the regressors
            if partial_out_exog:
                Ys_r, shock_r, _ = partial_out(
                    "jax",
                    Ys,
                    shock,
                    exog,
                    m_train,
                    leverage=False,
                    unmasked=True,
                )
            else:
                Ys_r, shock_r = Ys, jnp.concat([shock, exog], axis=1)
            n_shocks = shock_r.shape[1]

            Ys_test = Ys_r * m_test
            # Set up the main matrix
            q = n_shocks * n_splines

            XTX = SmoothLocalProjection._comp_XTX(
                shock_r, m_train, B, n_bands, partial_out_exog
            )
            XTX_test = SmoothLocalProjection._comp_XTX(
                shock_r, m_test, B, n_bands, partial_out_exog
            )
            if partial_out_exog:
                XTy = jnp.einsum("tih,hk,th->ki", shock_r, B, m_train * Ys_r).flatten()
                XTy_test = jnp.einsum(
                    "tih,hk,th->ki", shock_r, B, m_test * Ys_r
                ).flatten()
            else:
                XTy = jnp.einsum("ti,hk,th->ki", shock_r, B, m_train * Ys_r).flatten()
                XTy_test = jnp.einsum(
                    "ti,hk,th->ki", shock_r, B, m_test * Ys_r
                ).flatten()

            ypy = jnp.sum(jnp.square(Ys_test))

            reference_lam = 10.0
            sl = XTX + reference_lam * Ps
            L = blocked_cholesky(sl)
            res = jnp.eye(q).reshape((k, n, q))
            Linv = block_solve(L, res).reshape((q, q))

            RPR = Linv @ fill(Ps) @ Linv.T
            s, U = jnp.linalg.eigh(RPR)
            s = jnp.maximum(s, 0.0)
            Z = Linv.T @ U
            b = Z.T @ XTy
            b_test = Z.T @ XTy_test

            # Apa is a diagonal matrix
            At_T_At = Z.T @ fill(XTX_test) @ Z

            def eval_f(lam):
                d = 1 / (1 + (lam - reference_lam) * s)
                bd = b * d
                rss = ypy - 2 * b_test.T @ bd + bd.T @ At_T_At @ bd
                return rss

            return None, jax.vmap(eval_f)(lam_grid) / jnp.sum(m_test)

        _, mse = lax.scan(fold, None, jnp.arange(n_folds))
        mse = jnp.mean(mse, axis=0)

        lam_min = lam_grid[jnp.argmin(mse)]

        return lam_min, mse


@dataclass
class SmoothLocalProjectionResult:
    """SLP results with shock columns.

    Coefficients have shape (n_splines, n_shocks); IRFs and standard errors
    have shape (H + 1, n_shocks). Covariance uses shock-major spline blocks.
    """

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
        return self.B @ self.coef

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
        return np.stack(ses, axis=1)
