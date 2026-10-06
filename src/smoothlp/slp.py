from dataclasses import dataclass
from functools import cached_property, partial
from datetime import datetime
from numbers import Real
from typing import Literal

import jax
import numpy as np
from jax import lax, Array
from jax import numpy as jnp, scipy as jsp
from jaxtyping import Float
from jax.numpy import newaxis as nax

from smoothlp._blocked import blocked_cholesky, block_solve
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
        # First create the splines. Indexed by horizon x basis function

        assert not (not partial_out and lam == "cv"), (
            "Not partialing out and cv is not yet supported."
        )
        assert partial_out or self.p == 3, (
            "Non-cubic splines not yet supported with no partial out."
        )
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
        fit_function = self._fit if partial_out else self._fit_no_partial_out
        res = jax.block_until_ready(
            fit_function(
                self.B, self.endog, self.shock, self.exog, inclusion_mask, **params
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
    @partial(
        jax.jit,
        static_argnames=("H", "n_diff", "lam_selection", "lam_grid_size"),
    )
    def _fit_no_partial_out(
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
    ):
        """Fit without partialling out the exogenous variables."""
        T = endog.shape[0]
        k = B.shape[1]
        n_shocks = shock.shape[1]
        shock = jnp.concat([shock, exog], axis=1)
        n = shock.shape[1]
        # First compute the Bh x Bh matrices
        Bh = jnp.einsum("hi,hj->hij", B, B)

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

        # Stack the observations and build the mask
        if mask is None:
            mask = jnp.ones(T)
        Ys, mask_s = build_stacked("jax", endog, mask, H, "both")
        N = jnp.sum(mask_s)

        temp = B[0, :3].reshape((-1, 1)) @ B[0, :3].reshape((1, -1))

        def acc(carry, data):
            xxT = jnp.einsum("ti,tj,t->ij", shock, shock, data)
            res = temp[:, 0][:, jnp.newaxis, jnp.newaxis] * xxT[jnp.newaxis, :, :]
            res = res.at[:2, :, :].add(carry[[0, 1], :, :])
            carryp = temp[[1, 1, 2], [1, 2, 2]].reshape((3, 1, 1)) * xxT
            carryp = carryp.at[0, :, :].add(carry[2, :, :])
            return carryp, res

        final_carry, XTX_stacked = lax.scan(
            acc, jnp.zeros((3, n, n)), jnp.swapaxes(mask_s, 0, 1)
        )
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

        xxT_h = jnp.einsum("ti,tj,th->hij", shock, shock, mask_s)
        XTy = jnp.einsum("ti,hk,th->ki", shock, B, mask_s * Ys).reshape((q, 1))

        if lam_selection is not None:
            ypy = jnp.sum(jnp.square(Ys * mask_s))

            # Here I need to tune lambda
            reference_lam = 10
            sl = XTX_stacked + reference_lam * Ps

            L = blocked_cholesky(sl)
            res = jnp.eye(q).reshape((k, n, q))
            Linv = block_solve(L, res).reshape((q, q))

            RPR = Linv @ P @ Linv.T
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
                dfit = jnp.sum(ApA * d)
                gcv = rss / (1 - dfit / N) ** 2
                aic = jnp.log(rss) + 2 * dfit / N
                return jnp.array([rss, gcv, aic, dfit])

            lam_grid = N * jnp.pow(
                10, jnp.linspace(lam_grid_lower, lam_grid_upper, lam_grid_size)
            )
            values = jax.vmap(eval_f)(lam_grid)
            argmin_gcv = jnp.argmin(values[:, 1])
            jax.debug.print("Picking lambda position {pos}.", pos=argmin_gcv)
            min_gcv = lam_grid[argmin_gcv]
            lam = min_gcv

        # Here I need to tune lambda
        design = XTX_stacked + lam * Ps
        L = blocked_cholesky(design)
        theta = block_solve(L, XTy.reshape((k, n, 1)))
        theta = block_solve(L, theta, lower=False).reshape((q, 1))
        theta_s = theta.reshape((k, n))

        # Fit the main model.
        # design = XTX + lam * P
        # R = jnp.linalg.cholesky(design, upper=True)
        # theta = jsp.linalg.solve_triangular(R.T, XTy, lower=True)
        # theta = jsp.linalg.solve_triangular(R, theta, lower=False).flatten()
        # theta_s = theta.reshape((k, n))
        us = (Ys - jnp.einsum("ts,hk,ks->th", shock, B, theta_s)) * mask_s
        Xu = shock[:, jnp.newaxis, :] * us[:, :, jnp.newaxis]

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
        g = jnp.einsum("tk,ts,ksl->tl", us @ B, shock, W)

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
        lrv
        coef = theta[shock_indices].reshape((n_shocks, k)).T

        return None, Ys, P, mask_s, us, coef, lrv, lam, None

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=("H", "n_diff", "lam_selection", "lam_grid_size"),
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
        if lam_selection is not None:
            lam_grid = N * jnp.pow(
                10, jnp.linspace(lam_grid_lower, lam_grid_upper, lam_grid_size)
            )
            if lam_selection == "gcv":
                Xpy = jnp.einsum("tsh, th, hk -> sk", shock_r_m, Ys_r_m, B).reshape(q)
                ypy = jnp.sum(jnp.square(Y_r_m))
                n_fitted = k * (H + 1)
                lam, tune_trace = SmoothLocalProjection._tune_lambda(
                    lam_grid,
                    XpX,
                    Xpy,
                    ypy,
                    P,
                    N,
                    n_fitted,
                )
            elif lam_selection == "cv":
                lam, tune_trace = SmoothLocalProjection._tune_lambda_cv(
                    lam_grid, B, P, Ys, shock, exog, mask_s, 5
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

        return shock_r_m, Ys, P, mask_s, u, theta_hat_s.T, vcov_nw, lam, tune_trace

    @staticmethod
    def _tune_lambda(
        lam_grid,
        XpX,
        Xpy,
        ypy,
        P,
        n: int,
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

        values = jax.vmap(eval_f)(lam_grid)
        argmin_gcv = jnp.argmin(values[:, 1])
        min_gcv = lam_grid[argmin_gcv]

        return min_gcv, values

    @staticmethod
    # @partial(jax.jit, static_argnames=["n_folds"])
    def _tune_lambda_cv(
        lam_grid,
        B: Float[Array, "{H}+1 b"],
        P: Float[Array, "q q"],
        Ys: Float[Array, "T {H+1}"],
        shock: Float[Array, "T {H+1}"],
        exog: Float[Array, "T k"],
        mask_s: Float[Array, "T {H+1}"],
        n_folds: int,
    ):
        n_shocks = shock.shape[1]
        n_splines = B.shape[1]
        Th = jnp.sum(mask_s, axis=0)

        index = jnp.cumsum(mask_s, axis=0) / (Th[jnp.newaxis, :] / n_folds)
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
            Ys_r, shock_r, _ = partial_out(
                "jax",
                Ys,
                shock,
                exog,
                m_train,
                leverage=False,
                unmasked=True,
            )
            Ys_train = Ys_r * m_train
            Ys_test = Ys_r * m_test
            shock_train = shock_r * m_train[:, jnp.newaxis, :]
            shock_test = shock_r * m_test[:, jnp.newaxis, :]
            # Set up the main matrix
            q = n_shocks * n_splines
            XpX = jnp.einsum(
                "tih,tjh,hk,hl->ikjl", shock_train, shock_train, B, B
            ).reshape(q, q)
            XpX_test = jnp.einsum(
                "tih,tjh,hk,hl->ikjl", shock_test, shock_test, B, B
            ).reshape(q, q)
            Xpy = jnp.einsum("tsh, hk, th -> sk", shock_train, B, Ys_train).flatten()
            Xpy_test = jnp.einsum("tsh, hk, th -> sk", shock_test, B, Ys_test).flatten()
            ypy = jnp.sum(jnp.square(Ys_test))

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
            Z = Rinv @ U
            b = Z.T @ Xpy
            b_test = Z.T @ Xpy_test
            At_T_At = Z.T @ XpX_test @ Z

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
