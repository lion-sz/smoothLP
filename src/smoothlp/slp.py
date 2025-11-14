import warnings
from functools import cached_property
from dataclasses import dataclass

import numpy as np
from scipy import linalg as sla
import jax
from jax import Array, lax, numpy as jnp, scipy as jsp

from smoothlp.models import AbstractLocalProjection
from smoothlp.utils import bspline, demean


def invert(X):
    U = sla.cholesky(X)
    U_inv = sla.solve_triangular(U, np.eye(X.shape[0]))
    return U_inv @ U_inv.T


def invert_jax(X):
    U = jsp.linalg.cholesky(X)
    U_inv = jsp.linalg.solve_triangular(U, jnp.eye(X.shape[0]))
    return U_inv @ U_inv.T


@dataclass
class SmoothLocalProjection(AbstractLocalProjection):
    fe: tuple[np.ndarray] | np.ndarray | None

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
        n_diff: int = 2,
        fe: tuple[np.ndarray] | np.ndarray | None = None,
        mask: np.ndarray | None = None,
        mask_type: str = "both",
        mask_any_prev_h: bool = True,
    ):
        super().__init__(endog, shock, exog, H, mask, mask_type)
        if fe is None:
            self.fe = ()
        else:
            if fe and not isinstance(fe, (list, tuple)):
                self.fe = (fe,)
            else:
                self.fe = fe
        self.p = p
        self.n_diff = n_diff

        self.mask_any_prev_h = mask_any_prev_h

        self._Xs = None
        self._Ys = None
        self._omit = None
        self._us = None
        self._engine = None

    def fit(
        self,
        lam: float | None,
        lam_grid_lower: float = -5.0,
        lam_grid_upper: float = 5.0,
        lam_grid_size: int = 10_000,
        engine: str = "np",
    ):
        self._engine = engine
        # First create the splines. Indexed by horizon x basis function
        knots = np.arange(-2, self.H + 3).astype(np.float64)
        B = np.array([bspline(self.p, knots, x) for x in range(self.H + 1)])

        if len(self.fe) > 0:
            res = demean(self.fe, [self.endog, self.shock, self.exog], mask=self.mask)
            self.endog, self.shock, self.exog = res

        if engine == "jax":
            engine_fn = SmoothLocalProjection._fit_jax
        else:
            engine_fn = SmoothLocalProjection._fit_np

        X, Y, P, omit, u, theta_hat, lam, tune_trace = engine_fn(
            B,
            self.endog,
            self.shock,
            self.exog,
            self.mask,
            self.mask_type,
            self.mask_any_prev_h,
            self.H,
            lam,
            lam_grid_lower,
            lam_grid_upper,
            lam_grid_size,
        )
        return SmoothLocalProjectionResult(
            self,
            knots,
            P,
            lam,
            theta_hat,
            tune_trace,
            _Xs=np.where(omit, 0.0, X),
            _Ys=Y,
            _omit=omit,
            _us=u,
            _engine=engine,
        )

    @staticmethod
    def _fit_np(
        B,
        endog,
        shock,
        exog,
        mask,
        mask_type: str,
        mask_any_prev_h: bool,
        H: int,
        lam: float,
        lam_grid_lower: float,
        lam_grid_upper: float,
        lam_grid_size: int,
    ):
        # Build the stacked arrays
        T, k = exog.shape
        omit_obs = []
        Y = []
        for t in range(T):
            ty = endog[t : min(T, t + H + 1)]
            if t + H + 1 >= T:
                to = np.arange(t, t + H + 1) >= T
                ty = np.concat((ty, np.zeros(H + 1 - (T - t))))
                assert ty.shape[0] == H + 1
            else:
                to = np.full(ty.shape, False)
            # Apply the mask to y values
            if mask is not None and mask_type in ("y", "both"):
                temp_mask = mask[t : min(T, t + H + 1)]
                if t + H + 1 >= T:
                    temp_mask = np.concat((temp_mask, np.full(H + 1 - (T - t), False)))
                if mask_any_prev_h:
                    temp_mask = temp_mask.cumsum() > 0
                to = to | temp_mask
            Y.append(ty)
            omit_obs.append(to)
        Ys = np.stack(Y, axis=0)
        exog = exog.reshape((T, 1, k))
        exog = np.repeat(exog, H + 1, axis=1)
        shock = np.einsum("hv,t->thv", B, shock)
        X_s = np.concat((shock, exog), axis=2)
        # Set the X matrix to zero for all values out of the bound
        omit_obs = np.stack(omit_obs, axis=0).reshape((T, H + 1, 1)).astype(bool)
        # Apply the mask to X values
        if mask is not None and mask_type in ("x", "both"):
            temp_mask = np.repeat(mask.reshape((T, 1, 1)), H + 1, axis=1)
            omit_obs = omit_obs | temp_mask

        Y = Ys.flatten()
        X = X_s.reshape(T * (H + 1), X_s.shape[2])
        Y = Y[~omit_obs.flatten()]
        X = X[~omit_obs.flatten(), :]

        # Build the penalty matrix
        P = np.zeros((X.shape[1], X.shape[1]))
        D = np.eye(H + 1)
        for i in range(2):
            D = np.diff(D, axis=0)
        P[: (H + 1), : (H + 1)] = D.T @ D

        if lam is None:
            lam, tune_trace = SmoothLocalProjection._tune_lambda(
                X,
                Y,
                P,
                Y.shape[0],
                H + 1,
                lam_grid_lower,
                lam_grid_upper,
                lam_grid_size,
            )
        else:
            tune_trace = None
        # Compute the coefficients
        XtX_lam = X.T @ X + lam * P
        XtX_lam_inv = invert(XtX_lam)
        theta_hat = XtX_lam_inv @ (X.T @ Y)
        u = Ys - np.einsum("thk,k->th", X_s, theta_hat)

        return X_s, Ys, P, omit_obs, u, theta_hat, lam, tune_trace

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
        use_jax: bool = False,
    ):
        """Tune lambda using Demmler-Reinsch Orthogonalization

        This proceedure was adopted from the LocalProjections Julia package.
        See Rupper (2003), appendix B.1.1 for a reference.
        """
        if use_jax:
            lib = jnp
            sla_lib = jsp.linalg
        else:
            lib = np
            sla_lib = sla

        # First check if I should parital out some regressors
        if k_shock < X.shape[1]:
            exog = X[:, k_shock:]
            X = X[:, :k_shock]

            EpE = exog.T @ exog
            EpE_inv = invert_jax(EpE) if use_jax else invert(EpE)
            X = X - (exog @ EpE_inv) @ (exog.T @ X)
            y = y - (exog @ EpE_inv) @ (exog.T @ y)

            P = P[:k_shock, :k_shock]

        # Set up the orthogonalization
        XpX = X.T @ X
        R = sla_lib.cholesky(XpX, lower=False)
        Rinv = sla_lib.solve_triangular(R, lib.eye(k_shock))
        RTinv_P_Rinv = Rinv.T @ P @ Rinv
        s, U = lib.linalg.eigh(RTinv_P_Rinv + 1e-9 * P)
        A = X @ Rinv @ U
        b = A.T @ y

        ypy = lib.sum(lib.square(y))
        ApA = A.T @ A

        def eval_f(lam):
            d = 1 / (1 + lam * s)
            bd = b * d
            rss = ypy - 2 * b.T @ bd + bd.T @ ApA @ bd
            dfit = lib.sum(d)
            gcv = rss / (1 - dfit / n) ** 2
            return lib.array([rss, gcv, dfit])

        # Build the grid and map the step function over it
        lam_grid = lib.exp(lib.linspace(lam_grid_lower, lam_grid_upper, lam_grid_size))
        if use_jax:
            values = jax.vmap(eval_f)(lam_grid)
        else:
            values = np.array(list(map(eval_f, lam_grid)))

        argmin_gcv = lib.argmin(values[:, 1])
        min_gcv = lam_grid[argmin_gcv]

        if not use_jax:
            if argmin_gcv == 0 or (argmin_gcv == lam_grid_size - 1):
                warnings.warn(
                    f"Selected optimal penalty at the edge of the grid: {min_gcv} (pos {argmin_gcv})."
                )

        return min_gcv, values

    @staticmethod
    def _fit_jax(
        B,
        endog,
        shock,
        exog,
        input_mask,
        mask_type: str,
        mask_any_prev_h: bool,
        H: int,
        lam: float,
        lam_grid_lower: float,
        lam_grid_upper: float,
        lam_grid_size: int,
    ):
        # Assumes that the inputs are already demeaned.
        T, k = exog.shape

        # First create the stacked Y.
        y = jnp.repeat(endog, H + 1, axis=0).reshape((T, H + 1))
        Y = lax.scan(
            lambda _, v: (None, jnp.roll(v[0], -v[1])),
            None,
            (y.T, np.arange(H + 1)),
        )[1].T
        # Build the mask for out of bounds observations.
        mask = np.arange(T).reshape((T, 1)) + np.arange(H + 1).reshape((1, H + 1))
        mask = mask < T
        # Apply the covid mask
        if input_mask is not None:
            if mask_type in ("x", "both"):
                mask = mask & input_mask.reshape((T, 1))
            if mask_type in ("y", "both"):
                input_mask = jnp.repeat(input_mask, H + 1, axis=0).reshape((T, H + 1))
                input_mask = lax.scan(
                    lambda _, v: (None, jnp.roll(v[0], -v[1])),
                    None,
                    (input_mask.T, np.arange(H + 1)),
                )[1].T
                mask = mask & input_mask
        # Build the stacked arrays
        X = exog.reshape((T, 1, k))
        X = jnp.repeat(X, H + 1, axis=1)
        s = B[jnp.newaxis, ...] * shock.reshape((T, 1, 1))
        X = jnp.concat((s, X), axis=2)
        # Apply the mask
        X = X * mask[..., jnp.newaxis]
        Y = Y * mask

        Y_flat = Y.flatten()
        X_flat = X.reshape((T * (H + 1)), H + 1 + k)

        # Build the penalty matrix
        P = jnp.zeros((H + 1 + k, H + 1 + k))
        D = jnp.eye(H + 1)
        for i in range(2):
            D = jnp.diff(D, axis=0)
        P = P.at[: (H + 1), : (H + 1)].set(D.T @ D)

        # Lambda tuning
        if lam is None:
            lam, tune_trace = SmoothLocalProjection._tune_lambda(
                X_flat,
                Y_flat,
                P,
                jnp.sum(mask),
                H + 1,
                lam_grid_lower,
                lam_grid_upper,
                lam_grid_size,
                True,
            )
        else:
            tune_trace = None

        # Compute the coefficients
        XpX_lam = X_flat.T @ X_flat + lam * P
        XpX_lam_inv = invert_jax(XpX_lam)
        theta_hat = XpX_lam_inv @ X_flat.T @ Y_flat
        u = Y - jnp.einsum("thk,k->th", X, theta_hat)

        return X, Y, P, ~mask[..., jnp.newaxis], u, theta_hat, lam, tune_trace


@dataclass
class SmoothLocalProjectionResult:
    model: SmoothLocalProjection
    knots: np.ndarray
    P: np.ndarray

    lam: float
    coef: np.ndarray
    tune_trace: np.ndarray | None

    _Xs: np.ndarray | None
    _Ys: np.ndarray | None
    _omit: np.ndarray | None
    _us: np.ndarray | None
    _engine: str | None

    @property
    def H(self):
        return self.model.H

    @cached_property
    def B(self):
        b = np.array([bspline(self.model.p, self.knots, x) for x in range(self.H + 1)])
        return b

    @cached_property
    def irf(self):
        coef = self.B @ self.coef[: (self.H + 1)].reshape((-1, 1))
        return coef.flatten()

    @cached_property
    def vcov_nw(self):
        if self.model._engine == "jax":
            return self._vcov_nw_jax(self._Xs, self._us, self.lam, self.P)
        else:
            return self._vcov_nw_inner(self._Xs, self._us, self.lam, self.P, False)

    @staticmethod
    @jax.jit
    def _vcov_nw_jax(Xs, us, lam, P):
        return SmoothLocalProjectionResult._vcov_nw_inner(Xs, us, lam, P, True)

    @staticmethod
    def _vcov_nw_inner(X, u, lam, P, use_jax=False):
        np_lib = jnp if use_jax else np

        T, lags = u.shape
        k = X.shape[2]

        xpx1 = np_lib.sum(np_lib.transpose(X, (0, 2, 1)) @ X, axis=0)
        outer = xpx1 + 0.1 * lam * P
        if use_jax:
            outer = invert_jax(outer)
        else:
            outer = invert(outer)
        # Compute the Gammas
        Xpu = np_lib.einsum("thk,th->tk", X, u)
        if use_jax:
            Xpu = jnp.concat((jnp.zeros((lags, k)), Xpu), axis=0)

            def f(lrv, h):
                Gamma = (
                    1
                    / T
                    * jnp.einsum(
                        "tk,ti->ki",
                        Xpu[lags : (lags + T), :],
                        lax.dynamic_slice_in_dim(Xpu, lags - h, T),
                    )
                )
                w = 1 - h / (lags + 1)
                lrv += w * (Gamma + Gamma.T * (h > 0))
                return lrv, None

            lrv, _ = lax.scan(f, jnp.zeros((k, k)), jnp.arange(lags + 1))
        else:
            lrv = jnp.zeros((k, k))
            for j in range(lags + 1):
                Gamma = (
                    1
                    / T
                    * np_lib.einsum("tk,ti->ki", Xpu[j:, ...], Xpu[: (T - j), ...])
                )
                if j == 0:
                    lrv += Gamma
                else:
                    lrv += (1 - j / (lags + 1)) * (Gamma + Gamma.T)
        vcov_nw = T * outer @ lrv @ outer
        return vcov_nw

    @cached_property
    def se(self):
        vcov = (
            self.B @ self.vcov_nw[: (self.H + 1), : (self.H + 1)] @ self.B.transpose()
        )
        if isinstance(vcov, Array):
            se = jnp.sqrt(jnp.diag(vcov))
        else:
            se = np.sqrt(np.diag(vcov))
        return se
