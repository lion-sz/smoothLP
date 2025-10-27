import warnings
from datetime import datetime
from functools import cached_property

import numpy as np

from src.utils import bspline, demean


class SmoothLocalProjection:
    endog: np.ndarray
    exog: np.ndarray
    shock: np.ndarray
    fe: tuple[np.ndarray] | np.ndarray | None

    H: int
    p: int
    n_diff: int

    def __init__(
        self,
        endog: np.ndarray,
        shock: np.ndarray,
        exog: np.ndarray | None,
        H: int,
        p: int = 3,
        n_diff: int = 2,
        fe: tuple[np.ndarray] | np.ndarray | None = None,
    ):
        self.endog = endog
        self.shock = shock
        self.exog = exog
        if fe is None:
            self.fe = ()
        else:
            if fe and not isinstance(fe, (list, tuple)):
                self.fe = (fe,)
            else:
                self.fe = fe
        self.H = H
        self.p = p
        self.n_diff = 2

    def fit(
        self,
        lam: float | None,
        lam_grid_lower: float = -3.0,
        lam_grid_upper: float = 3.0,
        lam_grid_size: int = 10_000,
    ):
        # First create the splines. Indexed by horizon x basis function
        knots = np.arange(-2, self.H + 3).astype(np.float64)
        B = np.array([bspline(self.p, knots, x) for x in range(self.H + 1)])

        if len(self.fe) > 0:
            res = demean(self.fe, [self.endog, self.shock, self.exog])
            self.endog, self.shock, self.exog = res

        # Build the stacked arrays
        T = self.endog.shape[0]
        Ys = []
        Xs = []
        fes = []
        shock_s = []
        for t in range(T):
            Ys.append(self.endog[t : min(T, t + self.H + 1)])
            Xs.append(
                np.repeat(
                    self.exog[t, :].reshape((1, -1)), min(T - t, self.H + 1), axis=0
                )
            )
            shock_s.append(B[: min(T - t, self.H + 1), :] * self.shock[t])
            fes.append(list(np.repeat(fe[t], min(T - t, self.H + 1)) for fe in self.fe))
        Ys = np.concat(Ys)
        Xs = np.concat(Xs)
        shock_s = np.concat(shock_s)
        k_shock = shock_s.shape[1]
        X = np.concat([shock_s, Xs], axis=1)

        # Build the penalty matrix
        # For now only identity penalty is implemented
        P = np.zeros((X.shape[1], X.shape[1]))
        D = np.eye(self.H + 1)
        for i in range(2):
            D = np.diff(D, axis=1)
        P[: (self.H + 1), : (self.H + 1)] = D @ D.T

        if lam is None:
            lam, values = SmoothLocalProjection._tune_lambda(
                X, Ys, P, k_shock, lam_grid_lower, lam_grid_upper, lam_grid_size
            )

        # Compute the coefficients
        theta_hat = np.linalg.inv(X.transpose() @ X + lam * P) @ X.transpose() @ Ys

        return SmoothLocalProjectionResult(
            self.endog,
            self.exog,
            self.shock,
            self.H,
            self.p,
            knots,
            P,
            lam,
            theta_hat,
        ), values

    @staticmethod
    def _tune_lambda(
        X,
        y,
        P,
        k_shock: int,
        lam_grid_lower: float,
        lam_grid_upper: float,
        lam_grid_size: int,
    ):
        """Tune lambda using Demmler-Reinsch Orthogonalization

        This proceedure was adopted from the LocalProjections Julia package.
        See Rupper (2003), appendix B.1.1 for a reference.
        """
        n = y.shape[0]

        # First check if I should parital out some regressors
        if k_shock < X.shape[1]:
            exog = X[:, k_shock:]
            X = X[:, :k_shock]

            EpE_inv = np.linalg.inv(exog.T @ exog)
            X = X - (exog @ EpE_inv) @ (exog.T @ X)
            y = y - (exog @ EpE_inv) @ (exog.T @ y)

            P = P[:k_shock, :k_shock]

        # Set up the orthogonalization
        XpX = X.T @ X
        R = np.linalg.cholesky(XpX, upper=True)
        Rinv = np.linalg.inv(R)
        RTinv_P_Rinv = Rinv.T @ P @ Rinv
        s, U = np.linalg.eig(RTinv_P_Rinv + 1e-9 * P)
        A = X @ Rinv @ U
        b = A.T @ y

        ypy = np.sum(np.square(y))
        ApA = A.T @ A

        def eval_f(lam):
            d = 1 / (1 + lam * s)
            bd = b * d
            rss = ypy - 2 * b.T @ bd + bd.T @ ApA @ b
            dfit = np.sum(d)
            gcv = rss / (1 - dfit / n) ** 2
            return rss, gcv, dfit

        # Build the grid and map the step function over it
        lam_grid = np.exp(np.linspace(lam_grid_lower, lam_grid_upper, lam_grid_size))
        values = np.array(list(map(eval_f, lam_grid)))

        argmin_gcv = np.argmin(values[:, 1])
        min_gcv = lam_grid[argmin_gcv]

        if argmin_gcv == 0 or (argmin_gcv == lam_grid_size - 1):
            warnings.warn(
                f"Selected optimal penalty at the edge of the grid: {min_gcv} (pos {argmin_gcv})."
            )

        return min_gcv, values

    def _D0(self, h):
        return np.eye(h)

    def _D1(self, h):
        D = np.zeros((h, h))
        D[np.arange(1, h), np.arange(h - 1)] = -0.5
        D[np.arange(h - 1), np.arange(1, h)] = 0.5
        D[0, 0] = -1
        D[0, 1] = 1
        D[h - 1, h - 1] = 1
        D[h - 1, h - 2] = -1
        return D

    def _D2(self, h):
        D = np.diag(np.full((h,), -2))
        D[np.arange(1, h), np.arange(h - 1)] = 1
        D[np.arange(h - 1), np.arange(1, h)] = 1
        D[0, 0] = 1
        D[0, 1] = -2
        D[0, 2] = 1
        D[h - 1, h - 3] = 1
        D[h - 1, h - 2] = -2
        D[h - 1, h - 1] = 1
        return D


class SmoothLocalProjectionResult:
    endog: np.ndarray
    exog: np.ndarray
    shock: np.ndarray

    H: int
    p: int
    knots: np.ndarray
    P: np.ndarray

    lam: float
    theta_hat: np.ndarray

    def __init__(self, endog, exog, shock, H, p, knots, P, lam, theta_hat):
        self.endog = endog
        self.exog = exog
        self.shock = shock
        self.H = H
        self.p = p
        self.knots = knots
        self.P = P
        self.lam = lam
        self.theta_hat = theta_hat

    @cached_property
    def B(self):
        b = np.array([bspline(self.p, self.knots, x) for x in range(self.H + 1)])
        return b

    @cached_property
    def irf(self):
        coef = self.B @ self.theta_hat[: (self.H + 1)].reshape((-1, 1))
        return coef.flatten()

    @cached_property
    def vcov_nw(self):
        T = self.endog.shape[0]
        k = self.theta_hat.shape[0]
        B = np.array([bspline(self.p, self.knots, x) for x in range(self.H + 1)])

        # Build the time matrices
        s1 = datetime.now()
        xut = []
        xpx = np.zeros((k, k))
        for t in range(T):
            yt = self.endog[t : min(T, t + self.H + 1)]
            xt = np.repeat(
                self.exog[t, :].reshape((1, -1)), min(T - t, self.H + 1), axis=0
            )
            st = B[: min(T - t, self.H + 1), :] * self.shock[t]
            xt = np.concat([st, xt], axis=1)

            xpx += xt.T @ xt
            ut = (yt - xt @ self.theta_hat).reshape((-1, 1))
            xut.append(xt.T @ ut)

        s2 = datetime.now()
        # Build the outer matrices
        outer = xpx + 0.1 * self.lam * self.P
        outer = np.linalg.inv(outer)
        # LRV
        deltas = []
        for l in range(self.H + 1):
            dl = np.zeros((k, k))
            for t in range(l, T):
                dl += xut[t] @ xut[t - l].T
            deltas.append(1 / T * dl)
        lrv = deltas[0]
        for l in range(1, self.H + 1):
            lrv += (1 - l / (1 + self.H + 1)) * (deltas[l] + deltas[l].T)

        vcov_nw = T * outer @ lrv @ outer
        s3 = datetime.now()
        print(f"build: {(s2 - s1).total_seconds()}, lrv: {(s3 - s1).total_seconds()}")
        return vcov_nw

    @cached_property
    def irf_se(self):
        vcov = (
            self.B @ self.vcov_nw[: (self.H + 1), : (self.H + 1)] @ self.B.transpose()
        )
        se = np.sqrt(np.diag(vcov))
        return se
