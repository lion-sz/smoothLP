import warnings
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

    mask: np.ndarray | None
    mask_type: str
    mask_any_prev_h: bool

    _Xs: np.ndarray | None
    _Ys: np.ndarray | None
    _omit: np.ndarray | None
    _us: np.ndarray | None

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
        self.endog = endog
        self.shock = shock
        self.exog = exog
        assert len(self.shock.shape) == 1
        if fe is None:
            self.fe = ()
        else:
            if fe and not isinstance(fe, (list, tuple)):
                self.fe = (fe,)
            else:
                self.fe = fe
        self.H = H
        self.p = p
        self.n_diff = n_diff

        assert mask_type in ("x", "y", "both")
        self.mask = mask
        self.mask_type = mask_type
        self.mask_any_prev_h = mask_any_prev_h

        self._Xs = None
        self._Ys = None
        self._omit = None
        self._us = None

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
            # TODO: Use the mask here.
            res = demean(self.fe, [self.endog, self.shock, self.exog], mask=self.mask)
            self.endog, self.shock, self.exog = res

        # Build the stacked arrays
        T, k = self.exog.shape
        omit_obs = []
        Y = []
        for t in range(T):
            ty = self.endog[t : min(T, t + self.H + 1)]
            if t + self.H + 1 >= T:
                to = np.arange(t, t + self.H + 1) >= T
                ty = np.concat((ty, np.zeros(self.H + 1 - (T - t))))
                assert ty.shape[0] == self.H + 1
            else:
                to = np.full(ty.shape, False)
            # Apply the mask to y values
            if self.mask is not None and self.mask_type in ("y", "both"):
                temp_mask = self.mask[t : min(T, t + self.H + 1)]
                if t + self.H + 1 >= T:
                    temp_mask = np.concat(
                        (temp_mask, np.full(self.H + 1 - (T - t), False))
                    )
                if self.mask_any_prev_h:
                    temp_mask = temp_mask.cumsum() > 0
                to = to | temp_mask
            Y.append(ty)
            omit_obs.append(to)
        self._Ys = np.stack(Y, axis=0)
        exog = self.exog.reshape((T, 1, k))
        exog = np.repeat(exog, self.H + 1, axis=1)
        shock = np.einsum("hv,t->thv", B, self.shock)
        X_s = np.concat((shock, exog), axis=2)
        # Set the X matrix to zero for all values out of the bound
        omit_obs = np.stack(omit_obs, axis=0).reshape((T, self.H + 1, 1)).astype(bool)
        # Apply the mask to X values
        if self.mask is not None and self.mask_type in ("x", "both"):
            temp_mask = np.repeat(self.mask.reshape((T, 1, 1)), self.H + 1, axis=1)
            omit_obs = omit_obs | temp_mask
        self._omit = omit_obs
        self._Xs = np.where(self._omit, 0.0, X_s)

        Y = self._Ys.flatten()
        X = self._Xs.reshape(T * (self.H + 1), X_s.shape[2])
        Y = Y[~omit_obs.flatten()]
        X = X[~omit_obs.flatten(), :]

        # Build the penalty matrix
        # For now only identity penalty is implemented
        P = np.zeros((X.shape[1], X.shape[1]))
        D = np.eye(self.H + 1)
        for i in range(2):
            D = np.diff(D, axis=1)
        P[: (self.H + 1), : (self.H + 1)] = D @ D.T

        if lam is None:
            lam, values = SmoothLocalProjection._tune_lambda(
                X,
                Y,
                P,
                self.H + 1,
                lam_grid_lower,
                lam_grid_upper,
                lam_grid_size,
            )
        # Compute the coefficients
        theta_hat = np.linalg.inv(X.T @ X + lam * P) @ X.T @ Y
        self._us = self._Ys - np.einsum("thk,k->th", self._Xs, theta_hat)

        return SmoothLocalProjectionResult(
            self,
            knots,
            P,
            lam,
            theta_hat,
        )

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


class SmoothLocalProjectionResult:
    model: SmoothLocalProjection
    knots: np.ndarray
    P: np.ndarray

    lam: float
    theta_hat: np.ndarray

    def __init__(self, model, knots, P, lam, theta_hat):
        self.model = model
        self.knots = knots
        self.P = P
        self.lam = lam
        self.theta_hat = theta_hat

    @property
    def H(self):
        return self.model.H

    @cached_property
    def B(self):
        b = np.array([bspline(self.model.p, self.knots, x) for x in range(self.H + 1)])
        return b

    @cached_property
    def irf(self):
        coef = self.B @ self.theta_hat[: (self.H + 1)].reshape((-1, 1))
        return coef.flatten()

    @cached_property
    def vcov_nw(self):
        X = self.model._Xs
        u = self.model._us
        T, _, k = X.shape
        lags = self.H + 1
        xpx1 = np.einsum("thi,thk->ik", X, X)
        outer = xpx1 + 0.1 * self.lam * self.P
        outer = np.linalg.inv(outer)
        # Compute the Gammas
        Xpu = np.einsum("thk,th->tk", X, u)
        lrv = np.zeros((k, k))
        for j in range(lags + 1):
            Gamma = 1 / T * np.einsum("tk,ti->ki", Xpu[j:, ...], Xpu[: (T - j), ...])
            if j == 0:
                lrv += Gamma
            else:
                lrv += (1 - j / (lags + 1)) * (Gamma + Gamma.T)
        vcov_nw = T * outer @ lrv @ outer
        return vcov_nw

    @cached_property
    def irf_se(self):
        vcov = (
            self.B @ self.vcov_nw[: (self.H + 1), : (self.H + 1)] @ self.B.transpose()
        )
        se = np.sqrt(np.diag(vcov))
        return se
