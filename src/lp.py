from typing import NamedTuple

import numpy as np
from statsmodels import api as sm


class LocalProjection:
    """Local Projection

    Run the standard local projection from Jorda 2006.

    This method follows the R package `lpirfs`
    and both coefficients and standard errors in the basic one outcome,
    one shock case are tested against it.

    A not on standard errors:

    `lpirfs` computes either non-robust `HC0` or Newey-West standard errors,
    this package deviates and provides `HC3` standard errors by default
    (though `HC0`-`HC3` and NW are implemented).

    When computing the Newey-West estimator, for horizon `h` I use `h+1` lags.
    In this I follow the `lpirfs` package.
    I however apply the small sample correction, following `statsmodels`
    instead of `lpirfs`.
    """

    endog: np.ndarray
    exog: np.ndarray
    shock: np.ndarray

    H: int

    mask: np.ndarray | None
    mask_type: str

    def __init__(
        self,
        endog,
        exog,
        shock,
        H,
        mask: np.ndarray | None = None,
        mask_type: str = "both",
    ):
        assert len(endog.shape) == 1
        assert len(exog.shape) == 2
        assert len(shock.shape) == 1

        assert endog.shape[0] == exog.shape[0]
        assert endog.shape[0] == shock.shape[0]
        assert mask is None or endog.shape == mask.shape

        assert mask_type in ("x", "y", "both"), (
            f"Mask type '{mask_type}' not recognized"
        )

        self.endog = endog
        self.exog = exog
        self.shock = shock
        self.H = H
        self.mask = mask
        self.mask_type = mask_type

    def fit(self, vcov="HC3", engine="np"):
        assert engine in ("np", "sm"), f"Engine '{engine}' not recognised."
        assert vcov in ("HC0", "HC1", "HC3", "HAC"), f"vcov '{vcov}' not recognised."

        X = np.concatenate((self.shock.reshape(-1, 1), self.exog), axis=1)

        if engine == "np":
            engine_func = LocalProjection._estimation_np
        else:
            engine_func = LocalProjection._estimation_sm
        coef, vcov_hc0, vcov_hc1, vcov_hc3, vcov_hac = engine_func(
            self.endog, X, self.H, vcov, self.mask, self.mask_type
        )

        return LocalProjectionResult(
            # The data objects
            self.endog,
            self.exog,
            self.shock,
            self.H,
            # Estimated objects
            coef,
            vcov_hc0,
            vcov_hc1,
            vcov_hc3,
            vcov_hac,
        )

    @staticmethod
    def _estimation_np(
        Y: np.ndarray,
        X: np.ndarray,
        H: int,
        vcov: str,
        mask: np.ndarray | None,
        mask_type: str = "both",
    ):
        T, k = X.shape
        coef = np.zeros((H + 1, k))
        vcov_hc0 = np.zeros((H + 1, k, k)) if vcov == "HC0" else None
        vcov_hc1 = np.zeros((H + 1, k, k)) if vcov == "HC1" else None
        vcov_hc3 = np.zeros((H + 1, k, k)) if vcov == "HC3" else None
        vcov_hac = np.zeros((H + 1, k, k)) if vcov == "HAC" else None

        for h in range(H + 1):
            tx = X[: (T - h), :]
            ty = Y[h:]

            if mask is not None:
                if mask_type == "x":
                    tm = mask[: (T - h)]
                elif mask_type == "y":
                    tm = mask[h:]
                else:
                    mask = mask | np.roll(mask, -1)
                    tm = mask[: (T - h)]
                tx = tx[~tm, :]
                ty = ty[~tm]

            xpx_inv = np.linalg.inv(tx.transpose() @ tx)
            est_mat = xpx_inv @ tx.transpose()
            tc = est_mat @ ty
            coef[h, :] = tc

            if vcov is not None:
                t, k = tx.shape
                resid = ty - tx @ tc
                if vcov == "HC0":
                    vcov_hc0[h, :, :] = np.sum(np.square(resid)) / (t - k) * xpx_inv
                if vcov == "HC1":
                    vcov_hc1[h, :, :] = (
                        (t / (t - k))
                        * np.einsum("ij,j->ij", est_mat, np.square(resid))
                        @ np.transpose(est_mat)
                    )
                elif vcov == "HC3":
                    ktt = np.diag(tx @ est_mat)
                    gamma = est_mat @ resid
                    u_star = np.square(resid / (1 - ktt))
                    vcov_hc3[h, :, :] = ((t - 1) / t) * np.einsum(
                        "jk,k->jk", est_mat, u_star
                    ) @ np.transpose(est_mat) - ((t - 1) / t**2) * gamma @ np.transpose(
                        gamma
                    )
                elif vcov == "HAC":
                    vcov_hac[h, :, :] = LocalProjection._vcov_nw(
                        tx, xpx_inv, resid, h + 1
                    )

        return coef, vcov_hc0, vcov_hc1, vcov_hc3, vcov_hac

    @staticmethod
    def _vcov_nw(X: np.ndarray, XpX_inv: np.ndarray, resid: np.ndarray, lags: int):
        T, k = X.shape

        Xu = np.einsum("ij,i->ij", X, resid)
        # First estimate the long run variance
        lrv = np.zeros((k, k))
        for j in range(lags + 1):
            temp = Xu[j:, :].transpose() @ Xu[: (T - j), :]
            w = 1 - j / (lags + 1)
            if j == 0:
                lrv += temp
            else:
                lrv += w * (temp + temp.transpose())
        vcov = XpX_inv @ lrv @ XpX_inv
        vcov = (T / (T - k - 1)) * vcov
        return vcov

    @staticmethod
    def _estimation_sm(
        Y: np.ndarray,
        X: np.ndarray,
        H: int,
        vcov: str,
        mask: np.ndarray | None,
        mask_type: str = "both",
    ):
        T, k = X.shape
        coef = np.zeros((H + 1, k))
        vcov_hc0 = np.zeros((H + 1, k, k)) if vcov == "HC0" else None
        vcov_hc1 = np.zeros((H + 1, k, k)) if vcov == "HC1" else None
        vcov_hc3 = np.zeros((H + 1, k, k)) if vcov == "HC3" else None
        vcov_hac = np.zeros((H + 1, k, k)) if vcov == "HAC" else None

        for h in range(H + 1):
            tx = X[: (T - h), :]
            ty = Y[h:]

            if mask is not None:
                if mask_type == "x":
                    tx = tx[~mask[: (T - h)]]
                    ty = ty[~mask[: (T - h)]]
                elif mask_type == "y":
                    tx = tx[~mask[h:]]
                    ty = ty[~mask[h:]]
                elif mask_type == "both":
                    mask = mask | np.roll(mask, -1)
                    tx = tx[~mask[: (T - h)]]
                    ty = ty[~mask[: (T - h)]]

            tmod = sm.OLS(ty, tx).fit(cov_type=vcov if vcov != "HAC" else "HC0")
            coef[h, :] = tmod.params
            if vcov == "HC0":
                vcov_hc0[h, :, :] = tmod.cov_HC0
            elif vcov == "HC1":
                vcov_hc1[h, :, :] = tmod.cov_HC1
            elif vcov == "HC3":
                vcov_hc3[h, :, :] = tmod.cov_HC3
            else:
                vcov_hac[h, :, :] = sm.stats.cov_hac(tmod, nlags=h + 1)

        return coef, vcov_hc0, vcov_hc1, vcov_hc3, vcov_hac


class LocalProjectionResult(NamedTuple):
    endog: np.ndarray
    exog: np.ndarray
    shock: np.ndarray

    H: int

    coefs: np.ndarray
    vcov_HC0: np.ndarray | None
    vcov_HC1: np.ndarray | None
    vcov_HC3: np.ndarray | None
    vcov_HAC: np.ndarray | None

    @property
    def irf(self):
        return self.coefs[:, 0]

    @property
    def se(self):
        if self.vcov_HC0 is not None:
            return np.sqrt(self.vcov_HC0[:, 0, 0])
        elif self.vcov_HC1 is not None:
            return np.sqrt(self.vcov_HC1[:, 0, 0])
        elif self.vcov_HC3 is not None:
            return np.sqrt(self.vcov_HC3[:, 0, 0])
        elif self.vcov_HAC is not None:
            return np.sqrt(self.vcov_HAC[:, 0, 0])
        else:
            raise ValueError("No variance precomputed.")

    @property
    def se_hc0(self):
        if self.vcov_HC0 is not None:
            return np.sqrt(self.vcov_HC0[:, 0, 0])
        return None

    @property
    def se_hc1(self):
        if self.vcov_HC1 is not None:
            return np.sqrt(self.vcov_HC1[:, 0, 0])
        return None

    @property
    def se_hc3(self):
        if self.vcov_HC3 is not None:
            return np.sqrt(self.vcov_HC3[:, 0, 0])
        return None

    @property
    def se_hac(self):
        if self.vcov_HAC is not None:
            return np.sqrt(self.vcov_HAC[:, 0, 0])
        return None
