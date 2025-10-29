import numpy as np
from statsmodels import api as sm

from smoothlp.models import AbstractLocalProjection, AbstractLocalProjectionResult


class LocalProjection(AbstractLocalProjection):
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

    def fit(self, vcov="HC3", engine="np") -> "LocalProjectionResult":
        assert engine in ("np", "sm"), f"Engine '{engine}' not recognised."
        assert vcov in ("non-robust", "HC1", "HC3", "HAC"), (
            f"vcov '{vcov}' not recognised."
        )

        if self.exog is not None:
            X = np.concatenate((self.shock.reshape(-1, 1), self.exog), axis=1)
        else:
            X = self.shock.reshape((-1, 1))

        if engine == "np":
            engine_func = LocalProjection._estimation_np
        else:
            engine_func = LocalProjection._estimation_sm
        coef, vcov_vals = engine_func(
            self.endog,
            X,
            self.H,
            vcov,
            self.mask,
            self.mask_type,
        )

        return LocalProjectionResult(self, coef, vcov_vals, vcov)

    @staticmethod
    def _estimation_np(
        Y: np.ndarray,
        X: np.ndarray,
        H: int,
        vcov_type: str,
        mask: np.ndarray | None,
        mask_type: str = "both",
    ):
        T, k = X.shape
        coef = np.zeros((H + 1, k))
        vcov = np.zeros((H + 1, k, k))

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

            xpx_inv = np.linalg.inv(tx.T @ tx)
            est_mat = xpx_inv @ tx.T
            tc = est_mat @ ty
            coef[h, :] = tc

            if vcov_type is not None:
                t, k = tx.shape
                resid = ty - tx @ tc
                if vcov_type == "non-robust":
                    vcov[h, :, :] = np.sum(np.square(resid)) / (t - k) * xpx_inv
                if vcov_type == "HC1":
                    vcov[h, :, :] = (
                        (t / (t - k))
                        * np.einsum("ij,j->ij", est_mat, np.square(resid))
                        @ est_mat.T
                    )
                elif vcov_type == "HC3":
                    ktt = np.einsum("ij,ji->i", tx, est_mat)
                    u_star = np.square(resid / (1 - ktt))
                    vcov[h, :, :] = np.einsum("jk,k->jk", est_mat, u_star) @ est_mat.T
                elif vcov_type == "HAC":
                    vcov[h, :, :] = LocalProjection._vcov_nw(
                        tx,
                        xpx_inv,
                        resid,
                        h + 1,
                    )

        return coef, vcov

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
        vcov = (T / (T - k)) * vcov
        return vcov

    @staticmethod
    def _estimation_sm(
        Y: np.ndarray,
        X: np.ndarray,
        H: int,
        vcov_type: str,
        mask: np.ndarray | None,
        mask_type: str = "both",
    ):
        assert vcov_type in ("HC1", "HC3", "HAC")
        T, k = X.shape
        coef = np.zeros((H + 1, k))
        vcov = np.zeros((H + 1, k, k))

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

            tmod = sm.OLS(ty, tx).fit(
                cov_type=vcov_type if vcov_type != "HAC" else "HC0"
            )
            coef[h, :] = tmod.params
            if vcov_type == "HC1":
                vcov[h, :, :] = tmod.cov_HC1
            elif vcov_type == "HC3":
                vcov[h, :, :] = tmod.cov_HC3
            else:
                vcov[h, :, :] = sm.stats.cov_hac(tmod, nlags=h + 1)

        return coef, vcov


class LocalProjectionResult(AbstractLocalProjectionResult):
    pass
