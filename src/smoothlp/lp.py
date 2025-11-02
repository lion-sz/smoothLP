from functools import partial

import numpy as np
from statsmodels import api as sm
from scipy import linalg as sla
from line_profiler import profile
import jax
from jax import numpy as jnp, lax

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
        assert engine in ("np", "sm", "jax"), f"Engine '{engine}' not recognised."
        assert vcov in ("non-robust", "HC1", "HC3", "HAC"), (
            f"vcov '{vcov}' not recognised."
        )

        if self.exog is not None:
            X = np.concatenate((self.shock.reshape(-1, 1), self.exog), axis=1)
        else:
            X = self.shock.reshape((-1, 1))

        if engine == "np":
            engine_func = LocalProjection._estimation_np
        elif engine == "jax":
            assert self.mask is None, "Mask is not yet supported in jax engine."
            engine_func = LocalProjection._estimation_jax
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
        if engine == "jax":
            coef, vcov_vals = np.array(coef), np.array(vcov_vals)

        return LocalProjectionResult(self, coef, vcov_vals, vcov)

    @staticmethod
    @profile
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

            U = sla.cholesky(tx.T @ tx)
            Uinv = np.linalg.inv(U)
            XpX_inv = Uinv @ Uinv.T
            est_mat = XpX_inv @ tx.T
            tc = est_mat @ ty
            coef[h, :] = tc

            if vcov_type is not None:
                t, k = tx.shape
                resid = ty - tx @ tc
                if vcov_type == "non-robust":
                    vcov[h, :, :] = np.sum(np.square(resid)) / (t - k) * XpX_inv
                elif vcov_type == "HC1":
                    vcov[h, :, :] = (
                        (t / (t - k))
                        * (est_mat * np.square(resid).reshape((1, t)))
                        @ est_mat.T
                    )
                elif vcov_type == "HC3":
                    ktt = np.einsum("ij,ji->i", X, est_mat)
                    u_star = np.square(resid / (1 - ktt)).reshape((1, t))
                    vcov[h, :, :] = (est_mat * u_star) @ est_mat.T
                elif vcov_type == "HAC":
                    vcov[h, :, :] = LocalProjection._vcov_nw(
                        tx,
                        XpX_inv,
                        resid,
                        h + 1,
                    )
                else:
                    raise ValueError(f"Strange vcov encountered: '{vcov_type}'")

        return coef, vcov

    @staticmethod
    @profile
    def _vcov_nw(X: np.ndarray, XpX_inv: np.ndarray, resid: np.ndarray, lags: int):
        T, k = X.shape

        Xu = X * resid[:, np.newaxis]
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
    @partial(jax.jit, static_argnames=["H", "vcov_type", "mask_type"])
    def _estimation_jax(
        Y: np.ndarray,
        X: np.ndarray,
        H: int,
        vcov_type: str,
        mask: np.ndarray | None,
        mask_type: str = "both",
    ):
        T, k = X.shape
        XpX = X.T @ X
        XpX_end = jnp.einsum("ti,tj->tij", X[(-H - 1) :, :], X[(-H - 1) :, :])

        Y = jnp.concat([Y, np.zeros((H,))])
        mask = jnp.ones((T,))

        def inner(h):
            sub_x = jnp.arange(H + 1) >= H + 1 - h
            tmask = mask.at[-(H + 1) :].set((~sub_x).astype(float))
            tXpX = XpX - jnp.sum(XpX_end * sub_x.reshape((H + 1, 1, 1)), axis=0)
            # Iteration
            U = jax.scipy.linalg.cholesky(tXpX)
            Uinv = jnp.linalg.inv(U)
            XpX_inv = Uinv @ Uinv.T
            est_mat = XpX_inv @ X.T
            ty = lax.dynamic_slice_in_dim(Y, h, T)
            beta = est_mat @ ty

            XpX_inv_coll = None
            resid_coll = None
            vcov = None
            if vcov_type is not None:
                resid = tmask * (ty - X @ beta)
                if vcov_type == "non-robust":
                    sig_sqr = jnp.sum(jnp.square(resid)) / (T - h - k)
                    vcov = sig_sqr * XpX_inv
                elif vcov_type == "HC1":
                    r = jnp.square(resid).reshape((1, T))
                    vcov = ((T - h) / (T - h - k)) * (est_mat * r) @ est_mat.T
                elif vcov_type == "HC3":
                    ktt = jnp.einsum("ij,ji->i", X, est_mat)
                    u_star = jnp.square(resid / (1 - ktt)).reshape((1, T))
                    vcov = (est_mat * u_star) @ est_mat.T
                else:
                    XpX_inv_coll = XpX_inv
                    resid_coll = resid

            return beta, vcov, XpX_inv_coll, resid_coll

        (coef, vcov, XpX_inv_coll, resid_coll) = jax.vmap(inner)(jnp.arange(0, H + 1))

        if vcov_type == "HAC":
            lrv = LocalProjection._lrv_jax(X, resid_coll)
            scale = ((T - jnp.arange(H + 1)) / (T - jnp.arange(H + 1) - k)).reshape(
                (H + 1, 1, 1)
            )
            vcov = XpX_inv_coll @ lrv @ XpX_inv_coll * scale
        return coef, vcov

    @staticmethod
    @jax.jit
    def _lrv_jax(X, resid):
        T, k = X.shape
        H = resid.shape[0] - 1

        @partial(jax.vmap, in_axes=(0, None))
        def inner(resid, h):
            Xu = X * resid.reshape((T, 1))
            Xu_filled = jnp.concat([np.zeros((H + 1, k)), Xu])
            gamma = Xu.T @ lax.dynamic_slice_in_dim(Xu_filled, H + 1 - h, T)
            return gamma + gamma.T * (h > 0)

        _, gammas = lax.scan(
            lambda c, h: (None, inner(resid, h)), None, np.arange(H + 2)
        )

        hs = jnp.arange(H + 1)
        ls = jnp.arange(H + 2)
        r = jnp.meshgrid(ls, hs, indexing="ij")
        f = jax.vmap(lambda lag, h: jnp.maximum(0, 1 - lag / (h + 2)))
        w = f(*r).reshape((H + 2, H + 1, 1, 1))
        lrv = jnp.sum(gammas * w, axis=0)
        return lrv

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
