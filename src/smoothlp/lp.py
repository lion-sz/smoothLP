from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import numpy as np
from jax import numpy as jnp, Array
from statsmodels import api as sm
from jaxtyping import Float

from smoothlp.utils import comp_lrv, build_stacked, partial_out, invert
from smoothlp.models import AbstractLocalProjection, AbstractLocalProjectionResult


@dataclass(init=False)
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
        """Fit the local projection and return a :class:`LocalProjectionResult`

        All engines return only shock coefficients and their covariance block.

        Args:
            vcov (str): The variance-covariance estimator.
                Must be one of `non-robust`, `HC1`, `HC3` (default), or `HAC`.
            engine (str): The engine to use.
                Must be one of `np` (default), `sm` and `jax`.
                Jax can run on a GPU, which can be a lot faster, but also runs only
                with 32bits, so might lose a lot of precision.
                Don't use this for main estimations.
        """
        assert engine in ("np", "sm", "jax"), f"Engine '{engine}' not recognised."
        assert vcov in ("non-robust", "HC1", "HC3", "HAC"), (
            f"vcov '{vcov}' not recognised."
        )

        if engine == "np":
            engine_func = LocalProjection._estimation_np
        elif engine == "jax":
            engine_func = LocalProjection._estimation_jax
        else:
            engine_func = LocalProjection._estimation_sm

        inclusion_mask = None if self.mask is None else (~self.mask).astype(float)
        coef, vcov_vals = engine_func(
            self.endog,
            self.shock,
            self.exog,
            self.H,
            vcov,
            self.diff,
            inclusion_mask,
            self.mask_type,
        )

        return LocalProjectionResult(self, coef, vcov_vals, vcov, engine)

    @staticmethod
    @partial(jax.jit, static_argnames=["H", "vcov_type", "mask_type"])
    def _estimation_jax(Y, shock, exog, H, vcov_type, diff, mask, mask_type):
        return LocalProjection._estimation(
            "jax", Y, shock, exog, H, vcov_type, diff, mask, mask_type
        )

    @staticmethod
    def _estimation_np(Y, shock, exog, H, vcov_type, diff, mask, mask_type):
        return LocalProjection._estimation(
            "np", Y, shock, exog, H, vcov_type, diff, mask, mask_type
        )

    @staticmethod
    def _estimation(
        engine: Literal["jax", "np"],
        Y: Float[Array, " T"],
        shock: Float[Array, "T s"],
        exog: Float[Array, "T k"],
        H: int,
        vcov_type: str,
        diff: np.ndarray | None,
        mask: Float[Array, " T"] | None,
        mask_type: str = "both",
    ):
        T = Y.shape[0]
        if exog is None:
            k = 0
        else:
            k = exog.shape[1]
        s = shock.shape[1]

        if engine == "jax":
            lib = jnp
        elif engine == "np":
            lib = np
        else:
            raise ValueError("Can only use jax or np as engine.")

        # Build the stacked arrays
        if mask is None:
            mask = lib.ones(T)
        Ys, mask_s = build_stacked(engine, Y, mask, H, mask_type)
        if diff is not None:
            Ys = Ys - diff[:, lib.newaxis]
        Th = lib.sum(mask_s, axis=0)

        # Partial out the exogenous
        Ys_r_m, shock_r_m, exog_leverage = partial_out(
            engine, Ys, shock, exog, mask_s, leverage=vcov_type == "HC3"
        )

        # Main estimation
        XpX = lib.einsum("tih,tjh -> ijh", shock_r_m, shock_r_m)
        XpX_inv = invert(engine, XpX, in_axes=2)
        # XpX_inv = jax.vmap(partial(invert, engine), in_axes=2, out_axes=2)(XpX)
        beta = lib.einsum("ijh, tih, th -> jh", XpX_inv, shock_r_m, Ys_r_m)
        yhat = lib.einsum("tih, ih -> th", shock_r_m, beta)
        resid = Ys_r_m - yhat

        # Compute the standard errors
        vcov = None
        scale = (Th / (Th - k - s)).reshape(1, 1, H + 1)
        if vcov_type == "non-robust":
            sig_sqr = lib.sum(lib.square(resid), axis=0) / (Th - k - s)
            vcov = sig_sqr * XpX_inv
        elif vcov_type == "HC1":
            resid_sqr = lib.square(resid)
            inner = lib.einsum(
                "ikh, tkh, th, tlh, ljh -> ijh",
                XpX_inv,
                shock_r_m,
                resid_sqr,
                shock_r_m,
                XpX_inv,
            )
            vcov = scale * inner
        elif vcov_type == "HC3":
            sl = lib.einsum("tih, ijh, tjh -> th", shock_r_m, XpX_inv, shock_r_m)
            if exog_leverage is not None:
                ktt = sl + exog_leverage
            else:
                ktt = sl
            u_star = lib.square(resid / (1 - ktt))
            vcov = lib.einsum(
                "ikh, tkh, th, tlh, jlh -> ijh",
                XpX_inv,
                shock_r_m,
                u_star,
                shock_r_m,
                XpX_inv,
            )
        if vcov_type == "HAC":
            Xu = shock_r_m * resid[:, jnp.newaxis, :]
            lrv = lib.stack(
                [comp_lrv(Xu[:, :, h], h + 1) for h in range(H + 1)], axis=2
            )
            vcov = lib.einsum("ikh, klh, ljh -> ijh", XpX_inv, lrv, XpX_inv) * scale
        return beta.T, lib.moveaxis(vcov, -1, 0) if vcov is not None else None

    @staticmethod
    def _estimation_sm(
        Y: np.ndarray,
        shock: np.ndarray,
        exog: np.ndarray | None,
        H: int,
        vcov_type: str,
        diff: np.ndarray | None,
        mask: np.ndarray | None,
        mask_type: str = "both",
    ):
        mask = None if mask is None else mask == 1.0
        assert vcov_type in ("HC1", "HC3", "HAC")
        if exog is not None:
            X = np.concatenate((shock, exog), axis=1)
        else:
            X = shock
        T, k = X.shape
        coef = np.zeros((H + 1, k))
        vcov = np.zeros((H + 1, k, k))

        for h in range(H + 1):
            tx = X[: (T - h), :]
            if diff is None:
                ty = Y[h:]
            else:
                ty = Y[h:] - diff[: (T - h)]

            if mask is not None:
                if mask_type == "x":
                    tx = tx[mask[: (T - h)]]
                    ty = ty[mask[: (T - h)]]
                elif mask_type == "y":
                    tx = tx[mask[h:]]
                    ty = ty[mask[h:]]
                elif mask_type == "both":
                    if h > 0:
                        mask = mask & np.roll(mask, -1)
                    tx = tx[mask[: (T - h)]]
                    ty = ty[mask[: (T - h)]]

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

        n_shocks = shock.shape[1]
        return coef[:, :n_shocks], vcov[:, :n_shocks, :n_shocks]


@dataclass
class LocalProjectionResult(AbstractLocalProjectionResult):
    pass
