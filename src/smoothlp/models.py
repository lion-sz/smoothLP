from dataclasses import dataclass

import numpy as np
from jax import numpy as jnp


@dataclass(init=True)
class AbstractLocalProjection:
    """Baseclass for all Local Projection Methods."""

    endog: np.ndarray
    shock: np.ndarray
    exog: np.ndarray | None

    H: int

    mask: np.ndarray | None
    mask_type: str
    diff: np.ndarray | None

    def __init__(
        self,
        endog: np.ndarray,
        shock: np.ndarray,
        exog: np.ndarray,
        H: int,
        mask: np.ndarray | None = None,
        mask_type: str = "both",
        diff: np.ndarray | None = None,
    ):
        """These are the shared kwargs.

        Args:
            endog (np.ndarray, 1d): The endogenous variable.
            shock (np.ndarray, 1d or 2d): The shock variable(s).
                Can be 1D (single shock) or 2D (multiple shocks).
            exog (np.ndarray, 2d): The exogenous variables.
            H (int): The local projection horizon.
            mask (np.ndarray, 1d): Boolean mask; True omits an observation.
            mask_type (str): The mask type to use.
                Must be one of `x`, `y`, or `both` (default).
            diff (np.ndarray, 1d): Optional differencing term aligned with X_t.
                If provided, horizon h uses `endog[t+h] - diff[t]` as outcome.
        """
        assert endog.ndim == 1, "Outcome must be one dimensional."
        assert shock.ndim in (1, 2), "Shock must be one or two dimensional."
        assert exog is None or exog.ndim == 2, "Endog must be two dimensional."
        assert diff is None or diff.ndim == 1, "Diff must be one dimensional."
        assert mask is None or mask.ndim == 1, "Mask must be one dimensional."
        T = endog.shape[0]
        assert shock.shape[0] == T, "Shock and endog must have same length."
        assert exog is None or exog.shape[0] == T, (
            "Exog and endog must have same length."
        )
        assert diff is None or diff.shape[0] == T, (
            "Diff and endog must have same length."
        )
        assert mask is None or mask.shape[0] == T, (
            "Mask and endog must have same length."
        )
        assert mask_type in ("x", "y", "both")

        self.endog = endog
        # Cast shock to 2D internally
        self.shock = shock.reshape(-1, 1) if shock.ndim == 1 else shock
        self.exog = exog

        self.H = H

        self.mask = mask
        self.mask_type = mask_type
        self.diff = diff

    def fit(self, vcov: str | None) -> "AbstractLocalProjectionResult":
        raise NotImplementedError()


@dataclass
class AbstractLocalProjectionResult:
    """Result container for a local projection.

    Attributes:
        model (AbstractLocalProjection): The model estimated.
        irf (np.ndarray): The estimated impulse response.
        se (np.ndarray): The standard error of the impulse response.
        coef (np.ndarray): The estimated coefficients.
        vcov (np.ndarray): The estimated covariance matrix.
        vcov_type (str): The type of the covariance matrix.
    """

    model: AbstractLocalProjection

    coef: np.ndarray
    vcov: np.ndarray | None
    vcov_type: str | None
    _engine: str = "np"

    def __init__(
        self,
        model: AbstractLocalProjection,
        coef: np.ndarray,
        vcov: np.ndarray | None,
        vcov_type: str | None,
    ):
        self.model = model
        self.coef = coef
        self.vcov = vcov
        self.vcov_type = vcov_type

    @property
    def irf(self):
        # Return IRFs for all shocks (first n_shocks columns)
        n_shocks = self.model.shock.shape[1]
        if n_shocks == 1:
            return self.coef[:, 0]
        else:
            return self.coef[:, :n_shocks]

    @property
    def se(self):
        # Return standard errors for all shocks
        assert self.vcov is not None, "vcov is not available"
        n_shocks = self.model.shock.shape[1]
        lib = jnp if self._engine == "jax" else np
        if n_shocks == 1:
            return lib.sqrt(self.vcov[:, 0, 0])
        else:
            return lib.sqrt(
                lib.stack([self.vcov[:, i, i] for i in range(n_shocks)], axis=1)
            )
