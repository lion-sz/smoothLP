import numpy as np


class AbstractLocalProjection:
    endog: np.ndarray
    shock: np.ndarray
    exog: np.ndarray | None

    H: int

    mask: np.ndarray | None
    mask_type: str

    def __init__(
        self,
        endog: np.ndarray,
        shock: np.ndarray,
        exog: np.ndarray,
        H: int,
        mask: np.ndarray | None = None,
        mask_type: str = "both",
    ):
        assert endog.ndim == 1, "Outcome must be one dimensional."
        assert shock.ndim == 1, "Shock must be one dimensional."
        assert exog is None or exog.ndim == 2, "Endog must be two dimensional."
        assert mask is None or mask.ndim == 1, "Mask must be one dimensional."
        T = endog.shape[0]
        assert shock.shape[0] == T, "Shock and endog must have same length."
        assert exog is None or exog.shape[0] == T, (
            "Exog and endog must have same length."
        )
        assert mask is None or mask.shape[0] == T, (
            "Mask and endog must have same length."
        )
        assert mask_type in ("x", "y", "both")

        self.endog = endog
        self.shock = shock
        self.exog = exog

        self.H = H

        self.mask = mask
        self.mask_type = mask_type

    def fit(self, vcov_type: str | None) -> "AbstractLocalProjectionResult":
        raise NotImplementedError()


class AbstractLocalProjectionResult:
    model: AbstractLocalProjection

    coef: np.ndarray
    vcov: np.ndarray | None
    vcov_type: str | None

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
        return self.coef[:, 0]

    @property
    def se(self):
        return np.sqrt(self.vcov[:, 0, 0])
