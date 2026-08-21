import jax

jax.config.update("jax_enable_x64", True)

from smoothlp.models import (
    AbstractLocalProjection as AbstractLocalProjection,
    AbstractLocalProjectionResult as AbstractLocalProjectionResult,
)
from smoothlp import slp as slp
from smoothlp import lp as lp
