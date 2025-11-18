from functools import partial

import jax
from jax import Array, numpy as jnp, random as jrng, lax
from chex import dataclass

from simulator.components import SimpleAbstractComponent, component_map


@dataclass
class SinIRF(SimpleAbstractComponent):
    @staticmethod
    def generate(params: "SinIRF", key: Array, h: int):
        k1, k2 = jrng.split(key)
        magnitude = jrng.uniform(k1, (1,)) * 2 - 1
        shift = jrng.uniform(k2, (1,)) * h + 1
        irf = jnp.sin((shift + jnp.arange(h + 1)) * jnp.pi / (h + 1)) * magnitude
        return irf


@dataclass
class ZeroIRF(SimpleAbstractComponent):
    @staticmethod
    def generate(params: "ZeroIRF", key: Array, h: int):
        irf = jnp.zeros((h + 1,))
        return irf


@partial(jax.jit, static_argnames=("n", "h"))
def linear_combination(n: int, h: int, data: Array, irf: Array, shock: Array):
    # First expand the irfs to the observed horizon

    def scan_body(carry, irf):
        res, shock = carry
        res += shock * irf
        shock = jnp.roll(shock, 1)
        return (res, shock), None

    init = jnp.zeros_like(shock)
    (response, _), _ = lax.scan(scan_body, (init, shock), irf)

    response = response[h : (n + h)]
    return data + response


component_map["SinIRF"] = SinIRF
component_map["ZeroIRF"] = ZeroIRF
