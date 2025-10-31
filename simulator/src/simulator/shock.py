from typing import Any

from jax import Array, numpy as jnp, random as jrng
from chex import dataclass

from simulator.components import (
    SimpleAbstractComponent,
    ChainedAbstractComponent,
    component_map,
)


@dataclass
class GaussianShock(SimpleAbstractComponent):
    mu: float = 0.0
    sigma: float = 1.0

    @staticmethod
    def generate(params: "GaussianShock", key: Array, n: int) -> Array:
        return params.mu + params.sigma * jrng.normal(key, (n,))


@dataclass
class SparseGaussianShockParams:
    sc_params: Any
    freq: int
    random_start: int
    random_offset: int


class SparseShock(ChainedAbstractComponent):
    params_class: SparseGaussianShockParams
    single_component = True

    freq: int = 60
    random_start: int = 60
    random_offset: int = 5

    def _get_params(self, sc_params):
        return SparseGaussianShockParams(
            sc_params=sc_params,
            freq=self.freq,
            random_start=self.random_start,
            random_offset=self.random_offset,
        )

    @staticmethod
    # @partial(jax.jit, static_argnames=["sub_generator", "n"])
    def _generate(sub_generator, params, key: Array, n: int) -> Array:
        k1, k2, k3 = jrng.split(key, 3)
        shock = sub_generator(params.sc_params, k1, n)
        to_keep = jnp.arange(n) + jrng.randint(k2, (1,), 0, params.random_start)
        to_keep = to_keep + jrng.randint(
            k3, (n,), -params.random_offset, +params.random_offset
        )
        shock = jnp.where(to_keep % params.freq == 0, shock, 0.0)
        return shock


component_map["GaussianShock"] = GaussianShock
component_map["SparseShock"] = SparseShock
