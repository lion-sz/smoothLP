from jax import Array, lax, random as jrng
from chex import dataclass

from simulator.components import (
    SimpleAbstractComponent,
    component_map,
    ChainedAbstractComponent,
)


@dataclass
class GaussianNoise(SimpleAbstractComponent):
    mu: float = 0.0
    sigma: float = 0.1

    @staticmethod
    def generate(params: "GaussianNoise", key: Array, n: int) -> Array:
        return params.mu + params.sigma * jrng.normal(key, (n,))


class AR1(ChainedAbstractComponent):
    rho: float

    single_component = True
    has_params = True

    def _get_params(self, sc_params):
        return (sc_params, self.rho)

    @staticmethod
    def _generate(sc_generator, params, key, n):
        sc_params, rho = params
        shocks = sc_generator(sc_params, key, n + 100)
        _, shock = lax.scan(lambda x, eps: (rho * x + eps, rho * x + eps), 0.0, shocks)
        shocks = shocks[100:]
        return shocks


component_map["GaussianNoise"] = GaussianNoise
component_map["AR1"] = AR1
