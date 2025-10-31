from jax import Array, numpy as jnp, random as jrng
from chex import dataclass

from simulator.components import component_map, SimpleAbstractComponent


@dataclass
class SinPeriodicity(SimpleAbstractComponent):
    weekly: float
    monthly: float
    annual: float

    random_shifts: bool

    @staticmethod
    def generate(params: "SinPeriodicity", key: Array, n: int) -> Array:
        x = jnp.arange(n).reshape((1, n))
        shifts = (jrng.uniform(key, (3, 1)) * 2 - 1) * jnp.pi * params.random_shifts
        p = 2 * jnp.pi / jnp.array([7.0, 30.42, 365.0]).reshape((3, 1))
        magnitudes = jnp.array([params.weekly, params.monthly, params.annual]).reshape(
            (3, 1)
        )
        data = jnp.sum(magnitudes * jnp.sin(x * p + shifts), axis=0)
        return data


component_map["SinPeriodicity"] = SinPeriodicity
