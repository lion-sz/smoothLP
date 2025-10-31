from typing import NamedTuple, Union, Callable
from functools import partial

from jax import Array, random as jrng, numpy as jnp


class ChainedAbstractComponent:
    subcomponent: "AbstractComponent | list[AbstractComponent]"
    single_component: bool
    has_params: bool = True

    def __init__(self, subcomponents, **kwargs):
        if self.single_component:
            assert not isinstance(subcomponents, (list, tuple)), (
                "Only one subcomponent supported."
            )
        self.subcomponent = subcomponents
        for k, v in kwargs.items():
            assert k in self.__annotations__, f"Canont set {k}"
            self.__dict__[k] = v

    @classmethod
    def build(cls, subcomponents, **kwargs) -> NamedTuple:
        instance = cls(subcomponents, **kwargs)
        if instance.single_component:
            instance.generate = partial(
                instance._generate, instance.subcomponent.generate
            )
        else:
            instance.generate = partial(
                instance._generate, tuple(sc.generate for sc in subcomponents)
            )
        return instance

    def get_params(self) -> NamedTuple:
        if self.single_component:
            sc_params = self.subcomponent.get_params()
        else:
            sc_params = tuple(sc.get_params() for sc in self.subcomponent)
        if self.has_params:
            return self._get_params(sc_params)
        else:
            return sc_params

    def _get_params(self):
        raise NotImplementedError()

    @staticmethod
    def _generate(sc_generator: Callable, params: NamedTuple, key: Array, n: int):
        """Will be assigned in the build function, using the subcomponent."""
        raise NotImplementedError()


class SimpleAbstractComponent:
    @classmethod
    def build(cls, *args, **kwargs) -> "SimpleAbstractComponent":
        return cls(*args, **kwargs)

    def get_params(self) -> "SimpleAbstractComponent":
        return self

    @staticmethod
    def generate(params: "SimpleAbstractComponent", key: Array, n: int) -> Array:
        raise NotImplementedError()


AbstractComponent = Union[SimpleAbstractComponent, ChainedAbstractComponent]

component_map: dict[str, AbstractComponent] = {}


class LinearCombination(ChainedAbstractComponent):
    single_component = False
    has_params = False

    @staticmethod
    # @partial(jax.jit, static_argnames=["sub_generators", "n"])
    def _generate(
        sub_generators: list[Callable], params: NamedTuple, key: Array, n: int
    ):
        k = len(sub_generators)
        keys = jrng.split(key, k)
        outcome = jnp.zeros((n,))
        for i in range(k):
            outcome += sub_generators[i](params[i], keys[i], n)
        return outcome


component_map["LinearCombination"] = LinearCombination
