from typing import Callable

from tqdm import tqdm
from matplotlib import pyplot as plt
from box import Box, BoxList
import jax
from jax import Array, numpy as jnp, random as jrng

from simulator.components import AbstractComponent, component_map
from simulator.irf import linear_combination


class Simulation:
    conf: Box

    n: int
    h: int
    data_component: AbstractComponent
    shock_component: AbstractComponent
    irf_component: AbstractComponent
    combination: Callable

    def __init__(self, config: Box):
        self.conf = config

        self.n = config.n
        self.h = config.h

        self.data_component = Simulation._parse_components(config.data)
        self.shock_component = Simulation._parse_components(config.shock)
        self.irf_component = Simulation._parse_components(config.irf)

        if config.combination == "linear_combination":
            self.combination = linear_combination
        else:
            raise NotImplementedError(
                f"Combination '{config.combination}' not implemented."
            )

    @staticmethod
    def _parse_components(config):
        assert len(config) == 1, f"Cannot parse multiple components, {config}"
        name = list(config.keys())[0]
        conf = config[name]
        if conf is not None and "sc" in conf:
            if isinstance(conf.sc, Box):
                sc = Simulation._parse_components(conf.sc)
            elif isinstance(conf.sc, BoxList):
                sc = tuple(Simulation._parse_components(c) for c in conf.sc)
            else:
                raise ValueError(f"Subcomponents {conf.sc} not understood.")
            c = conf.copy()
            del c["sc"]
            component = component_map[name].build(sc, **c)
        else:
            component = component_map[name].build(**(conf if conf is not None else {}))
        return component

    def make_sampler(self):
        data_params = self.data_component.get_params()
        shock_params = self.shock_component.get_params()
        irf_params = self.irf_component.get_params()

        @jax.jit
        def sample(key: Array):
            data = self.data_component.generate(data_params, key, self.n)
            shocks = self.shock_component.generate(
                shock_params, key, self.n + 2 * self.h
            )
            irf = self.irf_component.generate(irf_params, key, self.h)
            combined = linear_combination(self.n, self.h, data, irf, shocks)

            return combined, shocks[self.h : (self.n + self.h)], irf

        return sample

    def run(self, model_fn, n, key):
        sample = self.make_sampler()
        err = []
        sqr_err = []
        pointwise_t = []
        keys = jrng.split(key, n)
        for i in tqdm(range(n), ncols=100):
            Y, s, irf = sample(keys[i])
            est, se = model_fn(Y, s, self.h)
            # Compute statistics
            t_err = est - irf
            err.append(t_err)
            sqr_err.append(jnp.square(t_err))
            pointwise_t.append(t_err / se)
        err = jnp.stack(err, axis=-1)
        sqr_err = jnp.stack(sqr_err, axis=-1)
        pointwise_t = jnp.stack(pointwise_t, axis=-1)
        results = {"err": err, "sqr_err": sqr_err, "pointwise_t": pointwise_t}
        results["bias"] = jnp.mean(err, axis=1)
        results["mse"] = jnp.mean(sqr_err, axis=1)
        results["share_rejected_90"] = jnp.mean(jnp.abs(pointwise_t) >= 1.64, axis=1)
        results["share_rejected_95"] = jnp.mean(jnp.abs(pointwise_t) >= 1.96, axis=1)
        return results

    @staticmethod
    def plot(results):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        plt.sca(axes[0])
        plt.plot(results["bias"], label="bias")
        plt.plot(jnp.sqrt(results["mse"]), label="mse")
        plt.axhline(0.0, linewidth=1, linestyle="--", color="black")
        plt.legend(loc="upper left")

        plt.sca(axes[1])
        plt.plot(results["share_rejected_90"], label="rej 90")
        plt.axhline(y=0.1, linewidth=1, linestyle="--", color="black")
        plt.plot(results["share_rejected_95"], label="rej 95")
        plt.axhline(y=0.05, linewidth=1, linestyle="--", color="black")
        plt.legend(loc="upper right")
        plt.tight_layout()

        return fig
