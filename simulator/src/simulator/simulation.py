from typing import Callable

from tqdm import tqdm
from matplotlib import pyplot as plt
from box import Box, BoxList
import numpy as np
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
            kd, ks, ki = jrng.split(key, 3)
            data = self.data_component.generate(data_params, kd, self.n)
            shocks = self.shock_component.generate(
                shock_params, ks, self.n + 2 * self.h
            )
            irf = self.irf_component.generate(irf_params, ki, self.h)
            combined = linear_combination(self.n, self.h, data, irf, shocks)

            return combined, shocks[self.h : (self.n + self.h)], irf

        return sample

    def run(self, model_fn, n, key):
        sample = self.make_sampler()
        ests = []
        ses = []
        irfs = []
        keys = jrng.split(key, n)
        for i in tqdm(range(n), ncols=100):
            Y, s, irf = sample(keys[i])
            est, se = model_fn(Y, s, self.h)
            irfs.append(irf)
            ests.append(est)
            ses.append(se)
        ests = np.stack(ests, axis=0)
        irfs = np.stack(irfs, axis=0)
        ses = np.stack(ses, axis=0)
        t = ests / ses
        results = {
            "irf": irfs,
            "est": ests,
            "se": ses,
            "pointwise_t": t,
            "bias": np.mean(ests - irfs, axis=0),
            "mse": np.mean(np.square(ests - irfs), axis=0),
            "share_rejected_90": np.mean(np.abs(t) >= 1.64, axis=0),
            "share_rejected_95": np.mean(np.abs(t) >= 1.95, axis=0),
        }
        results = {k: np.array(v) for k, v in results.items()}
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
