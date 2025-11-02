from itertools import product
from functools import partial
from typing import Callable
import pathlib

from matplotlib import pyplot as plt
import numpy as np
from box import Box
from jax import random as jrng

from simulator.simulation import Simulation


class SimulationRunner:
    _conf: Box
    _model_fn: Callable

    def __init__(self, config: Box, model_fn: Callable):
        self._conf = config
        self._model_fn = model_fn

    def generate(self):
        conf = self._conf

        configs = {}
        kwargs = conf.model_kwargs if "model_kwargs" in conf else [{"": {}}]
        combinations = list(product(conf.data, conf.irf, conf.shock, kwargs))

        for comb in combinations:
            sc = Box(n=conf.n, h=conf.h, combination=conf.combination)

            data_c, irf_c, shock_c, kwargs = comb
            data_n = list(data_c.keys())[0]
            irf_n = list(irf_c.keys())[0]
            shock_n = list(shock_c.keys())[0]
            kwargs_n = list(kwargs.keys())[0]

            name = f"{data_n}-{irf_n}-{shock_n}"

            sc["data"] = data_c[data_n]
            sc["irf"] = irf_c[irf_n]
            sc["shock"] = shock_c[shock_n]

            if kwargs_n != "":
                name += f"-{kwargs_n}"

            configs[name] = (sc, kwargs[kwargs_n])

        return configs

    def run(self, out_path: pathlib.Path, clean_out: bool = True):
        out_path.mkdir(exist_ok=True)
        if clean_out:
            for f in out_path.glob("*"):
                f.unlink()

        fixed_kwargs = self._conf.get("fixed_kwargs", {})

        key = jrng.key(self._conf.key)
        configs = self.generate()
        total_iter = len(configs)

        for i, (name, conf) in enumerate(configs.items()):
            print(f"Iteration {i + 1}/{total_iter}, config {name}.")
            sim = Simulation(conf[0])
            kwargs = fixed_kwargs | conf[1]
            _mod_fn = partial(self._model_fn, **kwargs)
            res = sim.run(_mod_fn, self._conf.n_runs, key)
            fig = sim.plot(res)
            fig.savefig(out_path / f"{name}.png")
            plt.close(fig)
            np.save(out_path / name, res)
