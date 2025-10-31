from itertools import product
from functools import partial
from typing import Callable
import pathlib

from box import Box
from jax import random as jrng

from simulator.simulation import Simulation


class SimulationRunner:
    _conf: Box
    _model_fn: Callable
    _model_fn_args: dict | None

    def __init__(
        self, config: Box, model_fn: Callable, model_fn_args: dict | None = None
    ):
        self._conf = config
        self._model_fn = model_fn
        self._model_fn_args = model_fn_args

    def generate(self):
        conf = self._conf

        configs = {}
        args = self._model_fn_args if self._model_fn_args is not None else [(None, {})]
        combinations = product(conf.data, conf.irf, conf.shock, args)

        for comb in combinations:
            sc = Box(n=conf.n, h=conf.h, combination=conf.combination[0])

            data_c, irf_c, shock_c, arg = comb
            data_n = list(data_c.keys())[0]
            irf_n = list(irf_c.keys())[0]
            shock_n = list(shock_c.keys())[0]

            name = f"{data_n}_{irf_n}_{shock_n}"

            sc["data"] = data_c[data_n]
            sc["irf"] = irf_c[irf_n]
            sc["shock"] = shock_c[shock_n]

            if arg is not None:
                arg_n = arg[0]
                name += f"_{arg_n}"

            configs[name] = (sc, arg[1])

        return configs

    def run(self, out_path: pathlib.Path):
        out_path.mkdir(exist_ok=True)

        key = jrng.key(self._conf.key)
        configs = self.generate()
        total_iter = len(configs)

        for i, (name, conf) in enumerate(configs.items()):
            print(f"Iteration {i + 1}/{total_iter}, config {name}.")
            sim = Simulation(conf[0])
            _mod_fn = partial(self._model_fn, **conf[1])
            res = sim.run(_mod_fn, self._conf.n_runs, key)
            fig = sim.plot(res)
            fig.savefig(out_path / f"{name}.png")
