#!/usr/bin/env python3

""" Perform parameter-variation study on OpenFOAM cases

This script runs a paramter-variation study on OpenFOAM cases given a YAML/JSON config
file (Supported through Hydra, default: config.yaml).

Output: A JSON Snapshot of the experiment and CSV data for experiment trials

Notes:
- Use multiObjOpt.py for optimization studies
- Parameters are sampled in a quasi-random fashion using SOBOL. If you want
  finer control over parameter values, convert parameters to choice type.

"""

import hydra, logging
from omegaconf import DictConfig, OmegaConf

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient

from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.core import OptimizationConfig, Experiment, Objective
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment

from core import *
import pandas as pd

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="config")
def exp_main(cfg : DictConfig) -> None:
    log.info("============= Configuration =============")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info("=========================================")
    search_space = gen_search_space(cfg.problem)

    gs = GenerationStrategy(
        steps=[
            # Can have multiple steps, but SOBOL is enough for parameter variation
            GenerationStep(
                model=Models.SOBOL,
                num_trials=cfg.meta.n_trials,
                min_trials_observed=cfg.meta.n_trials,
                max_parallelism=cfg.meta.n_parallel_trials,
            ),
        ]
    )

    objective=Objective(metric=HPCJobMetric(name=list(cfg.problem.objectives.items())[0][0], cfg=cfg), minimize=True)
    ax_client = AxClient(verbose_logging=False)
    exp = Experiment(
        name=f"{cfg.problem.name}_experiment",
        search_space=ax_client.make_search_space(parameters=search_space, parameter_constraints=[]),
        optimization_config=OptimizationConfig(objective=objective),
        runner=HPCJobRunner(cfg=cfg),
        is_test=False,  # Marking this experiment as a test experiment.
    )

    scheduler = Scheduler(
        experiment=exp,
        generation_strategy=gs,
        options=SchedulerOptions(),
    )

    scheduler.run_n_trials(max_trials=cfg.meta.n_trials)

    # Some post-processing
    params_df = pd.DataFrame()
    trials = scheduler.experiment.get_trials_by_indices(range(cfg.meta.n_trials))
    for tr in trials:
        params_df = pd.concat([params_df, pd.DataFrame({**tr.arm.parameters}, index=[tr.index])])

    # Write trial data
    exp_df = scheduler.experiment.fetch_data().df.drop_duplicates()
    exp_df = exp_df.set_index(["trial_index", "metric_name"]).unstack(level=1)["mean"]
    df = pd.merge(exp_df, params_df, left_index=True, right_index=True)
    df.to_csv(f"{cfg.problem.name}_report_pv.csv")

    # Save experiment for later
    save_experiment(exp, f"{cfg.problem.name}_experiment_pv.json")

if __name__ == "__main__":
    exp_main()
