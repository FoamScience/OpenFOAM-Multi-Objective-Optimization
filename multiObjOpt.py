#!/usr/bin/env python3

""" Perform multi-objective optimization on OpenFOAM cases using FullyBayesianMOO if possible

This script defines functions to perform multi-objective optimization on OpenFOAM
cases given a YAML/JSON config file (Supported through Hydra, default: config.yaml).

We use the Adaptive Experimentation Platform for optimization, PyFOAM for parameter substitution
and Hydra for 0-code configuration.

Output: A JSON Snapshot of the experiment and CSV data for experiment trials

Notes:
- You can also use a single objective
"""

import hydra, logging
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from ax.service.ax_client import AxClient
from core import *

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def exp_main(cfg : DictConfig) -> None:
    log.info("============= Configuration =============")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info("=========================================")
    search_space = gen_search_space(cfg.problem)
    objs = gen_objectives(cfg.problem)
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name=f"{cfg.problem.name}_experiment",
        parameters=search_space,
        objectives=objs,
        overwrite_existing_experiment=True,
        choose_generation_strategy_kwargs={'use_saasbo': True},
        is_test=False,
    )
    params_df = pd.DataFrame()
    for _ in range(int(cfg.meta.n_trials)):
        parameters, trial_index = ax_client.get_next_trial()
        data = {}
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, cfg, data))
        params_df = pd.concat([params_df, pd.DataFrame({**parameters, **data}, index=[trial_index])])
    objectives = []
    try:
        objectives = ax_client.experiment.optimization_config.objective.objectives
    except:
        objectives = [ax_client.experiment.optimization_config.objective]

    metric_names = [e.metric.name for e in objectives]
    exp_df = ax_client.experiment.fetch_data().df
    exp_df = exp_df.set_index(["trial_index", "metric_name"]).unstack(level=1)["mean"]
    df = pd.merge(exp_df, params_df, left_index=True, right_index=True)
    df.to_csv(f"{cfg.problem.name}_report.csv")
    ax_client.save_to_json_file(f"{cfg.problem.name}_snapshot.json")

if __name__ == "__main__":
    exp_main()
