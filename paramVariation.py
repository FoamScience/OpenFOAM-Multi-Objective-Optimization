#!/usr/bin/env python3

""" Perform parameter-variation study on OpenFOAM cases

This script runs a paramter-variation study on OpenFOAM cases given a YAML/JSON config
file (Supported through Hydra, default: config.yaml).

Output: A JSON Snapshot of the experiment and CSV data for experiment trials

Notes:
- Use multiObjOpt.py for optimization studies
- Float and Int parameters are sampled in a quasi-random fashion using SOBOL. If you want
  finer control over parameter values, convert them to choice type.

"""

import hydra, logging
from omegaconf import DictConfig, OmegaConf

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient

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
                max_parallelism=1,
            ),
        ]
    )
    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name=f"{cfg.problem.name}_experiment",
        parameters=search_space,
        overwrite_existing_experiment=True,
        is_test=False,
    )

    df = pd.DataFrame()
    for i in range(cfg.meta.n_trials):
        try:
            generator_run = gs.gen(
                experiment=ax_client.experiment,
                data=None,
                n=1,
            )
            trial = ax_client.experiment.new_trial(generator_run)
            trial.mark_running(no_runner_required=True)
            data = {}
            out = evaluate(trial.arms[0].parameters, cfg, data)
            trial.mark_completed()
            df = pd.concat([df, pd.DataFrame({**data, **trial.arms[0].parameters, "model": generator_run._model_key, **out}, index=[trial.index])])
        except:
            log.warning(f"Could not generate more than {i} trials.")

    # Write out data as a Pandas dataframe
    log.info(df)
    df.to_csv(f"{cfg.problem.name}_report.csv")
    # Also save a json snapshot
    ax_client.save_to_json_file(f"{cfg.problem.name}_snapshot.json")

if __name__ == "__main__":
    exp_main()
