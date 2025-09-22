#!/usr/bin/env python3

""" Perform multi-objective optimization on OpenFOAM cases

We use:
- Meta's Adaptive Experimentation Platform for optimization,
- foamlib for parameter substitution,

Artifacts: CSV data for experiment trials
"""

import sys, argparse, pprint, webbrowser, json
from omegaconf import DictConfig, OmegaConf
from .config import load_config, save_default_config, override_config
from .common import *
from .metrics import streaming_metric
from .orchestrate import (
    ExistingTrialsOptions, ExperimentOptions, OptimizationOptions,
    ConfigOrchestratorOptions, StoreOptions, TrialGenerationOptions, BaselineOptions,
)
from ax.api.client import MultiObjective, Orchestrator, db_settings_from_storage_config
import pandas as pd

from logging import Logger
from ax.utils.common.logger import get_logger
log : Logger = get_logger(__name__)

def optimize(cfg : DictConfig) -> None:
    """
    Main optimization loop
    """
    log.info("============= Running Configuration =============")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("=================================================")
    
    # 0 Pick up and validate the whole configuration
    exp_cfg = instantiate_with_nested_fields(ExperimentOptions, cfg['experiment'])
    gs_cfg = instantiate_with_nested_fields(TrialGenerationOptions, cfg['trial_generation'])
    opt_cfg = instantiate_with_nested_fields(OptimizationOptions, cfg['optimization'])
    orch_cfg = instantiate_with_nested_fields(ConfigOrchestratorOptions, cfg['orchestration_settings'])
    store_cfg = instantiate_with_nested_fields(StoreOptions, cfg['store'])
    baseline_cfg = instantiate_with_nested_fields(BaselineOptions, cfg['baseline'])
    client = store_cfg.load()

    # 1.0 Experiment setup
    has_experiment = False
    try:
        _ = client._experiment
        has_experiment = True
    except:
        pass

    if not has_experiment:
        client.configure_experiment(**exp_cfg.to_dict())

    ## 2.0 Trial generation
    gs_cfg.set_generation_strategy(client)
    log.info(f"Configured trial generation strategy: {client._generation_strategy.name}")
    log.info("=================================================")

    ## 3.0 Optimization setup
    if not has_experiment and not client._experiment.immutable_search_space_and_opt_config:
        client.configure_optimization(**opt_cfg.to_optimization_dict())
        client.configure_metrics(**opt_cfg.to_objective_metrics_dict())
        client.configure_metrics(**opt_cfg.to_tracking_metrics_dict())
    paramsdict = {
        "machineRampFactor": 10.0,
        "injectionToggleTimeGate1": 1.0,
        "injectionToggleTimeGate2": 1.0,
        "injectionToggleTimeGate3": 1.0,
        "injectionToggleTimeGate4": 1.0,
        "injectionRateGate1": 1.9,
        "injectionRateGate2": 1.9,
        "injectionRateGate3": 1.9,
        "injectionRateGate4": 1.9,
        "injectionInitialStateGate1": "on",
        "injectionInitialStateGate2": "off",
        "injectionInitialStateGate3": "on",
        "injectionInitialStateGate4": "off",
    }
    log.info("=================================================")
    log.info(f"Objectives:\n%s", pprint.pformat(client._experiment.optimization_config._objective))
    if client._experiment._tracking_metrics:
        log.info(f"Tracking metrics:\n%s", pprint.pformat(client._experiment._tracking_metrics))
        log.info("=================================================")
    client.configure_runner(**opt_cfg.to_runner_dict())
    client.set_early_stopping_strategy(orch_cfg.early_stopping_strategy)

    data_attacher = instantiate_with_nested_fields(ExistingTrialsOptions, cfg["existing_trials"])
    data_attacher.load_data(client)

    def callback(sched: Orchestrator):
        store_cfg.save(client)
        streaming_metric(client, cfg["optimization"])
        if not client._experiment.data_by_trial:
            return
        dfs = []
        for trial_id, od in client._experiment.data_by_trial.items():
            for _, mapdata in od.items():
                df = mapdata.df.copy()
                if "trial_index" not in df.columns:
                    df["trial_index"] = trial_id
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True, sort=False)
        metadf = {
            trial_index: {
                "job_id": trial.run_metadata.get("job_id"),
                "generation_node": trial.generator_run._model_key,
                "status": trial.status.__repr__().strip("<enum 'TrialStatus'>."),
                "case_path": trial.run_metadata.get("job", {}).get("case_path"),
                **client._experiment.trials[trial_index].arm.parameters
            }
            for trial_index, trial in client._experiment.trials.items()
        }
        if metadf:
            for key in metadf[0].keys():
                df[key] = df["trial_index"].map(lambda x: metadf.get(x, {}).get(key))
        df.to_csv(
            f"{cfg["optimization"]["case_runner"]["artifacts_folder"]}/{cfg["experiment"]["name"]}_report.csv",
            index=False
        )

    if not has_experiment and baseline_cfg.parameters:
        log.info("=============== Running Baseline ================")
        baseline_index = 0 if len(client._experiment._trials) == 0 else max(client._experiment._trials.keys()) + 1
        from .orchestrate import ManualGenerationNode
        from ax.generation_strategy.generation_strategy import GenerationStrategy
        from ax.core.trial import Trial
        from pyre_extensions import assert_is_instance
        scheduler = Orchestrator(
            experiment=client._experiment,
            generation_strategy=GenerationStrategy(
                name="baseline",
                nodes=[ManualGenerationNode(node_name="baseline", parameters=baseline_cfg.parameters)]),
            options=orch_cfg.to_scheduler_options(),
            db_settings=db_settings_from_storage_config(client._storage_config)
            if client._storage_config is not None
            else None,
        )
        scheduler.run_n_trials(max_trials=1, timeout_hours=orch_cfg.timeout_hours, idle_callback=callback)
        client._experiment.status_quo = assert_is_instance(
            client._experiment.trials[baseline_index], Trial
        ).arm

    scheduler = Orchestrator(
        experiment=client._experiment,
        generation_strategy=client._generation_strategy_or_choose(),
        options=orch_cfg.to_scheduler_options(),
        db_settings=db_settings_from_storage_config(client._storage_config)
        if client._storage_config is not None
        else None,
    )
    log.info("============ Running optimization ===============")
    scheduler.run_all_trials(timeout_hours=orch_cfg.timeout_hours, idle_callback=callback)
    store_cfg.save(client)

    log.info("=========== Best set of parameters ==============")
    best_parameters, prediction = None, None
    if isinstance(client._experiment.optimization_config._objective, MultiObjective):
        for best_parameters, prediction, _, _ in client.get_pareto_frontier(use_model_predictions=True):
            log.info("Pareto-frontier configuration:\n%s", json.dumps(best_parameters, indent=2))
            log.info("Predictions for Pareto-frontier configuration (mean, variance):\n%s", json.dumps(prediction, indent=2))
    else:
        best_parameters, prediction, _, _ = client.get_best_parameterization(use_model_predictions=True)
        log.info("Best parameter set:\n%s", json.dumps(best_parameters, indent=2))
        log.info("Predictions for best parameter set (mean, variance):\n%s", json.dumps(prediction, indent=2))

    log.info("=================================================")
    cards = client.compute_analyses(display=False)
    store_cfg.save(client, cards)
    for card in cards:
        if not card._repr_html_():
            continue
        if card.subtitle.__contains__("error occurred while computing"):
            log.warning(f"{card.title}, {card.subtitle}")
            continue
        html = f"""<!DOCTYPE html><html lang="en" data-theme="light"><head>
        <title>{card.title}</title>
        {HTML_CARD_HEADER}
        </head><body>
        <h1>{client._experiment.name} - {card.title}</h1>
        <p>{card.subtitle}</p>
        {card._repr_html_()}
        </body>{HTML_CARD_SCRIPT}</html>"""
        html_path = f"artifacts/{client._experiment.name}_{unixlike_filename(card.title)}.html"
        with open(html_path, "w") as f:
            f.write(html)
            webbrowser.open(html_path)
    log.info("==================== End ========================")

def main():
    parser = argparse.ArgumentParser(description="Multi-objective optimization for OpenFOAM cases.")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help='Path to config YAML file')
    parser.add_argument('--generate-config', action='store_true', help='Generate a default config file and exit')
    parser.add_argument('-V', '--version', action='version', version='%(prog)s ' + VERSION)
    parser.add_argument('overrides', nargs=argparse.REMAINDER, help='Config overrides in ++key=value format')
    args = parser.parse_args()

    if args.generate_config:
        save_default_config(args.config)
        sys.exit(0)

    cfg = load_config(args.config)
    if args.overrides:
        overrides = [o for o in args.overrides if o.startswith('++') or '=' in o]
        if overrides:
            cfg = override_config(cfg, overrides)
    optimize(cfg)
