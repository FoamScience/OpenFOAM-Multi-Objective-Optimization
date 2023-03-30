#!/usr/bin/env python3

""" Perform multi-objective optimization on OpenFOAM cases using FullyBayesianMOO if possible

This script defines functions to perform multi-objective optimization on OpenFOAM
cases given a YAML/JSON config file (Supported through Hydra, default: config.yaml).

We use the Adaptive Experimentation Platform for optimization, PyFOAM for parameter substitution
and Hydra for 0-code configuration.

Output: CSV data for experiment trials

Things to improve:
- Optimization restart? Maybe from JSON file as a start.
- Dependent parameters.

Notes:
- You can also use a single objective
"""

import hydra, logging, json
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from ax.service.ax_client import AxClient, MultiObjective
from ax.service.scheduler import ObjectiveThreshold, Scheduler, SchedulerOptions
from ax.core import OptimizationConfig, Experiment, Objective, MultiObjectiveOptimizationConfig
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.feature_importances import plot_feature_importance_by_feature

from core import *

import zmq
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://172.22.0.8:5555")

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def exp_main(cfg : DictConfig) -> None:
    log.info("============= Configuration =============")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info("=========================================")
    search_space = gen_search_space(cfg.problem)
    metrics = [HPCJobMetric(name=key, cfg=cfg) for key, _ in cfg.problem.objectives.items()]
    objectives=[Objective(metric=m, minimize=item.minimize) for m, (_, item) in zip(metrics, cfg.problem.objectives.items())]
    thresholds=[ObjectiveThreshold(metric=m, bound=float(item.threshold), relative=False) for m, (_, item) in zip(metrics, cfg.problem.objectives.items())]
    ax_client = AxClient(verbose_logging=False)
    optimization_config = MultiObjectiveOptimizationConfig(objective=MultiObjective(objectives), objective_thresholds=thresholds) \
            if len(objectives) > 1 else OptimizationConfig(objectives[0])
    exp = Experiment(
        name=f"{cfg.problem.name}_experiment",
        search_space=ax_client.make_search_space(parameters=search_space, parameter_constraints=[]),
        optimization_config=optimization_config,
        # something like this will switch to single-objective optimization
        runner=HPCJobRunner(cfg=cfg),
        is_test=False,
    )
    gs = choose_generation_strategy(
        search_space=exp.search_space, 
        no_winsorization=True,
        num_trials=cfg.meta.n_trials,
        max_parallelism_cap=cfg.meta.n_parallel_trials,
        use_saasbo=cfg.meta.use_saasbo,
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
    df.to_csv(f"{cfg.problem.name}_report_opt.csv")

    try:
        log.info("==== Pareto optimal paramters: ===")
        pareto_params = scheduler.get_pareto_optimal_parameters()
        log.info(pareto_params)
        log.info(json.dumps(pareto_params, indent=4))
        log.info("==================================")
    except:
        log.info("==== Best paramters: ===")
        pareto_params = scheduler.get_best_parameters()
        log.info(pareto_params)
        log.info(json.dumps(pareto_params, indent=4))
        log.info("==================================")


    # Plot Pareto frontier
    try:
        metric_names = [e.metric.name for e in objectives]
        sobol_frontier = compute_posterior_pareto_frontier(
            experiment=exp,
            data=exp.fetch_data(),
            primary_objective=objectives[1].metric,
            secondary_objective=objectives[0].metric,
            absolute_metrics=metric_names,
            num_points=int(cfg.meta.n_pareto_points),
        )
        plot_frontier(sobol_frontier, 0.9, f"{cfg.problem.name}_fronier")
        log.info(scheduler.get_hypervolume())
        # Frontier dataframe
        params_df = pd.DataFrame(
            sobol_frontier.param_dicts,
            index=range(cfg.meta.n_pareto_points)
        )
        metrics_df = pd.DataFrame(
            [{k:v[i] for k,v in sobol_frontier.means.items()} for i in range(cfg.meta.n_pareto_points)],
            index=range(cfg.meta.n_pareto_points)
        )
        sems_df = pd.DataFrame(
            [{k+"_sems":v[i] for k,v in sobol_frontier.sems.items()} for i in range(cfg.meta.n_pareto_points)],
            index=range(cfg.meta.n_pareto_points)
        )
        df = pd.merge(params_df, metrics_df, left_index=True, right_index=True)
        df = pd.merge(df, sems_df, left_index=True, right_index=True)
        df.to_csv(f"{cfg.problem.name}_frontier_report.csv")
        log.info(df)
    except:
        log.warning("Could not plot paleto front, not a multi-objective optimization?")

    # Feature Importance
    try:
        cur_model = scheduler.generation_strategy.model
        feature_importance = plot_feature_importance_by_feature(cur_model, relative=True)
        render(feature_importance)
        with open(f'{cfg.problem.name}_feature_importance.html', 'w') as outfile:
            outfile.write(render_report_elements(
                f"{cfg.problem.name}_feature_importance", 
                html_elements=[plot_config_to_html(feature_importance)], 
                header=False,
            ))
        importances = pd.DataFrame([{j['y'][k]:j['x'][k] for k in range(len(cfg.problem.parameters.keys()))} for j in feature_importance.data['data']],
            index=[feature_importance.data['layout']['updatemenus'][0]['buttons'][i]['label'] for i in range(len(feature_importance.data['data']))])
        importances.to_csv(f"{cfg.problem.name}_feature_importance_report.csv")
        log.info(importances)
    except:
        log.warning("Could not compute feature importance, no Gaussian process has been called?")

import time

if __name__ == "__main__":
    exp_main()
    socket.send(b"stop")
