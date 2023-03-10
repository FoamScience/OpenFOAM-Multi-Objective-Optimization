#!/usr/bin/env python3
"""Post-process a JSON snapshot of an optimization run of OpenFOAM cases

This script writes paramters used on trials, generates pareto fronts and plots
feature importance if the optimization has gone through a Gaussian procedure.

Output:
- Pareto front plots (HTML and CSV data)
-Feature Importance
- Trial data reports

"""

import hydra, logging
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import plot_config_to_html, render
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.utils.report.render import render_report_elements

from core import *

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="params")
def explore_main(cfg : DictConfig) -> None:
    log.info("============= Configuration =============")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info("=========================================")
    ax_client = AxClient.load_from_json_file(f"{cfg.problem.name}_snapshot.json")
    # Pareto frontier
    try:
        objectives = ax_client.experiment.optimization_config.objective.objectives
        metric_names = [e.metric.name for e in objectives]
        sobol_frontier = compute_posterior_pareto_frontier(
            experiment=ax_client.experiment,
            data=ax_client.experiment.fetch_data(),
            primary_objective=objectives[1].metric,
            secondary_objective=objectives[0].metric,
            absolute_metrics=metric_names,
            num_points=int(cfg.meta.n_pareto_points),
        )
        plot_frontier(sobol_frontier, 0.9, f"{cfg.problem.name}_fronier")
        log.info(ax_client.get_hypervolume())
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
        log.warning("Couldn't plot paleto front, not a multi-objective optimization?")
    log.info(ax_client.experiment.fetch_data().df)

    # Feature Importance
    try:
        feature_importance = ax_client.get_feature_importances()
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
        log.warning("Couldn't compute feature importance, no Gaussian process hasbeen called?")
    
    ## Predict and evaluate how good the optimization was
    pred_params = [{ k:vv for a in v for k,vv in a.items()} for _,v in cfg.verify.items()]
    preds = ax_client.get_model_predictions_for_parameterizations(pred_params)
    pred_means = [{k:v[0] for k,v in i.items()} for i in preds]
    pred_covar = [{k+"_covariance":v[1] for k,v in i.items()} for i in preds]
    log.info(pred_means)
    log.info(pred_covar)
    
    params_df = pd.DataFrame()
    for parameters in pred_params:
        data = {}
        log.info(evaluate(parameters, cfg, data))
        log.info(data)
        params_df = pd.concat([params_df, pd.DataFrame({**parameters, **data}, index=[0])])
    params_df.to_csv(f"{cfg.problem.name}_verify_report.csv")

if __name__ == "__main__":
    explore_main()
