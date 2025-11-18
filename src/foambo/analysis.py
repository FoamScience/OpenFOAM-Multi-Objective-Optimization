#!/usr/bin/env python3

""" Produce analysis cards for optimization runs """

import webbrowser
from .common import *
from .orchestrate import StoreOptions
from ax.core.parameter import ChoiceParameter
from ax.api.client import Client
from ax.analysis.overview import OverviewAnalysis
from ax.analysis.insights import InsightsAnalysis
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
from ax.analysis.plotly.marginal_effects import MarginalEffectsPlot
from ax.analysis.plotly.objective_p_feasible_frontier import ObjectivePFeasibleFrontierPlot
from ax.analysis.plotly.progression import ProgressionPlot
from ax.analysis.plotly.scatter import ScatterPlot
from ax.analysis.plotly.sensitivity import SensitivityAnalysisPlot
from ax.analysis.plotly.top_surfaces import TopSurfacesAnalysis
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.analysis.healthcheck.constraints_feasibility import ConstraintsFeasibilityAnalysis
from ax.analysis.healthcheck.no_effects_analysis import TestOfNoEffectAnalysis
from ax.analysis.healthcheck.regression_analysis import RegressionAnalysis
from ax.analysis.healthcheck.search_space_analysis import SearchSpaceAnalysis
from ax.plot.pareto_frontier import scatter_plot_with_hypervolume_trace_plotly, scatter_plot_with_pareto_frontier_plotly
from pprint import pformat
from itertools import combinations

def compute_analysis_cards(cfg: DictConfig, client: Client | None, open_html=False, export_json=False):
    set_experiment_name(cfg['experiment']['name'])
    if not client:
        store_cfg = instantiate_with_nested_fields(StoreOptions, cfg['store'])
        client = store_cfg.load()
    exp = client._experiment
    artifacts_folder = cfg['optimization']['case_runner']['artifacts_folder']
    if export_json:
        figures_folder = f"{artifacts_folder}/figures"
        os.makedirs(figures_folder, exist_ok=True)

    analyses_to_run = [OverviewAnalysis(),
                       ConstraintsFeasibilityAnalysis(),
                       TestOfNoEffectAnalysis(),
                       RegressionAnalysis(),
                       SearchSpaceAnalysis(trial_index=len(exp.trials)-1)]

    experiment_has_choice_params = any([isinstance(exp.parameters[param], ChoiceParameter) for param in exp.parameters])
    for metric in exp.metrics:
        analyses_to_run.append(ParallelCoordinatesPlot(metric_name=metric))
        analyses_to_run.append(ProgressionPlot(metric_name=metric))
        analyses_to_run.append(TopSurfacesAnalysis(metric_name=metric))
        if experiment_has_choice_params and metric in [m.metric_names[0] for m in exp.optimization_config.objective.objectives]:
            analyses_to_run.append(MarginalEffectsPlot(metric_name=metric))
    analyses_to_run.append(SensitivityAnalysisPlot(metric_names=exp.metrics, top_k=10))

    if len(exp.metrics) > 1:
        analyses_to_run.append(ObjectivePFeasibleFrontierPlot(show_pareto_frontier=True))
        metric_pairs = list(combinations(exp.metrics, 2))
        for metric_pair in metric_pairs:
            analyses_to_run.append(ScatterPlot(
                x_metric_name=metric_pair[0],
                y_metric_name=metric_pair[1],
                show_pareto_frontier=True,
                title=f"{metric_pair[0]} vs. {metric_pair[1]}"
            ))
    cards = client.compute_analyses(display=False, analyses=analyses_to_run)
    for card in cards:
        if "Marginal Effects" in card.title:
            metric_in_card = next(metric for metric in exp.metrics if metric in card._repr_html_())
            card.title = f"{card.title} on {metric_in_card}"
        if "Top Surfaces Analysis" in card.title:
            metric_in_card = next(metric for metric in exp.metrics if metric in card._repr_html_())
            card.title = f"{card.title} for {metric_in_card}"

    for card in cards:
        if not card._repr_html_():
            continue
        if card.subtitle.__contains__("error occurred while computing"):
            log.warning(f"{card.title}: {card.subtitle}")
            continue
        html = f"""<!DOCTYPE html><html lang="en" data-theme="light"><head>
        <title>{card.title}</title>
        {HTML_CARD_HEADER}
        </head><body>
        <h1>{client._experiment.name} - {card.title}</h1>
        <p>{card.subtitle}</p>
        {card._repr_html_()}
        </body>{HTML_CARD_SCRIPT}</html>"""
        html_path = f"{artifacts_folder}/{client._experiment.name}_{unixlike_filename(card.title)}.html"
        with open(html_path, "w") as f:
            f.write(html)
            if open_html:
                webbrowser.open(html_path)

        if export_json:
            if isinstance(card, PlotlyAnalysisCard):
                try:
                    fig = card.get_figure()
                    json_path = f"{figures_folder}/{client._experiment.name}_{unixlike_filename(card.title)}.json"
                    with open(json_path, "w") as f:
                        f.write(fig.to_json())
                except Exception as e:
                    log.warning(f"Failed to export {card.title} to JSON: {e}")
            elif hasattr(card, 'fig') and card.fig is not None:
                try:
                    json_path = f"{figures_folder}/{client._experiment.name}_{unixlike_filename(card.title)}.json"
                    with open(json_path, "w") as f:
                        f.write(card.fig.to_json())
                except Exception as e:
                    log.warning(f"Failed to export {card.title} to JSON: {e}")
    return cards

def plot_pareto_frontier(cfg: DictConfig, client, front=None, open_html=False, export_json=False):
    artifacts_folder = cfg['optimization']['case_runner']['artifacts_folder']
    if export_json:
        figures_folder = f"{artifacts_folder}/figures"
        os.makedirs(figures_folder, exist_ok=True)

    figures = []
    figure_html = []
    if not front:
        front = client.get_pareto_frontier(use_model_predictions=True)
    df = client._experiment.fetch_data().df
    opt_config = client._experiment.optimization_config
    fig = scatter_plot_with_hypervolume_trace_plotly(client._experiment)
    figures.append(fig)
    figure_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    if export_json:
        json_path = f"{figures_folder}/{client._experiment.name}_hypervolume_trace.json"
        with open(json_path, "w") as f:
            f.write(fig.to_json())

    X_labels = [
            pformat(client._experiment.trials[tr].arm.parameters, width=1).replace("\n", "<br>").replace('{', '').replace('}', '').replace("'", '')
            for tr in client._experiment.trials]
    objective_metrics = opt_config.objective.metric_names
    metric_pairs = list(combinations(objective_metrics, 2))
    for metrics in metric_pairs:
        Y_pareto = np.array([[f[1][metrics[0]][0], f[1][metrics[1]][0]] for f in front])
        X_pareto = [f[0] for f in front]
        x = df.loc[df["metric_name"] == metrics[0], "mean"].to_numpy()
        y = df.loc[df["metric_name"] == metrics[1], "mean"].to_numpy()
        Y = np.column_stack([x, y])
        get_metric_bound = lambda metric: next(
            threshold.bound
            for threshold in opt_config.all_constraints
            if metric == threshold.metric.name)
        get_metric_relativity = lambda metric: next(
            threshold.relative
            for threshold in opt_config.all_constraints
            if metric == threshold.metric.name)
        ref_point = [get_metric_bound(metrics[0]), get_metric_bound(metrics[1])]
        for i in range(2):
            if get_metric_relativity(metrics[i]):
                try:
                    baseline = df.loc[(df["metric_name"] == metrics[i]) & (df["arm_name"] == "baseline"), "mean"].values[0]
                    ref_point[i] = get_metric_bound(metrics[i]) * baseline
                except Exception:
                    ref_point[1] = None
                    log.error(f"Relative constraint was set for {metrics[0]} but no baseline arm was found!")
        get_minimize = lambda metric: next(
            obj.minimize
            for obj in opt_config.objective.objectives
            if metric in obj.metric_names
        )
        try:
            fig = scatter_plot_with_pareto_frontier_plotly(
                Y=Y,
                Y_pareto=Y_pareto,
                reference_point=ref_point,
                metric_x=metrics[0],
                metric_y=metrics[1],
                minimize=(get_minimize(metrics[0]), get_minimize(metrics[1])),
                hovertext=X_labels,
                )
            hv_line = next(dt for dt in fig.data if dt['line']['shape'] == 'hv')
            Y_pareto_labels = [
                pformat(params, width=1).replace("\n", "<br>").replace('{', '').replace('}', '').replace("'", '')
                for params in X_pareto]
            hv_line.hovertext = Y_pareto_labels
            hv_line.mode = 'lines+markers'
            hv_line.marker.symbol = 'circle'
            hv_line.marker.size = 16
            figures.append(fig)
            figure_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

            if export_json:
                json_path = f"{figures_folder}/{client._experiment.name}_pareto_frontier_{metrics[0]}_vs_{metrics[1]}.json"
                with open(json_path, "w") as f:
                    f.write(fig.to_json())
        except Exception:
            log.error(f"Something went wrong with the generation of Pareto Frontier for {metrics}")

    html=f"""<!DOCTYPE html><html lang="en" data-theme="light"><head>
        <title>Pareto frontier analysis for {client._experiment.name} experiment</title>
        {HTML_CARD_HEADER}
        </head><body>
        <h1>Hypervolume trace</h1>
        <p>Hypervolume (HV) measures the volume of objective space
        dominated by the Pareto frontier relative to a reference point.</p>
        <p>A flat HV trace indicates that the frontier isn’t improving much;
        maybe the optimization has converged.
        As long as HV keeps increasing, there is probably some merit in continuing the optimization</p>
        {figure_html[0]}
        <h1>Pareto frontier</h1>
        <p>The Pareto Frontier shows parameter sets that strike a balance between objective pairs.
        Changing parameter values will worsen either objective, which is designated by the Horizontal-Vertical line.
        The bounds from outcome constraints are shown as a reference point (the star) </p>
    """
    for fg in range(1, len(figure_html)):
        html = f"""{html} <div> {figure_html[fg]} </div>"""
    html = f"""{html} </body>{HTML_CARD_SCRIPT}</html>"""
    html_path = f"{artifacts_folder}/{client._experiment.name}_pareto_frontier.html"
    with open(html_path, "w") as f:
        f.write(html)
        if open_html:
            webbrowser.open(html_path)
    log.info("==================== End ========================")
    return figures
