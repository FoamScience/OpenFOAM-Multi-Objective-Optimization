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
        store_cfg = StoreOptions.model_validate(dict(cfg['store']))
        client = store_cfg.load()
    exp = client._experiment
    artifacts_folder = cfg['optimization']['case_runner']['artifacts_folder']
    if export_json:
        figures_folder = f"{artifacts_folder}/figures"
        os.makedirs(figures_folder, exist_ok=True)

    from ax.api.client import MultiObjective
    is_multi_objective = isinstance(exp.optimization_config._objective, MultiObjective)
    has_status_quo = exp.status_quo is not None
    n_params = len(exp.parameters)
    # Check if a BO model has been fitted (adapter exists on the generation strategy)
    has_model = False
    try:
        gs = client._generation_strategy
        has_model = gs is not None and gs.adapter is not None
    except Exception:
        pass

    analyses_to_run = [ConstraintsFeasibilityAnalysis(),
                       SearchSpaceAnalysis(trial_index=len(exp.trials)-1)]

    # OverviewAnalysis internally runs ResultsAnalysis/CrossValidation/TopSurfaces
    # which need a fitted model and 2+ params (for Sobol second-order indices)
    if has_model and n_params > 1:
        analyses_to_run.insert(0, OverviewAnalysis())

    if has_status_quo:
        analyses_to_run.append(RegressionAnalysis())
    if n_params > 1 and not is_multi_objective:
        analyses_to_run.append(TestOfNoEffectAnalysis())

    experiment_has_choice_params = any([isinstance(exp.parameters[param], ChoiceParameter) for param in exp.parameters])
    for metric in exp.metrics:
        analyses_to_run.append(ParallelCoordinatesPlot(metric_name=metric))
        analyses_to_run.append(ProgressionPlot(metric_name=metric))
        # TopSurfaces/Sensitivity use subset_output which fails on multi-output GPs
        if n_params > 1 and has_model and not is_multi_objective:
            analyses_to_run.append(TopSurfacesAnalysis(metric_name=metric))
        if experiment_has_choice_params and is_multi_objective and has_model and \
                metric in [m.metric_names[0] for m in exp.optimization_config.objective.objectives]:
            analyses_to_run.append(MarginalEffectsPlot(metric_name=metric))
    if n_params > 1 and has_model and not is_multi_objective:
        for metric in exp.metrics:
            analyses_to_run.append(SensitivityAnalysisPlot(metric_name=metric, top_k=10))

    if len(exp.metrics) > 1:
        analyses_to_run.append(ObjectivePFeasibleFrontierPlot())
        metric_pairs = list(combinations(exp.metrics, 2))
        for metric_pair in metric_pairs:
            analyses_to_run.append(ScatterPlot(
                x_metric_name=metric_pair[0],
                y_metric_name=metric_pair[1],
                show_pareto_frontier=True,
                title=f"{metric_pair[0]} vs. {metric_pair[1]}"
            ))
    # Suppress Ax's internal ERROR logs for analyses that fail gracefully
    import logging as _logging
    _ax_analysis_logger = _logging.getLogger("ax.analysis.analysis")
    _prev_level = _ax_analysis_logger.level
    _ax_analysis_logger.setLevel(_logging.CRITICAL)
    cards = client.compute_analyses(display=False, analyses=analyses_to_run)
    _ax_analysis_logger.setLevel(_prev_level)
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


def plot_streaming_metrics(cfg: DictConfig, client: Client):
    """Plot streaming metric progressions per trial, with early-stopping thresholds.

    Generates an interactive HTML report with Plotly figures in the artifacts
    folder.  Designed to be called incrementally from the optimization callback.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc

    exp = client._experiment
    artifacts_folder = cfg["optimization"]["case_runner"]["artifacts_folder"]
    os.makedirs(artifacts_folder, exist_ok=True)

    # Use lookup_data() to get attached streaming data (not fetch_data which re-evaluates metrics)
    fetched = exp.lookup_data()
    data = fetched.full_df if hasattr(fetched, 'full_df') else fetched.df
    if data.empty:
        log.debug("Streaming plot: no data yet")
        return
    if "step" not in data.columns:
        log.debug(f"Streaming plot: no 'step' column (columns: {list(data.columns)})")
        return

    streaming_df = data[data["step"].notna() & (data["step"] > 0)]
    if streaming_df.empty:
        log.debug(f"Streaming plot: no rows with step > 0 (total rows: {len(data)}, "
                  f"step values: {data['step'].unique()[:10].tolist() if 'step' in data.columns else 'N/A'})")
        return

    streaming_metrics = sorted(streaming_df["metric_name"].unique())
    if not streaming_metrics:
        return

    thresholds = {}
    es_cfg = cfg.get("orchestration_settings", {}).get("early_stopping_strategy")
    if es_cfg:
        _extract_es_thresholds(es_cfg, thresholds)

    all_trials = sorted(streaming_df["trial_index"].unique())
    n_trials = len(all_trials)
    palette = pc.qualitative.Plotly * ((n_trials // len(pc.qualitative.Plotly)) + 1)
    trial_colors = {t: palette[i] for i, t in enumerate(all_trials)}

    trial_status = {idx: t.status.name for idx, t in exp.trials.items()}

    n_metrics = len(streaming_metrics)
    fig = make_subplots(
        rows=n_metrics, cols=1,
        subplot_titles=streaming_metrics,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    for i, metric in enumerate(streaming_metrics, 1):
        mdf = streaming_df[streaming_df["metric_name"] == metric]

        for tidx in sorted(mdf["trial_index"].unique()):
            tdf = mdf[mdf["trial_index"] == tidx].sort_values("step")
            status = trial_status.get(tidx, "?")
            fig.add_trace(go.Scatter(
                x=tdf["step"], y=tdf["mean"],
                mode="lines",
                name=f"T{tidx} ({status})",
                legendgroup=f"T{tidx}",
                showlegend=(i == 1),
                line=dict(color=trial_colors.get(tidx), width=1.5),
                hovertemplate=f"T{tidx}<br>step=%{{x}}<br>{metric}=%{{y:.4g}}<extra></extra>",
            ), row=i, col=1)

        if metric in thresholds:
            th = thresholds[metric]
            if "value" in th:
                fig.add_hline(
                    y=th["value"], row=i, col=1,
                    line=dict(color="red", width=2, dash="dash"),
                    annotation_text=f"Threshold: {th['value']}",
                    annotation_position="top left",
                    annotation_font_color="red",
                )
            if th.get("min_prog", 0) > 0:
                fig.add_vline(
                    x=th["min_prog"], row=i, col=1,
                    line=dict(color="orange", width=1.5, dash="dot"),
                    annotation_text=f"Min prog: {th['min_prog']}",
                    annotation_position="bottom right",
                    annotation_font_color="orange",
                )
            if "percentile" in th:
                fig.add_annotation(
                    text=f"Percentile: {th['percentile']}th",
                    xref=f"x{i} domain", yref=f"y{i} domain",
                    x=0.98, y=0.95, showarrow=False,
                    font=dict(size=11, color="brown"),
                    bgcolor="wheat", bordercolor="brown", borderwidth=1,
                    row=i, col=1,
                )

        fig.update_yaxes(title_text=metric, row=i, col=1)

    fig.update_xaxes(title_text="Progression step", row=n_metrics, col=1)
    fig.update_layout(
        title=f"{exp.name} — Streaming Metrics ({n_trials} trials)",
        height=350 * n_metrics,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(b=100),
        hovermode="x unified",
    )

    out_path = f"{artifacts_folder}/{exp.name}_streaming_metrics.html"
    html = f"""<!DOCTYPE html><html lang="en" data-theme="light"><head>
    <title>{exp.name} - Streaming Metrics</title>
    {HTML_CARD_HEADER}
    </head><body>
    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
    </body></html>"""
    with open(out_path, "w") as f:
        f.write(html)


def _extract_es_thresholds(es_cfg, thresholds: dict):
    """Recursively extract early-stopping thresholds from config."""
    if not isinstance(es_cfg, dict):
        try:
            es_cfg = dict(es_cfg)
        except (TypeError, ValueError):
            return
    t = es_cfg.get("type", "")
    if t in ("or", "and"):
        _extract_es_thresholds(es_cfg.get("left", {}), thresholds)
        _extract_es_thresholds(es_cfg.get("right", {}), thresholds)
    elif t == "threshold":
        for sig in es_cfg.get("metric_signatures", []):
            thresholds[sig] = {
                "value": es_cfg.get("metric_threshold"),
                "min_prog": es_cfg.get("min_progression", 0),
            }
    elif t == "percentile":
        for sig in es_cfg.get("metric_signatures", []):
            thresholds[sig] = {
                "percentile": es_cfg.get("percentile_threshold"),
                "min_prog": es_cfg.get("min_progression", 0),
            }
