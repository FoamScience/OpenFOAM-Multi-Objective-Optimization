#!/usr/bin/env python3

"""Analysis for foamBO optimization runs.

Separated into data computation and plotting layers:
  - compute_*() functions return plain data structures (dicts, lists)
  - plot_*() functions take those structures and produce Plotly figures
  - save/render functions handle HTML file I/O

The web UI and API server can use compute_* functions directly.
"""

import logging
import webbrowser
from .common import *
from .orchestrate import StoreOptions
from ax.core.parameter import ChoiceParameter
from ax.api.client import Client, MultiObjective
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


# ---------------------------------------------------------------------------
# Data computation (no Plotly, no file I/O)
# ---------------------------------------------------------------------------

def compute_pareto_data(client: Client, front=None):
    """Compute Pareto frontier data from the experiment.

    Returns dict with keys:
      - front: list of (params, predictions, trial_index, arm_name)
      - Y_all: dict of metric_name -> list of values (all trials)
      - Y_pareto: dict of metric_pair -> np.array of pareto points
      - ref_points: dict of metric_pair -> [ref_x, ref_y]
      - minimize: dict of metric_name -> bool
      - metric_pairs: list of (metric_a, metric_b)
      - labels: list of param strings per trial
    """
    exp = client._experiment
    opt_config = exp.optimization_config
    df = exp.fetch_data().df

    if not front:
        try:
            front = client.get_pareto_frontier(use_model_predictions=True)
        except Exception:
            front = client.get_pareto_frontier(use_model_predictions=False)

    objective_metrics = opt_config.objective.metric_names
    metric_pairs = list(combinations(objective_metrics, 2))

    labels = [
        pformat(exp.trials[tr].arm.parameters, width=1)
        .replace("\n", " ").replace('{', '').replace('}', '').replace("'", '')
        for tr in exp.trials
    ]

    minimize = {}
    for obj in opt_config.objective.objectives:
        for mn in obj.metric_names:
            minimize[mn] = obj.minimize

    Y_all = {}
    for m in objective_metrics:
        Y_all[m] = df.loc[df["metric_name"] == m, "mean"].tolist()

    Y_pareto = {}
    ref_points = {}
    for pair in metric_pairs:
        Y_pareto[pair] = np.array([[f[1][pair[0]][0], f[1][pair[1]][0]] for f in front])
        # Compute reference points from constraints
        ref = [None, None]
        try:
            for i, metric in enumerate(pair):
                threshold = next(
                    t for t in opt_config.all_constraints
                    if metric == t.metric.name
                )
                bound = threshold.bound
                if threshold.relative:
                    try:
                        bl = df.loc[(df["metric_name"] == metric) & (df["arm_name"] == "baseline"), "mean"].values[0]
                        bound = bound * bl
                    except Exception:
                        pass
                ref[i] = bound
        except (StopIteration, Exception):
            pass
        ref_points[pair] = ref

    pareto_params = [f[0] for f in front]

    return {
        "front": front,
        "Y_all": Y_all,
        "Y_pareto": Y_pareto,
        "ref_points": ref_points,
        "minimize": minimize,
        "metric_pairs": metric_pairs,
        "labels": labels,
        "pareto_params": pareto_params,
        "objective_metrics": objective_metrics,
    }


def compute_streaming_data(client: Client, cfg: DictConfig):
    """Extract streaming metric data and thresholds.

    Returns dict with keys:
      - metrics: dict of metric_name -> dict of trial_index -> {steps: [], values: []}
      - trial_statuses: dict of trial_index -> status string
      - thresholds: dict of metric_name -> {value/percentile, min_prog}
      - experiment_name: str
      - n_trials: int
    """
    exp = client._experiment
    fetched = exp.lookup_data()
    data = fetched.full_df if hasattr(fetched, 'full_df') else fetched.df

    result = {
        "metrics": {},
        "trial_statuses": {idx: t.status.name for idx, t in exp.trials.items()},
        "thresholds": {},
        "experiment_name": exp.name,
        "n_trials": 0,
    }

    if data.empty or "step" not in data.columns:
        return result

    streaming_df = data[data["step"].notna() & (data["step"] > 0)]
    if streaming_df.empty:
        return result

    all_trials = sorted(streaming_df["trial_index"].unique())
    result["n_trials"] = len(all_trials)

    for metric in sorted(streaming_df["metric_name"].unique()):
        mdf = streaming_df[streaming_df["metric_name"] == metric]
        per_trial = {}
        for tidx in sorted(mdf["trial_index"].unique()):
            tdf = mdf[mdf["trial_index"] == tidx].sort_values("step")
            per_trial[tidx] = {
                "steps": tdf["step"].tolist(),
                "values": tdf["mean"].tolist(),
            }
        result["metrics"][metric] = per_trial

    es_cfg = cfg.get("orchestration_settings", {}).get("early_stopping_strategy")
    if es_cfg:
        _extract_es_thresholds(es_cfg, result["thresholds"])
        _resolve_percentile_thresholds(result["metrics"], result["thresholds"])

    return result


def compute_objective_progression(client: Client):
    """Compute per-metric progression over trial index.

    Returns dict of metric_name -> list of (trial_index, value).
    """
    exp = client._experiment
    df = exp.fetch_data().df
    if df.empty:
        return {}

    result = {}
    for metric in df["metric_name"].unique():
        mdf = df[df["metric_name"] == metric]
        points = list(zip(mdf["trial_index"].tolist(), mdf["mean"].tolist()))
        points.sort(key=lambda x: x[0])
        result[metric] = points
    return result


# ---------------------------------------------------------------------------
# Plotly rendering (takes computed data, returns figures)
# ---------------------------------------------------------------------------

def plot_pareto_figures(pareto_data: dict, experiment_name: str):
    """Build Plotly figures from computed Pareto data.

    Returns list of Plotly figure objects.
    """
    figures = []

    # Hypervolume trace (requires Ax experiment object — uses Ax's built-in)
    # This is kept as-is since it needs the experiment directly
    # Callers should use scatter_plot_with_hypervolume_trace_plotly separately

    for pair in pareto_data["metric_pairs"]:
        Y_pareto = pareto_data["Y_pareto"][pair]
        ref_point = pareto_data["ref_points"][pair]
        Y = np.column_stack([
            pareto_data["Y_all"][pair[0]],
            pareto_data["Y_all"][pair[1]],
        ])

        try:
            fig = scatter_plot_with_pareto_frontier_plotly(
                Y=Y,
                Y_pareto=Y_pareto,
                reference_point=ref_point,
                metric_x=pair[0],
                metric_y=pair[1],
                minimize=(
                    pareto_data["minimize"].get(pair[0], True),
                    pareto_data["minimize"].get(pair[1], True),
                ),
                hovertext=pareto_data["labels"],
            )
            hv_line = next(dt for dt in fig.data if dt['line']['shape'] == 'hv')
            pareto_labels = [
                pformat(params, width=1).replace("\n", "<br>")
                .replace('{', '').replace('}', '').replace("'", '')
                for params in pareto_data["pareto_params"]
            ]
            hv_line.hovertext = pareto_labels
            hv_line.mode = 'lines+markers'
            hv_line.marker.symbol = 'circle'
            hv_line.marker.size = 16
            figures.append(fig)
        except Exception:
            log.error(f"Pareto frontier plot failed for {pair}")

    return figures


def plot_streaming_figures(streaming_data: dict):
    """Build Plotly figure from computed streaming data.

    Returns a single Plotly figure with subplots per metric.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc

    metrics = streaming_data["metrics"]
    if not metrics:
        return None

    metric_names = sorted(metrics.keys())
    n_metrics = len(metric_names)
    thresholds = streaming_data["thresholds"]
    trial_statuses = streaming_data["trial_statuses"]

    # Collect all trial indices across all metrics
    all_trials = sorted({tidx for per_trial in metrics.values() for tidx in per_trial})
    n_trials = len(all_trials)
    palette = pc.qualitative.Plotly * ((n_trials // len(pc.qualitative.Plotly)) + 1)
    trial_colors = {t: palette[i] for i, t in enumerate(all_trials)}

    fig = make_subplots(
        rows=n_metrics, cols=1,
        subplot_titles=metric_names,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    for i, metric in enumerate(metric_names, 1):
        per_trial = metrics[metric]
        for tidx in sorted(per_trial.keys()):
            td = per_trial[tidx]
            status = trial_statuses.get(tidx, "?")
            fig.add_trace(go.Scatter(
                x=td["steps"], y=td["values"],
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
    exp_name = streaming_data.get("experiment_name", "")
    fig.update_layout(
        title=f"{exp_name} — Streaming Metrics ({streaming_data['n_trials']} trials)",
        height=350 * n_metrics,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(b=100),
        hovermode="x unified",
    )

    return fig


# ---------------------------------------------------------------------------
# File I/O / HTML rendering
# ---------------------------------------------------------------------------

def save_html(html: str, path: str, open_browser: bool = False):
    """Write an HTML string to a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    if open_browser:
        webbrowser.open(path)


def figure_to_html(fig, title: str, experiment_name: str) -> str:
    """Wrap a Plotly figure in a standalone HTML page."""
    return f"""<!DOCTYPE html><html lang="en" data-theme="light"><head>
    <title>{experiment_name} - {title}</title>
    {HTML_CARD_HEADER}
    </head><body>
    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
    </body></html>"""


# ---------------------------------------------------------------------------
# High-level entry points (backward compatible)
# ---------------------------------------------------------------------------

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

    is_multi_objective = isinstance(exp.optimization_config.objective, MultiObjective)
    has_status_quo = exp.status_quo is not None
    n_params = len(exp.parameters)
    has_model = False
    try:
        gs = client._generation_strategy
        has_model = gs is not None and gs.adapter is not None
    except Exception:
        pass

    analyses_to_run = [ConstraintsFeasibilityAnalysis(),
                       SearchSpaceAnalysis(trial_index=len(exp.trials)-1)]

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
        save_html(html, html_path, open_browser=open_html)

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

    exp = client._experiment
    pareto_data = compute_pareto_data(client, front=front)

    # Hypervolume trace (Ax built-in, needs experiment object directly)
    figures = []
    figure_html = []
    fig_hv = scatter_plot_with_hypervolume_trace_plotly(exp)
    figures.append(fig_hv)
    figure_html.append(fig_hv.to_html(full_html=False, include_plotlyjs='cdn'))

    if export_json:
        json_path = f"{figures_folder}/{exp.name}_hypervolume_trace.json"
        with open(json_path, "w") as f:
            f.write(fig_hv.to_json())

    # Pareto scatter plots
    pareto_figs = plot_pareto_figures(pareto_data, exp.name)
    for fig in pareto_figs:
        figures.append(fig)
        figure_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    if export_json:
        for i, pair in enumerate(pareto_data["metric_pairs"]):
            if i < len(pareto_figs):
                json_path = f"{figures_folder}/{exp.name}_pareto_frontier_{pair[0]}_vs_{pair[1]}.json"
                with open(json_path, "w") as f:
                    f.write(pareto_figs[i].to_json())

    html = f"""<!DOCTYPE html><html lang="en" data-theme="light"><head>
        <title>Pareto frontier analysis for {exp.name} experiment</title>
        {HTML_CARD_HEADER}
        </head><body>
        <h1>Hypervolume trace</h1>
        <p>Hypervolume (HV) measures the volume of objective space
        dominated by the Pareto frontier relative to a reference point.</p>
        <p>A flat HV trace indicates that the frontier isn't improving much;
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
    html_path = f"{artifacts_folder}/{exp.name}_pareto_frontier.html"
    save_html(html, html_path, open_browser=open_html)
    log.info("==================== End ========================")
    return figures


def plot_streaming_metrics(cfg: DictConfig, client: Client):
    """Compute streaming data and save as interactive HTML."""
    streaming_data = compute_streaming_data(client, cfg)
    if not streaming_data["metrics"]:
        return

    fig = plot_streaming_figures(streaming_data)
    if fig is None:
        return

    exp_name = streaming_data["experiment_name"]
    artifacts_folder = cfg["optimization"]["case_runner"]["artifacts_folder"]
    os.makedirs(artifacts_folder, exist_ok=True)

    out_path = f"{artifacts_folder}/{exp_name}_streaming_metrics.html"
    html = figure_to_html(fig, "Streaming Metrics", exp_name)
    save_html(html, out_path)


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
        # metric_signatures / metric_names (backward compat)
        sigs = es_cfg.get("metric_signatures") or es_cfg.get("metric_names") or []
        for sig in sigs:
            thresholds[sig] = {
                "type": "threshold",
                "value": es_cfg.get("metric_threshold"),
                "min_progression": es_cfg.get("min_progression", 0),
            }
    elif t == "percentile":
        sigs = es_cfg.get("metric_signatures") or es_cfg.get("metric_names") or []
        for sig in sigs:
            thresholds[sig] = {
                "type": "percentile",
                "percentile": es_cfg.get("percentile_threshold"),
                "min_progression": es_cfg.get("min_progression", 0),
            }


def _resolve_percentile_thresholds(metrics: dict, thresholds: dict):
    """For percentile-type thresholds, compute the actual percentile curve
    from completed trial data so the dashboard can draw it."""
    import numpy as np
    for metric_name, th in thresholds.items():
        if th.get("type") != "percentile":
            continue
        per_trial = metrics.get(metric_name, {})
        if not per_trial:
            continue
        pct = th["percentile"]
        # Collect all completed trials' final-step values to build a reference,
        # then compute per-step percentile across all trials
        # Gather values at each step across all trials
        step_vals: dict[float, list] = {}
        for td in per_trial.values():
            steps = td.get("steps", [])
            values = td.get("values", [])
            for s, v in zip(steps, values):
                step_vals.setdefault(s, []).append(v)
        if not step_vals:
            continue
        sorted_steps = sorted(step_vals.keys())
        pct_steps = []
        pct_values = []
        for s in sorted_steps:
            vals = step_vals[s]
            if len(vals) >= 2:  # need at least 2 trials for a meaningful percentile
                pct_steps.append(s)
                pct_values.append(float(np.percentile(vals, pct)))
        if pct_steps:
            th["resolved_steps"] = pct_steps
            th["resolved_values"] = pct_values


# ---------------------------------------------------------------------------
# Specialization Estimate — GP-based convergence & specialization cost
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


def _get_model_bounds(sub_model):
    """Derive bounds from the model's training data (handles extra dims like 'step')."""
    import torch
    X = sub_model.train_inputs[0]  # (n, d)
    lo = X.min(dim=0).values.tolist()
    hi = X.max(dim=0).values.tolist()
    # Widen bounds slightly to allow exploration beyond training range
    return [[l - 0.01 * max(abs(h - l), 1e-6), h + 0.01 * max(abs(h - l), 1e-6)]
            for l, h in zip(lo, hi)]


def _get_objective_models(gs):
    """Extract per-objective sub-models from the surrogate (ModelListGP)."""
    surr = getattr(gs.adapter.generator, "surrogate", None)
    if surr is None or surr.model is None:
        return None, None
    model = surr.model
    outcomes = getattr(surr, "_outcomes", None) or []
    if hasattr(model, "models"):
        return model.models, outcomes
    return [model], outcomes


def _compute_max_pi_for_model(
    sub_model, bounds, best_f, minimize, fixed_features=None,
    n_restarts=8, raw_samples=64,
):
    """Compute max Probability of Improvement for a single-output GP model.

    Args:
        fixed_features: dict of {dim_index: value} to fix during optimization.
            Used to pin context dimensions to specific values.
    """
    import torch
    from botorch.acquisition.analytic import ProbabilityOfImprovement
    from botorch.optim import optimize_acqf

    pi_acqf = ProbabilityOfImprovement(
        model=sub_model, best_f=best_f, maximize=not minimize,
    )
    bounds_t = torch.tensor(bounds, dtype=torch.double).T  # (2, d)

    # Fix context dimensions by collapsing their bounds
    if fixed_features:
        for dim_idx, val in fixed_features.items():
            bounds_t[0, dim_idx] = val
            bounds_t[1, dim_idx] = val

    try:
        _, max_pi_val = optimize_acqf(
            acq_function=pi_acqf,
            bounds=bounds_t,
            q=1,
            num_restarts=n_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
        )
        return float(max_pi_val)
    except Exception as e:
        _log.debug("PI optimization failed: %s", e)
        return None


def _simulate_convergence(model, bounds, best_f, minimize, threshold, max_steps=30):
    """Simulate BO convergence by fantasizing observations at EI-optimal points.

    At each step:
    1. Find x* = argmax ExpectedImprovement
    2. Fantasize observation y* = posterior_mean(x*) at x*
    3. Update best_f if y* improves
    4. Compute max PI on the fantasy model
    5. Stop if max PI < threshold

    Returns the number of simulated steps to convergence.
    """
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement, ProbabilityOfImprovement
    from botorch.optim import optimize_acqf

    bounds_t = torch.tensor(bounds, dtype=torch.double).T
    current_model = model
    current_best = best_f

    _log.info("  Simulating convergence (max %d steps, threshold=%.3f)...", max_steps, threshold)
    for step in range(1, max_steps + 1):
        try:
            # Find where EI is maximized
            ei = LogExpectedImprovement(
                model=current_model, best_f=current_best, maximize=not minimize,
            )
            candidate, _ = optimize_acqf(
                ei, bounds=bounds_t, q=1, num_restarts=4, raw_samples=32,
            )

            # Fantasize: get posterior mean at candidate as the "observation"
            with torch.no_grad():
                posterior = current_model.posterior(candidate)
                y_fantasy = posterior.mean.squeeze(-1)

            # Update best_f
            y_val = y_fantasy.item()
            improved = (minimize and y_val < current_best) or (not minimize and y_val > current_best)
            if improved:
                current_best = y_val

            # Create fantasy model (condition on the new observation)
            current_model = current_model.condition_on_observations(
                X=candidate, Y=y_fantasy,
            )

            # Check max PI on the fantasy model
            pi = ProbabilityOfImprovement(
                model=current_model, best_f=current_best, maximize=not minimize,
            )
            _, max_pi = optimize_acqf(
                pi, bounds=bounds_t, q=1, num_restarts=4, raw_samples=32,
            )
            pi_val = max_pi.item()

            if step % 50 == 0 or step == 1:
                _log.info("    step %d: PI=%.4f best_f=%.4f%s",
                           step, pi_val, current_best, " *" if improved else "")

            if pi_val < threshold:
                _log.info("  Converged after %d simulated trials", step)
                return step

        except Exception as e:
            _log.debug("Simulation step %d failed: %s", step, e)
            return max_steps

    _log.info("  Did not converge in %d steps (final PI=%.4f)", max_steps, pi_val)
    return max_steps


def _find_observed_best_f(sub_model, minimize):
    """Get the best observed value from the model's training targets.

    Uses the actual training data (not GP prediction) so PI measures
    the probability of improving over what we've actually seen.
    """
    import torch
    Y = sub_model.train_targets  # (n,)
    if minimize:
        return float(Y.min().item())
    else:
        return float(Y.max().item())


def _estimate_trials_remaining(pi_history, threshold):
    """Fit exponential decay to PI history and extrapolate to threshold crossing."""
    import numpy as np
    vals = np.array(pi_history)
    if len(vals) < 3:
        return None
    # Only use positive values for log fit
    mask = vals > 0
    if mask.sum() < 3:
        return None
    x = np.arange(len(vals))[mask]
    y = np.log(vals[mask])
    # Linear fit on log(PI) = a - b*t
    try:
        b, a = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if b >= 0:  # PI not decaying
        return None
    # Solve: a + b*t = log(threshold)
    if threshold <= 0:
        return None
    t_cross = (np.log(threshold) - a) / b
    remaining = max(0, int(np.ceil(t_cross - len(vals) + 1)))
    return min(remaining, 999)


def compute_convergence_pi(client, gs, improvement_bar=0.1):
    """Part A: Per-objective max PI for convergence tracking.

    Returns dict with per-metric estimates and metadata.
    """
    import torch
    from ax.core.objective import MultiObjective

    models, outcomes = _get_objective_models(gs)
    if models is None:
        return {"estimates": {}, "error": "No fitted model"}

    exp = client._experiment
    opt_config = exp.optimization_config

    # Derive bounds from the model's actual training data to match its input dim
    # (includes extra features like 'step' added by Ax transforms)
    bounds = _get_model_bounds(models[0])

    # Get objective info
    if isinstance(opt_config.objective, MultiObjective):
        obj_info = {o.metric.name: o.minimize for o in opt_config.objective.objectives}
    else:
        obj_info = {opt_config.objective.metric.name: opt_config.objective.minimize}

    # Disable context transforms for clean per-point PI
    from .api_server import _disable_context_transforms, _restore_context_transforms
    _disabled = _disable_context_transforms(gs)

    estimates = {}
    try:
        for i, (sub_model, outcome_name) in enumerate(zip(models, outcomes)):
            if outcome_name not in obj_info:
                continue
            minimize = obj_info[outcome_name]

            # Find GP-predicted best
            best_f = _find_observed_best_f(sub_model, minimize)
            if best_f is None:
                estimates[outcome_name] = {"status": "insufficient_data", "max_pi": None}
                continue

            # For minimize objectives, PosteriorMean returns the minimum but PI needs
            # the best_f in the model's native scale. Use the raw posterior mean value.
            max_pi = _compute_max_pi_for_model(
                sub_model, bounds, best_f, minimize,
            )
            if max_pi is None:
                estimates[outcome_name] = {"status": "insufficient_data", "max_pi": None}
                continue

            if max_pi < improvement_bar:
                status = "converged"
            else:
                status = "improving"

            estimates[outcome_name] = {
                "status": status,
                "max_pi": round(max_pi, 4),
                "best_so_far": round(best_f, 6),
                "convergence_pct": round((1 - min(max_pi, 1.0)) * 100, 1),
            }
    finally:
        _restore_context_transforms(_disabled)

    n_completed = sum(1 for t in exp.trials.values()
                      if t.status.name == "COMPLETED")
    return {
        "trial": n_completed,
        "estimates": estimates,
        "pi_threshold": improvement_bar,
    }


def compute_specialization_cost(client, gs, context_point, improvement_bar=0.1):
    """Part B: Per-objective PI at fixed context using a specialized GP.

    Fits a fresh GP on training data near the specified context point
    (no context transforms), then computes max PI on it. This gives
    a more accurate estimate than using the robust GP with fixed features.
    """
    import torch
    import numpy as np
    from ax.core.objective import MultiObjective
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    models, outcomes = _get_objective_models(gs)
    if models is None:
        return {"estimates": {}, "error": "No fitted model"}

    exp = client._experiment
    opt_config = exp.optimization_config

    if isinstance(opt_config.objective, MultiObjective):
        obj_info = {o.metric.name: o.minimize for o in opt_config.objective.objectives}
    else:
        obj_info = {opt_config.objective.metric.name: opt_config.objective.minimize}

    # Get feature ordering from the adapter
    param_names = getattr(gs.adapter, 'parameters', [])
    ctx_indices = [param_names.index(cn) for cn in context_point if cn in param_names]
    ctx_values_raw = torch.tensor(
        [float(context_point[param_names[idx]]) for idx in ctx_indices],
        dtype=torch.double,
    )

    estimates = {}
    for i, (robust_model, outcome_name) in enumerate(zip(models, outcomes)):
        if outcome_name not in obj_info:
            continue
        minimize = obj_info[outcome_name]

        try:
            # Get training data from the robust model
            X_all = robust_model.train_inputs[0]   # (n, d)
            Y_all = robust_model.train_targets      # (n,)

            # Filter: select trials near the target context using the GP's
            # lengthscales to define "nearby" (within 2 lengthscales).
            if ctx_indices:
                ctx_X = X_all[:, ctx_indices]  # (n, n_ctx)
                # Map raw context values to training data space
                param_lo = torch.tensor([
                    float(exp.search_space.parameters[param_names[idx]].lower)
                    for idx in ctx_indices], dtype=torch.double)
                param_hi = torch.tensor([
                    float(exp.search_space.parameters[param_names[idx]].upper)
                    for idx in ctx_indices], dtype=torch.double)
                ctx_lo = ctx_X.min(dim=0).values
                ctx_hi = ctx_X.max(dim=0).values
                ctx_range = (ctx_hi - ctx_lo).clamp(min=1e-6)
                param_range = (param_hi - param_lo).clamp(min=1e-6)
                ctx_target = ctx_lo + (ctx_values_raw - param_lo) / param_range * ctx_range

                # Select the closest N trials to the target context,
                # where N = total_trials / n_context_scenarios.
                # This gives each context point its fair share of data.
                param_lo = torch.tensor([
                    float(exp.search_space.parameters[param_names[idx]].lower)
                    for idx in ctx_indices], dtype=torch.double)
                param_hi = torch.tensor([
                    float(exp.search_space.parameters[param_names[idx]].upper)
                    for idx in ctx_indices], dtype=torch.double)
                param_range = (param_hi - param_lo).clamp(min=1e-6)
                ctx_target = ctx_lo + (ctx_values_raw - param_lo) / param_range * ctx_range
                dist = ((ctx_X - ctx_target) / ctx_range).norm(dim=1)

                # n_context_scenarios from the robust config (not from unique data)
                n_scenarios = max(1, len(context_point))  # at least split by context dims
                # Select 1/4 of data — enough to fit a meaningful GP
                n_select = max(10, len(X_all) // 4)
                _, nearest_idx = dist.topk(min(n_select, len(dist)), largest=False)
            else:
                nearest_idx = torch.arange(len(X_all))

            X_sub = X_all[nearest_idx]
            Y_sub = Y_all[nearest_idx]
            n_sub = len(X_sub)

            if n_sub < 3:
                estimates[outcome_name] = {
                    "max_pi_specialized": None,
                    "estimated_specialized_trials": None,
                    "n_nearby_trials": n_sub,
                }
                continue

            # Drop context dims from features (specialized model doesn't need them)
            keep_dims = [j for j in range(X_sub.shape[1]) if j not in ctx_indices]
            X_spec = X_sub[:, keep_dims]
            d_spec = X_spec.shape[1]

            ctx_str = ", ".join(f"{k}={v:.4g}" for k, v in context_point.items())
            _log.info("Specialization: %s @ [%s] (minimize=%s, %d nearby points, %d design dims)",
                      outcome_name, ctx_str, minimize, n_sub, d_spec)
            # Fit a fresh specialized GP
            spec_model = SingleTaskGP(
                X_spec, Y_sub.unsqueeze(-1),
                input_transform=Normalize(d=d_spec),
            )
            mll = ExactMarginalLogLikelihood(spec_model.likelihood, spec_model)
            fit_gpytorch_mll(mll)

            # Compute best_f from the specialized training data
            best_f = float(Y_sub.min().item()) if minimize else float(Y_sub.max().item())

            # Compute max PI on the specialized model
            spec_bounds = _get_model_bounds(spec_model)
            max_pi = _compute_max_pi_for_model(
                spec_model, spec_bounds, best_f, minimize,
            )

            if max_pi is None:
                estimates[outcome_name] = {
                    "max_pi_specialized": None,
                    "estimated_specialized_trials": None,
                    "n_nearby_trials": n_sub,
                }
                continue

            # Estimate remaining trials via BO simulation on the specialized GP.
            # Repeatedly: find argmax EI, fantasize an observation, recompute PI.
            if max_pi > improvement_bar:
                est_trials = _simulate_convergence(
                    spec_model, spec_bounds, best_f, minimize,
                    improvement_bar, max_steps=30,
                )
            else:
                est_trials = 0

            estimates[outcome_name] = {
                "max_pi_specialized": round(max_pi, 4),
                "estimated_specialized_trials": est_trials,
                "n_nearby_trials": n_sub,
            }
        except Exception as e:
            _log.debug("Specialization cost failed for %s: %s", outcome_name, e)
            estimates[outcome_name] = {
                "max_pi_specialized": None,
                "estimated_specialized_trials": None,
            }

    return {
        "context_point": context_point,
        "estimates": estimates,
        "pi_threshold": improvement_bar,
    }


def format_convergence_log(convergence_result, specialization_result=None):
    """Format convergence + specialization for stdout logging."""
    lines = []
    trial = convergence_result.get("trial", "?")
    lines.append(f"--- Convergence (trial {trial}) ---")
    for metric, est in convergence_result.get("estimates", {}).items():
        status = est.get("status", "?")
        pi = est.get("max_pi")
        pct = est.get("convergence_pct")
        if status == "converged":
            lines.append(f"  {metric:20s}: converged  (PI={pi:.3f})")
        elif status == "improving":
            remaining = est.get("estimated_trials_remaining", "?")
            lines.append(f"  {metric:20s}: improving  (PI={pi:.3f}, {pct:.0f}% converged)")
        else:
            lines.append(f"  {metric:20s}: {status}")

    if specialization_result and specialization_result.get("estimates"):
        ctx = specialization_result.get("context_point", {})
        ctx_str = ", ".join(f"{k}={v:.4g}" for k, v in ctx.items())
        lines.append(f"\n--- Specialization Cost (at context: {ctx_str}) ---")
        for metric, est in specialization_result.get("estimates", {}).items():
            pi = est.get("max_pi_specialized")
            trials = est.get("estimated_specialized_trials")
            if pi is None:
                lines.append(f"  {metric:20s}: insufficient data")
            elif trials == 0:
                lines.append(f"  {metric:20s}: already specialized  (PI={pi:.3f})")
            else:
                lines.append(f"  {metric:20s}: ~{trials} additional specialized trials  (PI={pi:.3f})")

    return "\n".join(lines)
