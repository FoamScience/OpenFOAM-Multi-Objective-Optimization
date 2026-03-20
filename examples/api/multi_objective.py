#!/usr/bin/env python3
"""
Multi-objective optimization using foamBO's fluent API.
Minimizes F1 and F3 simultaneously on a single parameter x.
F1 and (shifted) F3 have different structures, creating a non-trivial Pareto frontier.

Demonstrates:
- Multi-objective optimization with fn= callables
- Custom kernel for better surrogate fit
- Pareto frontier access
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import logging
import numpy as np
from benchmark import F1, F3, foambo_metric
from foambo import FoamBO
from gpytorch.kernels import ScaleKernel, MaternKernel, AdditiveKernel

log = logging.getLogger("foambo.example")


class AdditiveMatern(AdditiveKernel):
    """Additive kernel: smooth trend + rough detail."""
    def __init__(self, **kwargs):
        super().__init__(
            ScaleKernel(MaternKernel(nu=2.5)),
            ScaleKernel(MaternKernel(nu=0.5)),
        )

# we keep a single parameter for plotting simplicity
client = (
    FoamBO("MultiObjF1F3")
    .parameter("x", bounds=[-100.0, 200.0])
    .minimize("F1", fn=lambda p: float(F1(p["x"])))
    .minimize("F3", fn=lambda p: float(F3(p["x"] - 50)))
    .kernel(AdditiveMatern)
    .transforms(exclude=["BilogY", "Winsorize"])
    .stop(max_trials=60, improvement_bar=0.01, min_trials=30, window_size=10)
    .run(parallelism=3, ttl=10)
)

if client is None:
    log.warning("Optimization was interrupted.")
    sys.exit(1)

log.info("Pareto frontier:")
try:
    front = client.get_pareto_frontier(use_model_predictions=True)
    for params, prediction, arm_name, _ in front:
        log.info(f"  {arm_name}: x={params['x']:>7.2f} "
                 f"-> F1={prediction['F1'][0]:>7.2f}, F3={prediction['F3'][0]:>7.2f}")
except Exception as e:
    log.warning(f"Pareto frontier not available: {e}")
    front = client.get_pareto_frontier(use_model_predictions=False)
    for params, prediction, arm_name, _ in front:
        log.info(f"  {arm_name}: x={params['x']:>7.2f} "
                 f"-> F1={prediction['F1'][0]:>7.2f}, F3={prediction['F3'][0]:>7.2f}")

# Best point for each objective
df = client._experiment.fetch_data().df
for metric in ["F1", "F3"]:
    best = min(client._experiment.trials.values(), key=lambda t:
        df.loc[(df['trial_index']==t.index)&(df['metric_name']==metric),'mean'].values[-1]
        if len(df.loc[(df['trial_index']==t.index)&(df['metric_name']==metric),'mean'].values) else 999)
    val = df.loc[(df['trial_index']==best.index)&(df['metric_name']==metric),'mean'].values[-1]
    log.info(f"Best for {metric}: x={best.arm.parameters['x']:.2f}, {metric}={val:.4f}")

try:
    cv = FoamBO.cross_validate(client)
    for metric in ["F1", "F3"]:
        errors = [abs(r["observed_mean"] - r["predicted_mean"])
                  for r in cv if r["metric_name"] == metric]
        if errors:
            log.info(f"Cross-validation MAE for {metric}: {np.mean(errors):.3f}")
except RuntimeError as e:
    log.warning(f"Cross-validation skipped: {e}")

# Visualizer with a callback showing true/GP tradeoff curves + pareto frontier
def plot_objectives(parameters: dict) -> str:
    """2x2 plot: true functions, GP surrogates with confidence, trials, Pareto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64

    qx = parameters.get("x")

    # Collect trial data
    df = client._experiment.fetch_data().df
    trial_x, trial_f1, trial_f3 = [], [], []
    for trial in client._experiment.trials.values():
        tx = trial.arm.parameters["x"]
        f1v = df.loc[(df["trial_index"] == trial.index) & (df["metric_name"] == "F1"), "mean"].values
        f3v = df.loc[(df["trial_index"] == trial.index) & (df["metric_name"] == "F3"), "mean"].values
        if len(f1v) and len(f3v):
            trial_x.append(tx)
            trial_f1.append(f1v[-1])
            trial_f3.append(f3v[-1])

    # Dense x grids — finer for true functions, coarser for GP
    x_fine = np.linspace(-100, 200, 500)
    x_true_f1 = [float(F1(xi)) for xi in x_fine]
    x_true_f3 = [float(F3(xi - 50)) for xi in x_fine]
    gp_x = np.linspace(-100, 200, 200)

    # GP predictions
    gp_f1_mean, gp_f1_sem, gp_f3_mean, gp_f3_sem = None, None, None, None
    try:
        from ax.core.observation import ObservationFeatures
        adapter = client._generation_strategy.adapter
        obs_feats = [ObservationFeatures(parameters={"x": float(xi)}) for xi in gp_x]
        fm, fc = adapter.predict(obs_feats)
        gp_f1_mean = np.array(fm["F1"])
        gp_f3_mean = np.array(fm["F3"])
        gp_f1_sem = np.array([np.sqrt(max(0, v)) for v in fc["F1"]["F1"]])
        gp_f3_sem = np.array([np.sqrt(max(0, v)) for v in fc["F3"]["F3"]])
    except Exception:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Plot (0,0): F1 — true + GP + trials
    ax = axes[0, 0]
    ax.plot(x_fine, x_true_f1, "b-", alpha=0.4, label="True F1")
    if gp_f1_mean is not None:
        ax.plot(gp_x, gp_f1_mean, "b--", linewidth=1.5, label="GP F1")
        ax.fill_between(gp_x, gp_f1_mean - 2*gp_f1_sem, gp_f1_mean + 2*gp_f1_sem,
                        alpha=0.15, color="blue", label="GP ± 2σ")
    ax.scatter(trial_x, trial_f1, c="red", s=25, zorder=5, label="Trials")
    if qx is not None:
        ax.axvline(qx, color="green", linestyle="--", alpha=0.7)
        ax.scatter([qx], [float(F1(qx))], c="lime", s=100, marker="*", zorder=6, edgecolors="black")
    ax.set_xlabel("x")
    ax.set_ylabel("F1")
    ax.set_title("F1(x): true vs GP (min near x≈0)")
    ax.legend(fontsize=7)

    # Plot (0,1): shifted F3 — true + GP + trials
    ax = axes[0, 1]
    ax.plot(x_fine, x_true_f3, "r-", alpha=0.4, label="True F3")
    if gp_f3_mean is not None:
        ax.plot(gp_x, gp_f3_mean, "r--", linewidth=1.5, label="GP F3")
        ax.fill_between(gp_x, gp_f3_mean - 2*gp_f3_sem, gp_f3_mean + 2*gp_f3_sem,
                        alpha=0.15, color="red", label="GP ± 2σ")
    ax.scatter(trial_x, trial_f3, c="red", s=25, zorder=5, label="Trials")
    if qx is not None:
        ax.axvline(qx, color="green", linestyle="--", alpha=0.7)
        ax.scatter([qx], [float(F3(qx - 50))], c="lime", s=100, marker="*", zorder=6, edgecolors="black")
    ax.set_xlabel("x")
    ax.set_ylabel("F3")
    ax.set_title("F3(x−50): true vs GP (min near x≈50)")
    ax.legend(fontsize=7)

    # Plot (1,0): Both objectives overlaid
    ax = axes[1, 0]
    ax.plot(x_fine, x_true_f1, "b-", alpha=0.3)
    ax.plot(x_fine, x_true_f3, "r-", alpha=0.3)
    if gp_f1_mean is not None:
        ax.plot(gp_x, gp_f1_mean, "b--", linewidth=1.5, label="GP F1")
        ax.fill_between(gp_x, gp_f1_mean - 2*gp_f1_sem, gp_f1_mean + 2*gp_f1_sem, alpha=0.1, color="blue")
    if gp_f3_mean is not None:
        ax.plot(gp_x, gp_f3_mean, "r--", linewidth=1.5, label="GP F3")
        ax.fill_between(gp_x, gp_f3_mean - 2*gp_f3_sem, gp_f3_mean + 2*gp_f3_sem, alpha=0.1, color="red")
    if qx is not None:
        ax.axvline(qx, color="green", linestyle="--", alpha=0.7, label=f"Query x={qx:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("Objective value")
    ax.set_title("Both objectives: GP surrogates overlaid")
    ax.legend(fontsize=7)

    # Plot (1,1): Pareto front (F1 vs F3)
    ax = axes[1, 1]
    # GP-predicted tradeoff curve, colored by x to show the sweep direction
    if gp_f1_mean is not None and gp_f3_mean is not None:
        ax.scatter(gp_f1_mean, gp_f3_mean, c=gp_x, cmap="coolwarm", s=3, alpha=0.5, zorder=2, label="GP tradeoff (colored by x)")
    # True tradeoff curve, colored by x
    sc = ax.scatter(x_true_f1, x_true_f3, c=x_fine, cmap="coolwarm", s=1, alpha=0.2, zorder=1)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="x value")
    ax.scatter(trial_f1, trial_f3, c="steelblue", s=30, alpha=0.6, zorder=3, label="Trials")
    try:
        front = client.get_pareto_frontier(use_model_predictions=True)
        pf1 = [p[1]["F1"][0] for p in front]
        pf3 = [p[1]["F3"][0] for p in front]
        ax.scatter(pf1, pf3, c="red", s=60, zorder=5, marker="D", label="Pareto")
    except Exception:
        pass
    if qx is not None:
        ax.scatter([float(F1(qx))], [float(F3(qx - 50))], c="lime", s=120, marker="*",
                   zorder=6, edgecolors="black", label=f"Query x={qx:.1f}")
    ax.set_xlabel("F1")
    ax.set_ylabel("F3")
    ax.set_title("Pareto frontier: F1 vs F3")
    ax.legend(fontsize=7)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


log.info("Launching visualizer with objectives callback...")
log.info("(Press Ctrl+C to stop the server)")
FoamBO.load("MultiObjF1F3").show(sensitivity_fn=plot_objectives)
