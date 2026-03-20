#!/usr/bin/env python3
"""
Single-objective F1 optimization using foamBO's fluent API.
No YAML, no shell commands, no OpenFOAM case files needed.

Demonstrates:
- Pure-Python metric evaluation via fn=
- Custom GP kernel setup and trial data transforms' effects
- Cross-validation of the fitted surrogate
- Visualization with a custom sensitivity callback
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import logging
import numpy as np
from benchmark import F1, foambo_metric
from foambo import FoamBO
from gpytorch.kernels import ScaleKernel, MaternKernel, AdditiveKernel

log = logging.getLogger("foambo.example")

class AdditiveMatern(AdditiveKernel):
    """Additive kernel: smooth trend (Matern-2.5) + rough detail (Matern-0.5)."""
    def __init__(self, **kwargs):
        super().__init__(
            ScaleKernel(MaternKernel(nu=2.5)),
            ScaleKernel(MaternKernel(nu=0.5)),
        )

client = (
    FoamBO("SingleObjF1")
    .parameter("x", bounds=[-100.0, 200.0])
    .minimize("F1", fn=foambo_metric("F1", k=1, m=0, lb=0.01))
    .kernel(AdditiveMatern)
    #.transforms(exclude=["BilogY", "Winsorize"])
    .stop(max_trials=50, improvement_bar=0.01, min_trials=25, window_size=10)
    .run(parallelism=3, ttl=10)
)

if client is None:
    log.warning("Optimization was interrupted.")
    sys.exit(1)

log.info("Surrogate model predictions (mean +/- SEM):")
from ax.core.observation import ObservationFeatures
adapter = client._generation_strategy.adapter
test_x = [-50, 0, 10, 50, 100, 150]
obs_feats = [ObservationFeatures(parameters={"x": float(x)}) for x in test_x]
f_mean, f_cov = adapter.predict(obs_feats)
for i, x in enumerate(test_x):
    mean = f_mean["F1"][i]
    sem = np.sqrt(f_cov["F1"]["F1"][i])
    log.info(f"  x={x:>5}: F1 = {mean:.2f} +/- {sem:.2f}")

log.info("Cross-validation (observed vs predicted):")
cv = FoamBO.cross_validate(client)
errors = []
for row in cv:
    err = abs(row["observed_mean"] - row["predicted_mean"])
    errors.append(err)
    log.info(f"  Trial {row['trial_index']:>2}: observed={row['observed_mean']:>8.2f}  "
             f"predicted={row['predicted_mean']:>8.2f}  error={err:.2f}")
log.info(f"  Mean absolute error: {np.mean(errors):.3f}")


# Visualizer with custom F1 plot callback
def plot_f1_with_trials(parameters: dict) -> str:
    """Plot the true F1 curve overlaid with all trial points."""
    # this callback should return a base64 image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64

    x_curve = np.linspace(-100, 200, 500)
    y_curve = F1(x_curve, k=1, m=0, lb=0.01)

    # Collect trial data
    df = client._experiment.fetch_data().df
    trial_x, trial_y = [], []
    for trial in client._experiment.trials.values():
        tx = trial.arm.parameters["x"]
        ty = df.loc[(df["trial_index"] == trial.index) &
                    (df["metric_name"] == "F1"), "mean"].values
        if len(ty):
            trial_x.append(tx)
            trial_y.append(ty[-1])

    # GP surrogate from the Ax model (adapter handles all transforms)
    gp_mean, gp_sem, gp_x = None, None, np.linspace(-100, 200, 200)
    try:
        from ax.core.observation import ObservationFeatures
        adapter = client._generation_strategy.adapter
        obs_feats = [ObservationFeatures(parameters={"x": float(xi)}) for xi in gp_x]
        f_mean, f_cov = adapter.predict(obs_feats)
        gp_mean = np.array(f_mean["F1"])
        gp_sem = np.array([np.sqrt(max(0, v)) for v in f_cov["F1"]["F1"]])
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 4))
    if gp_mean is not None:
        ax.fill_between(gp_x, gp_mean - 2*gp_sem, gp_mean + 2*gp_sem,
                        alpha=0.15, color="orange", label="GP ± 2σ")
        ax.plot(gp_x, gp_mean, "orange", linewidth=1.5, alpha=0.8, label="GP mean")
    ax.plot(x_curve, y_curve, "b-", alpha=0.5, label="True F1")
    ax.scatter(trial_x, trial_y, c="red", s=40, zorder=5, label="Trials")

    # Mark the queried point
    qx = parameters.get("x")
    if qx is not None:
        qy = float(F1(qx, k=1, m=0, lb=0.01))
        ax.axvline(qx, color="green", linestyle="--", alpha=0.7)
        ax.scatter([qx], [qy], c="green", s=100, marker="*", zorder=6, label=f"Query x={qx:.1f}")

    ax.set_xlabel("x")
    ax.set_ylabel("F1")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=5, borderaxespad=0, frameon=False)
    fig.suptitle("F1 Benchmark: true function vs surrogate model", y=1.12, fontsize=12)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

log.info("Launching visualizer with F1 overlay plot...")
log.info("(Press Ctrl+C to stop the server)")

# Reloading the client to run the visualizer UI through .show()
FoamBO.load("SingleObjF1").show(sensitivity_fn=plot_f1_with_trials)
