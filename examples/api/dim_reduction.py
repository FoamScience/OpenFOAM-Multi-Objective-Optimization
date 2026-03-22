#!/usr/bin/env python3
"""
Dimensionality reduction demo.

Three parameters (x, y, z) but only x affects the objectives.
y and z are pure noise — foamBO's .reduce() detects them as irrelevant
via Sobol sensitivity and fixes them mid-optimization.

Demonstrates:
- Multi-objective optimization with dummy parameters
- Automatic parameter screening via Sobol sensitivity indices
- Fixing low-importance parameters mid-optimization
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import logging
import numpy as np
from benchmark import F1, F3
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


# Optimization with dimensionality reduction
# x drives both objectives; y and z are dummy
client = (
    FoamBO("DimReductionMO")
    .parameter("x", bounds=[-100.0, 200.0])
    .parameter("y", bounds=[-50.0, 50.0])
    .parameter("z", bounds=[-20.0, 20.0])
    .minimize("F1", fn=lambda p: float(F1(p["x"])))
    .minimize("F3", fn=lambda p: float(F3(p["x"] - 50)))
    .kernel(AdditiveMatern)
    .transforms(exclude=["BilogY", "Winsorize"])
    .reduce(after_trials=10, min_importance=0.05, fix_at="best")
    .stop(max_trials=30, improvement_bar=0.01, min_trials=15, window_size=10)
    .run(parallelism=1, ttl=5)
)

if client is None:
    log.warning("Optimization was interrupted.")
    sys.exit(1)


exp = client._experiment
log.info("Parameter status after dimensionality reduction:")
for pname, param in exp.search_space.parameters.items():
    ptype = type(param).__name__
    if ptype == "FixedParameter":
        log.info(f"  FIXED:  {pname} = {param.value}")
    else:
        log.info(f"  ACTIVE: {pname} [{param.lower}, {param.upper}]")
df = client._experiment.fetch_data().df
for metric in ["F1", "F3"]:
    best = min(exp.trials.values(), key=lambda t:
        df.loc[(df['trial_index']==t.index)&(df['metric_name']==metric),'mean'].values[-1]
        if len(df.loc[(df['trial_index']==t.index)&(df['metric_name']==metric),'mean'].values) else 999)
    val = df.loc[(df['trial_index']==best.index)&(df['metric_name']==metric),'mean'].values[-1]
    log.info(f"Best for {metric}: x={best.arm.parameters['x']:.2f}, {metric}={val:.4f}")
