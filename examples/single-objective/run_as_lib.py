#!/usr/bin/env python3
"""
Single-objective F1 optimization using foamBO as a Python library.
No YAML, no shell commands, no OpenFOAM case files needed.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from benchmark import foambo_metric
from foambo import FoamBO

client = (
    FoamBO("SingleObjF1")
    .parameter("x", bounds=[-100.0, 200.0])
    .minimize("F1", fn=foambo_metric("F1", k=1, m=0, lb=0.01))
    .stop(max_trials=50, improvement_bar=0.1, min_trials=10, window_size=5)
    .run(parallelism=3, poll_interval=1, ttl=10)
)

if client is not None:
    predictions = client.predict([{"x": 10}, {"x": 50}, {"x": 100}])
    print("\nSurrogate model predictions (mean, SEM):")
    for params, pred in zip([{"x": 10}, {"x": 50}, {"x": 100}], predictions):
        print(f"  x={params['x']:>5}: F1 = {pred['F1'][0]:.2f} \u00b1 {pred['F1'][1]:.2f}")
