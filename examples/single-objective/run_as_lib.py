#!/usr/bin/env python3
"""
Single-objective F1 optimization using foamBO as a Python library.
Equivalent to: uvx foamBO --config SOM.yaml
"""

from foambo import FoamBO

client = (
    FoamBO("SingleObjF1", case="./case", trials="./trials", artifacts="./artifacts")
    .parameter("x", bounds=[-100.0, 200.0])
    .minimize("F1", command="python3 ../../benchmark.py --F F1 --k 1 --m 0 --lb 0.01")
    .substitute("/FxDict", x="x")
    .stop(max_trials=50, improvement_bar=0.1, min_trials=10, window_size=5)
    .run(parallelism=3, poll_interval=10, ttl=10)
)

if client is not None:
    predictions = client.predict([{"x": 10}, {"x": 50}, {"x": 100}])
    print("\nSurrogate model predictions (mean, SEM):")
    for params, pred in zip([{"x": 10}, {"x": 50}, {"x": 100}], predictions):
        print(f"  x={params['x']:>5}: F1 = {pred['F1'][0]:.2f} ± {pred['F1'][1]:.2f}")
