#!/usr/bin/env python3
"""Metric script for MF+robust test case.

Operating conditions (like flowRate/rpm in pump optimization):
  load  ∈ [0.8, 1.2]: scales F1 amplitude (higher load → worse F1)
  speed ∈ [0.9, 1.1]: shifts F2 optimum (speed change moves efficiency curve)

Fidelity model (Forrester-Sobester-Keane 2007 style):
  fidelity=1 (expensive): true F1 / F2
  fidelity=0 (cheap):     smoothed (σ=15) * A + B*(x - C)
  — Linear bias term shifts cheap minima away from expensive ones, so
  GP cannot rely on cheap alone to locate the true optimum. This makes
  qMFHVKG actually value expensive evaluations instead of exploiting
  the cheap surface (which happened when cheap was a near-unbiased
  smoothing).
"""
import sys, os, json
import numpy as np


def z(x, k=1, m=0, lb=0.01):
    cond = np.abs(x)/k - np.floor(np.abs(x)/k)
    return np.where(cond < lb, 1 - m + (m/lb)*cond, 1 - m + (m/(1-lb))*(1-cond))

def F1_raw(x):
    return 3e-9*np.abs((x-40)*(x-185)*x*(x+50)*(x+180))*z(x) + 10*np.abs(np.sin(0.1*x))

def F3_raw(x):
    return 3*np.abs(np.log(1000*np.abs(x)+1))*z(x) + 30 - 30*np.abs(np.cos(x/(10*np.pi)))

def smooth_eval(fn, x, sigma=15, n_samples=50):
    offsets = np.linspace(-3*sigma, 3*sigma, n_samples)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    vals = np.array([fn(x + dx) for dx in offsets])
    return float(np.dot(weights, vals))


# Forrester-style low-fidelity parameters (per objective).
# cheap(x) = A * smooth(x) + B * (x - C)
# A: amplitude scaling (|A|<1 → cheap under/over-predicts magnitude)
# B: linear drift slope (introduces systematic bias)
# C: pivot point (bias = 0 at x = C, grows linearly away)
CHEAP_F1 = {"A": 0.7, "B": 0.20, "C": 0.0}
CHEAP_F2 = {"A": 0.9, "B": 0.15, "C": 0.0}

FIDELITY_COST = {0: 0.7, 1: 1.0}


def cheap_eval(fn, x, params):
    return params["A"] * smooth_eval(fn, x) + params["B"] * (x - params["C"])

with open(os.path.join(os.getcwd(), "params.json")) as f:
    params = json.load(f)

x = float(params["x"])
load = float(params.get("load", 1.0))
speed = float(params.get("speed", 1.0))
fidelity = int(params.get("fidelity", 1))

# Operating condition effects:
# load scales F1 amplitude (like flowRate scaling pressure head)
# speed shifts F2 optimum by 10*(speed-1) (like rpm shifting efficiency curve)
speed_shift = 10.0 * (speed - 1.0)

metric_name = sys.argv[1]
if metric_name == "F1":
    if fidelity == 1:
        print(float(load * F1_raw(x)))
    else:
        print(load * cheap_eval(F1_raw, x, CHEAP_F1))
elif metric_name == "F2":
    if fidelity == 1:
        print(float(F3_raw(x - 50 + speed_shift)))
    else:
        # F2 pivots around x=50 (its expensive min); CHEAP_F2 "C"
        # offset is applied in the shifted domain via -50 already.
        print(cheap_eval(F3_raw, x - 50 + speed_shift, CHEAP_F2))
elif metric_name == "cost":
    print(FIDELITY_COST.get(fidelity, 1.0))
else:
    print(f"Unknown metric: {metric_name}", file=sys.stderr)
    sys.exit(1)
