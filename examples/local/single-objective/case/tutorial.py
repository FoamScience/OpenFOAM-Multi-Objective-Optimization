#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import plotly.express as px

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.service.utils.best_point import get_best_parameters_from_model_predictions
from ax.modelbridge.registry import Models
from ax.service.utils.report_utils import exp_to_df
from ax.utils.notebook.plotting import render

from benchmark import Fs

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--F', type=str, help='Function to test')
parser.add_argument('--k', type=float, help='k')
parser.add_argument('--m', type=float, help='m')
parser.add_argument('--lb', type=float, help='lambda')
args = parser.parse_args()

def evaluate(parameters):
    ev = Fs[args.F](np.array([parameters["x1"]]),args.k,args.m,args.lb)
    return {args.F: (ev[0], 0.0)}

ax_client = AxClient()
ax_client.create_experiment(
    name="moo_experiment",
    parameters=[
        {
            "name": f"x{i+1}",
            "type": "range",
            "bounds": [-200.0, 200.0],
        }
        for i in range(1)
    ],
    objectives={
        args.F: ObjectiveProperties(minimize=True, threshold=None), 
    },
    overwrite_existing_experiment=True,
    is_test=True,
)

for i in range(400):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

ax_client.get_trials_data_frame().to_csv("trials.csv")

print("==========================================")
ax_client.fit_model()
render(ax_client.get_optimization_trace())
print("==========================================")
xs = [x for x in range(-200,200,30)]
preds = ax_client.get_model_predictions_for_parameterizations([{"x1": x} for x in xs])
calcs = Fs[args.F](np.array(xs),args.k,args.m,args.lb)
df = pd.DataFrame({'x':xs,
                   'F_pred':[p[args.F][0] for p in preds],
                   'F_sems':[p[args.F][1] for p in preds],
                   'F':[c for c in calcs]})
print("==========================================")
df.to_csv("preds.csv")
print(exp_to_df(ax_client.experiment))
