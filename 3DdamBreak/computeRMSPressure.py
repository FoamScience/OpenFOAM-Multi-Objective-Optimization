#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

pr = sys.argv[1] if len(sys.argv) > 1 else 1

df = pd.read_csv("pressure.csv")
dfexp = pd.read_csv("pressure_experiment.csv")
# correct P
origin = np.interp(df['Time'][0], dfexp['Time'], dfexp[f'P{pr}'])
diff = df[f"P{pr}"][0] - origin
df[f"P{pr}_c"] = df[f"P{pr}"] - diff
# interpolate corresponding elements from expriment
df[f"P{pr}_int"] = np.interp(df['Time'], dfexp['Time'], dfexp[f"P{pr}"])
# compute RMS
rms = np.sqrt(mean_squared_error(df[f"P{pr}_int"], df[f"P{pr}_c"]))
print(f"{rms:.4f}")
