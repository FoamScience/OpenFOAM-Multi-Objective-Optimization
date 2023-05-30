#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

pr = sys.argv[1] if len(sys.argv) > 1 else 4

dfexp = pd.read_csv("height_experiment.csv")
dfexp[f"smooth_H{pr}"] = savgol_filter(dfexp[f"H{pr}"], 100, 3)

df = pd.read_csv(f"waterHeight.csv")
df[f"H{pr}_exp"] = np.interp(df['Time'], dfexp['Time'], dfexp[f"smooth_H{pr}"])
rms = np.sqrt(mean_squared_error(df[f"H{pr}_exp"], df[f"H{pr}"]))
print(f"{rms:.4f}")

#df.plot(x="Time", y=[f"H{pr}_exp", f"H{pr}"], title=f"Water height at $H_{pr}$")
#plt.show()
