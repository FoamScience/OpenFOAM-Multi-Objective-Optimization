#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [ "numpy", "foamlib" ]
# ///
"""
Benchmark functions from:
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9684455

Usable as a CLI (for foamBO YAML configs) or as a Python import (for library usage).
"""

import numpy as np


def z(x, k, m, lb):
    cond = np.abs(x)/k - np.floor(np.abs(x)/k)
    return np.where(cond < lb, 1 - m + (m/lb)*cond, 1 - m + (m/(1-lb))*(1-cond))

def F1(x, k=1, m=0, lb=0.01):
    return 3e-9*np.abs((x-40)*(x-185)*x*(x+50)*(x+180))*z(x,k,m,lb) + 10*np.abs(np.sin(0.1*x))

def F2(x, k=1, m=0, lb=0.01):
    return F1(F1(x,k,m,lb), k, m, lb)

def F3(x, k=1, m=0, lb=0.01):
    return 3*np.abs(np.log(1000*np.abs(x)+1))*z(x,k,m,lb) + 30 - 30*np.abs(np.cos(x/(10*np.pi)))

def F4(x, k=1, m=0, lb=0.01):
    return F3(F3(x,k,m,lb), k, m, lb)

Fs = {"F1": F1, "F2": F2, "F3": F3, "F4": F4}

# --- foamBO-compatible callable wrappers (parameters dict -> scalar)
def foambo_metric(func_name, k=1, m=0, lb=0.01):
    """Return a callable suitable for FoamBO.minimize(fn=...)."""
    fn = Fs[func_name]
    def metric(parameters):
        return float(fn(parameters["x"], k, m, lb))
    return metric

if __name__ == '__main__':
    import sys, os, argparse
    from foamlib import FoamFile

    parser = argparse.ArgumentParser()
    parser.add_argument('--F', type=str, help='Function to test')
    parser.add_argument('--k', type=float, help='k')
    parser.add_argument('--m', type=float, help='m')
    parser.add_argument('--lb', type=float, help='lambda')
    parser.add_argument('--n', type=int, help='progressions', default=1)
    parser.add_argument('--p', help='plot', default=False)
    args = parser.parse_args()

    if args.p:
        import plotly.express as px
        x = np.linspace(-200, 200, 1000)
        fig = px.line(x=x, y=Fs[args.F](x, args.k, args.m, args.lb))
        fig.update_layout(xaxis_title='x', yaxis_title=args.F, width=2048, height=768)
        fig.write_image(f"{args.F}.png")

    foamfile = FoamFile(os.path.join(os.getcwd(), "FxDict"))
    x = float(foamfile['x'])
    if args.n == 1:
        print(Fs[args.F](np.array([x, x]), args.k, args.m, args.lb)[0])
        sys.exit(0)
    from scipy import stats
    ll = [Fs[args.F](np.array([x, x]), args.k, args.m + np.random.normal(0, 1), args.lb)[0]
          for _ in range(args.n)]
    print(f"({np.mean(ll)}, {stats.sem(ll)})")
