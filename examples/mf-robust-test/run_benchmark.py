#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [ "numpy", "plotly", "kaleido" ]
# ///
"""Run benchmark configs, compare results, and generate plots.

Usage:
    python3 run_benchmark.py [--run] [--compare] [--plot] [--configs ...]

Groups:
    base configs:    base, robust, mf_robust_spec_n0 (anchored MF), mf_robust_spec_n3
    specializations: robust_spec_n0, robust_spec_n3
    deprecated:      mf, mf_robust (no sobol_anchor — qMFHVKG stays at fid=0)
"""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np

# mf, mf_robust dropped from recommended set — they fail to escape cheap fidelity.
# Use mf_robust_spec_n0 / mf_robust_spec_n3 (sobol_anchor pins fid=1 init).
ALL_BASE = ["base", "robust", "mf_robust_spec_n0", "mf_robust_spec_n3"]
ALL_SPEC = ["robust_spec_n0", "robust_spec_n3"]
ALL_DEPRECATED = ["mf", "mf_robust"]

# --- Benchmark functions ---
def z(x, k=1, m=0, lb=0.01):
    cond = np.abs(x)/k - np.floor(np.abs(x)/k)
    return np.where(cond < lb, 1 - m + (m/lb)*cond, 1 - m + (m/(1-lb))*(1-cond))

def F1(x):
    return 3e-9*np.abs((x-40)*(x-185)*x*(x+50)*(x+180))*z(x) + 10*np.abs(np.sin(0.1*x))

def F3(x):
    return 3*np.abs(np.log(1000*np.abs(x)+1))*z(x) + 30 - 30*np.abs(np.cos(x/(10*np.pi)))

def F2_shifted(x):
    """F3 shifted by 50 to create Pareto tradeoff with F1."""
    return F3(x - 50)

def smooth_eval(fn, x, sigma=15, n_samples=50):
    """Gaussian-smoothed evaluation (used inside Forrester low-fi)."""
    offsets = np.linspace(-3*sigma, 3*sigma, n_samples)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    return float(np.dot(weights, np.array([fn(x + dx) for dx in offsets])))


# Forrester-style low-fidelity parameters (must match case/metric.py).
# cheap(x) = A * smooth(x) + B * (x - C)
CHEAP_F1 = {"A": 0.7, "B": 0.20, "C": 0.0}
CHEAP_F2 = {"A": 0.9, "B": 0.15, "C": 0.0}


def cheap_eval(fn, x, params):
    return params["A"] * smooth_eval(fn, x) + params["B"] * (x - params["C"])


REF_POINT = np.array([200.0, 70.0])

# --- Theory ---
def F1_op(x, load=1.0):
    """F1 with operating condition: load scales amplitude."""
    return load * F1(x)

def F2_op(x, speed=1.0):
    """F2 with operating condition: speed shifts optimum."""
    return F2_shifted(x + 10.0 * (speed - 1.0))

def compute_theoretical_pareto(n_points=2000, load_range=(0.8, 1.2), speed_range=(0.9, 1.1), n_ops=10):
    x = np.linspace(-100, 200, n_points)
    loads = np.linspace(load_range[0], load_range[1], n_ops)
    speeds = np.linspace(speed_range[0], speed_range[1], n_ops)
    # All (x, load, speed) combos — best achievable over any operating point
    all_f1, all_f2 = [], []
    for xi in x:
        for ld in loads:
            for sp in speeds:
                all_f1.append(F1_op(xi, ld))
                all_f2.append(F2_op(xi, sp))
    # Nominal (load=1, speed=1) for landscape plots
    f1 = np.array([F1(xi) for xi in x])
    f2 = np.array([F2_shifted(xi) for xi in x])
    # Cheap fidelity (Forrester low-fi) at nominal — must match metric.py:
    #   F1_cheap(x) = A1*smooth(F1, x) + B1*(x - C1)
    #   F2_cheap(x) = A2*smooth(F3, x-50) + B2*((x-50) - C2)
    # Note: F2's cheap bias pivots in the *shifted* domain (x-50), matching
    # metric.py's `cheap_eval(F3_raw, x - 50 + speed_shift, CHEAP_F2)`.
    f1_cheap = np.array([cheap_eval(F1, xi, CHEAP_F1) for xi in x])
    f2_cheap = np.array([cheap_eval(F3, xi - 50.0, CHEAP_F2) for xi in x])
    # Robust: worst-case over operating conditions
    f1_r = np.array([max(F1_op(xi, ld) for ld in loads) for xi in x])
    f2_r = np.array([max(F2_op(xi, sp) for sp in speeds) for xi in x])
    return {"x": x,
            "nr": {"F1": f1, "F2": f2},
            "cheap": {"F1": f1_cheap, "F2": f2_cheap},
            "r": {"F1": f1_r, "F2": f2_r},
            "all": {"F1": np.array(all_f1), "F2": np.array(all_f2)}}

def get_pareto_front(points):
    """Extract non-dominated points from Nx2 array."""
    pts = points[points[:, 0].argsort()]
    pf = [pts[0]]
    for p in pts[1:]:
        if p[1] < pf[-1][1]:
            pf.append(p)
    return np.array(pf)

def compute_hypervolume(points, ref):
    mask = np.all(points < ref, axis=1)
    pts = points[mask]
    if len(pts) == 0:
        return 0.0
    pf = get_pareto_front(pts)
    hv = 0.0
    for i, p in enumerate(pf):
        w = (pf[i+1][0] if i+1 < len(pf) else ref[0]) - p[0]
        hv += w * (ref[1] - p[1])
    return hv

# --- Data loading ---
def load_results(results_dir):
    artifacts = Path(results_dir) / "artifacts"
    csv_files = sorted(artifacts.glob("*_report.csv"))
    if not csv_files:
        return None
    trial_data = {}
    with open(csv_files[-1]) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "COMPLETED":
                continue
            idx = int(row["trial_index"])
            if idx not in trial_data:
                trial_data[idx] = {
                    "idx": idx,
                    "x": float(row.get("x", 0)),
                    "load": float(row.get("load", 1.0)),
                    "speed": float(row.get("speed", 1.0)),
                    "fidelity": int(float(row.get("fidelity", 1))) if row.get("fidelity") else 1,
                    "cost": 1.0,
                }
            metric, val = row["metric_name"], float(row["mean"])
            if metric in ("F1", "F2"):
                trial_data[idx][metric] = val
            elif metric == "cost":
                trial_data[idx]["cost"] = val
    return sorted([t for t in trial_data.values() if "F1" in t and "F2" in t],
                  key=lambda t: t["idx"])

def target_trials(trials, name):
    """Filter to target-fidelity trials for MF configs."""
    if "mf" in name:
        return [t for t in trials if t.get("fidelity", 1) == 1]
    return trials

def cumulative_hv(trials, ref):
    hvs, pts = [], []
    for t in trials:
        pts.append([t["F1"], t["F2"]])
        hvs.append(compute_hypervolume(np.array(pts), ref))
    return hvs

# --- Runner ---
def run_configs(configs, base_dir):
    for name in configs:
        config_path = base_dir / "configs" / f"{name}.yaml"
        results_path = base_dir / "results" / name
        is_spec = "spec" in name
        if not is_spec:
            # Clean non-specialization runs
            if results_path.exists():
                shutil.rmtree(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        (results_path / "trials").mkdir(exist_ok=True)
        (results_path / "artifacts").mkdir(exist_ok=True)

        print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
        result = subprocess.run(
            ["uv", "run", "foamBO", "--config", str(config_path.resolve()), "--no-ui"],
            cwd=str(results_path),
        )
        if result.returncode != 0:
            print(f"  WARNING: {name} exited with code {result.returncode}")

# --- Text comparison ---
def compare(configs, base_dir):
    theory = compute_theoretical_pareto()
    hv_nr = compute_hypervolume(np.column_stack([theory["all"]["F1"], theory["all"]["F2"]]), REF_POINT)
    hv_r = compute_hypervolume(np.column_stack([theory["r"]["F1"], theory["r"]["F2"]]), REF_POINT)

    print(f"\n{'='*60}")
    print(f"THEORETICAL HV: best-case={hv_nr:.0f}, robust(worst-case)={hv_r:.0f}")
    print(f"{'='*60}\n")
    print(f"{'Config':<20} {'Trials':<8} {'HV':<10} {'HV%':<7} {'F1*':<9} {'F2*':<9} {'Cost':<7}")
    print("-" * 72)

    for name in configs:
        trials = load_results(base_dir / "results" / name)
        if not trials:
            print(f"{name:<20} N/A"); continue
        tt = target_trials(trials, name)
        if not tt:
            print(f"{name:<20} {len(trials):<8} (no target fidelity)"); continue
        pts = np.array([[t["F1"], t["F2"]] for t in tt])
        hv = compute_hypervolume(pts, REF_POINT)
        print(f"{name:<20} {len(trials):<8} {hv:<10.0f} {100*hv/hv_nr:<6.1f}% "
              f"{min(t['F1'] for t in tt):<9.3f} {min(t['F2'] for t in tt):<9.3f} "
              f"{sum(t['cost'] for t in trials):<7.1f}")

    # Convergence milestones
    print(f"\n{'='*60}\nCONVERGENCE (HV)\n{'='*60}")
    ms = [5, 10, 15, 20, 25, 30, 35, 40]
    print(f"{'Config':<20}" + "".join(f"{'T='+str(m):<9}" for m in ms))
    print("-" * (20 + 9 * len(ms)))
    for name in configs:
        trials = load_results(base_dir / "results" / name)
        if not trials: continue
        hvs = cumulative_hv(trials, REF_POINT)
        row = f"{name:<20}"
        for m in ms:
            row += f"{hvs[m-1]:<9.0f}" if m <= len(hvs) else f"{'—':<9}"
        print(row)

    # Robustness eval: test best x* across operating condition corners
    print(f"\n{'='*60}\nROBUSTNESS: x* at (load, speed) corners\n{'='*60}")
    op_test = [(0.8, 0.9), (0.8, 1.1), (1.0, 1.0), (1.2, 0.9), (1.2, 1.1)]
    print(f"{'Config':<20} {'x*':<7}" + "".join(f"{'L'+str(l)+'/S'+str(s):<14}" for l, s in op_test))
    print("-" * (27 + 14 * len(op_test)))
    for name in configs:
        trials = load_results(base_dir / "results" / name)
        if not trials: continue
        tt = target_trials(trials, name)
        if not tt: continue
        best = min(tt, key=lambda t: t["F1"] + t["F2"])
        row = f"{name:<20} {best['x']:<7.1f}"
        for ld, sp in op_test:
            row += f"{F1_op(best['x'],ld):.1f}/{F2_op(best['x'],sp):.1f}  "
        print(row)

# --- Plots ---
def plot_all(configs, base_dir):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out = base_dir / "results" / "plots"
    out.mkdir(exist_ok=True)
    theory = compute_theoretical_pareto()
    colors = {
        "base": "#636EFA", "robust": "#EF553B", "mf": "#00CC96", "mf_robust": "#AB63FA",
        "robust_spec_n0": "#FFA15A", "robust_spec_n3": "#FF6692",
        "mf_robust_spec_n0": "#B6E880", "mf_robust_spec_n3": "#19D3F3",
    }

    # 1. Pareto front comparison
    fig = go.Figure()
    # Theoretical front from all (x, noise) evaluations
    pf_pts = np.column_stack([theory["all"]["F1"], theory["all"]["F2"]])
    pf = get_pareto_front(pf_pts[np.all(pf_pts < REF_POINT, axis=1)])
    fig.add_trace(go.Scatter(x=pf[:, 0], y=pf[:, 1], mode="lines",
                             name="Theoretical PF (non-robust)", line=dict(color="gray", dash="dash")))
    # Robust theoretical PF
    pf_r_pts = np.column_stack([theory["r"]["F1"], theory["r"]["F2"]])
    pf_r_feas = pf_r_pts[np.all(pf_r_pts < REF_POINT, axis=1)]
    if len(pf_r_feas) > 1:
        pf_r = get_pareto_front(pf_r_feas)
        fig.add_trace(go.Scatter(x=pf_r[:, 0], y=pf_r[:, 1], mode="lines",
                                 name="Theoretical PF (robust)", line=dict(color="gray", dash="dot")))
    # Cheap fidelity PF (Forrester low-fi — what low-fi BO sees)
    pf_c_pts = np.column_stack([theory["cheap"]["F1"], theory["cheap"]["F2"]])
    pf_c_feas = pf_c_pts[np.all(pf_c_pts < REF_POINT, axis=1)]
    if len(pf_c_feas) > 1:
        pf_c = get_pareto_front(pf_c_feas)
        fig.add_trace(go.Scatter(x=pf_c[:, 0], y=pf_c[:, 1], mode="lines",
                                 name="Cheap fidelity PF (Forrester low-fi)",
                                 line=dict(color="lightblue", dash="dashdot", width=1)))
    for name in configs:
        trials = load_results(base_dir / "results" / name)
        if not trials: continue
        tt = target_trials(trials, name)
        if not tt: continue
        pts = np.array([[t["F1"], t["F2"]] for t in tt])
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers",
                                 name=name, marker=dict(color=colors.get(name, "black"), size=6)))
        # Add Pareto front line
        feasible = pts[np.all(pts < REF_POINT, axis=1)]
        if len(feasible) > 1:
            epf = get_pareto_front(feasible)
            fig.add_trace(go.Scatter(x=epf[:, 0], y=epf[:, 1], mode="lines",
                                     name=f"{name} PF", line=dict(color=colors.get(name), width=2),
                                     showlegend=False))
    fig.update_layout(title="Pareto Fronts", xaxis_title="F1 (minimize)", yaxis_title="F2 (minimize)",
                      width=900, height=700)
    fig.write_html(str(out / "pareto_fronts.html"))
    fig.write_image(str(out / "pareto_fronts.png"), scale=2)
    print(f"  Saved: {out / 'pareto_fronts.png'}")

    # 2. Convergence (HV over trials)
    fig = go.Figure()
    max_trials = 0
    for name in configs:
        trials = load_results(base_dir / "results" / name)
        if not trials: continue
        hvs = cumulative_hv(trials, REF_POINT)
        max_trials = max(max_trials, len(hvs))
        fig.add_trace(go.Scatter(y=hvs, mode="lines+markers", name=name,
                                 line=dict(color=colors.get(name)), marker=dict(size=3)))
    # Theoretical HV lines
    hv_nr = compute_hypervolume(np.column_stack([theory["all"]["F1"], theory["all"]["F2"]]), REF_POINT)
    hv_r = compute_hypervolume(np.column_stack([theory["r"]["F1"], theory["r"]["F2"]]), REF_POINT)
    fig.add_hline(y=hv_nr, line=dict(color="gray", dash="dash", width=1),
                  annotation_text=f"Theoretical (best-case): {hv_nr:.0f}")
    fig.add_hline(y=hv_r, line=dict(color="gray", dash="dot", width=1),
                  annotation_text=f"Theoretical (robust): {hv_r:.0f}")
    fig.update_layout(title="Hypervolume Convergence", xaxis_title="Trial", yaxis_title="HV",
                      width=900, height=500)
    fig.write_html(str(out / "convergence.html"))
    fig.write_image(str(out / "convergence.png"), scale=2)
    print(f"  Saved: {out / 'convergence.png'}")

    # 3. Objective landscape: F1(x) and F2(x) with trial x-values overlaid
    x_dense = theory["x"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["F1(x) landscape", "F2(x) = F3(x-50) landscape"])
    # Theoretical curves: expensive (true) + cheap (Forrester low-fi: A·smooth + B·(x−C))
    f1_exp = theory["nr"]["F1"]
    f2_exp = theory["nr"]["F2"]
    f1_cheap = theory["cheap"]["F1"]
    f2_cheap = theory["cheap"]["F2"]
    fig.add_trace(go.Scatter(x=x_dense, y=f1_cheap, mode="lines", name="F1 (cheap, Forrester low-fi)",
                             line=dict(color="lightblue", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_dense, y=f1_exp, mode="lines", name="F1 (expensive, true)",
                             line=dict(color="steelblue", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_dense, y=f2_cheap, mode="lines", name="F2 (cheap, Forrester low-fi)",
                             line=dict(color="lightsalmon", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_dense, y=f2_exp, mode="lines", name="F2 (expensive, true)",
                             line=dict(color="tomato", width=2)), row=2, col=1)
    # Overlay trial x-values
    for name in configs:
        trials = load_results(base_dir / "results" / name)
        if not trials: continue
        tt = target_trials(trials, name)
        if not tt: continue
        c = colors.get(name, "black")
        xs = [t["x"] for t in tt]
        f1s = [t["F1"] for t in tt]
        f2s = [t["F2"] for t in tt]
        fig.add_trace(go.Scatter(x=xs, y=f1s, mode="markers", name=name,
                                 marker=dict(color=c, size=5), legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=f2s, mode="markers", name=name,
                                 marker=dict(color=c, size=5), legendgroup=name,
                                 showlegend=False), row=2, col=1)
    fig.update_layout(title="Objective Landscapes with Trial Evaluations",
                      height=800, width=1000)
    fig.update_yaxes(title_text="F1(x)", row=1, col=1)
    fig.update_yaxes(title_text="F2(x)", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.write_html(str(out / "landscape.html"))
    fig.write_image(str(out / "landscape.png"), scale=2)
    print(f"  Saved: {out / 'landscape.png'}")

    # 4. Trial objectives over time (F1 and F2 per trial)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["F1", "F2"])
    for name in configs:
        trials = load_results(base_dir / "results" / name)
        if not trials: continue
        tt = target_trials(trials, name)
        idxs = list(range(len(tt)))
        c = colors.get(name, "black")
        fig.add_trace(go.Scatter(x=idxs, y=[t["F1"] for t in tt], mode="markers",
                                 name=name, marker=dict(color=c, size=4), legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=idxs, y=[t["F2"] for t in tt], mode="markers",
                                 name=name, marker=dict(color=c, size=4), legendgroup=name,
                                 showlegend=False), row=2, col=1)
    fig.update_layout(title="Objective Values per Trial", height=700, width=900)
    fig.update_xaxes(title_text="Trial index", row=2, col=1)
    fig.update_yaxes(title_text="F1", row=1, col=1)
    fig.update_yaxes(title_text="F2", row=2, col=1)
    fig.write_html(str(out / "objectives.html"))
    fig.write_image(str(out / "objectives.png"), scale=2)
    print(f"  Saved: {out / 'objectives.png'}")

    # 4. Specialization comparison: before/after
    spec_pairs = [
        ("robust", "robust_spec_n0", "Robust → spec noise=0"),
        ("robust", "robust_spec_n3", "Robust → spec noise=3"),
        ("robust", "mf_robust_spec_n0", "Robust → MF+Robust spec noise=0"),
        ("robust", "mf_robust_spec_n3", "Robust → MF+Robust spec noise=3"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[2] for p in spec_pairs])
    for i, (parent, child, title) in enumerate(spec_pairs):
        r, c = i // 2 + 1, i % 2 + 1
        parent_trials = load_results(base_dir / "results" / parent)
        child_trials = load_results(base_dir / "results" / child)
        if parent_trials:
            pts = np.array([[t["F1"], t["F2"]] for t in target_trials(parent_trials, parent)])
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers",
                                     name=f"{parent} (parent)", marker=dict(color="lightgray", size=4),
                                     legendgroup=parent, showlegend=(i==0)), row=r, col=c)
        if child_trials:
            # Show only new trials (beyond parent count)
            n_parent = len(parent_trials) if parent_trials else 0
            new = [t for t in child_trials if t["idx"] >= n_parent]
            old = [t for t in child_trials if t["idx"] < n_parent]
            if old:
                pts_old = np.array([[t["F1"], t["F2"]] for t in target_trials(old, child)])
                if len(pts_old):
                    fig.add_trace(go.Scatter(x=pts_old[:, 0], y=pts_old[:, 1], mode="markers",
                                             name="inherited", marker=dict(color="silver", size=3),
                                             showlegend=(i==0)), row=r, col=c)
            if new:
                pts_new = np.array([[t["F1"], t["F2"]] for t in target_trials(new, child)])
                if len(pts_new):
                    fig.add_trace(go.Scatter(x=pts_new[:, 0], y=pts_new[:, 1], mode="markers",
                                             name="specialized", marker=dict(color=colors.get(child, "red"), size=6),
                                             showlegend=(i==0)), row=r, col=c)
    fig.update_layout(title="Specialization: Parent vs Specialized Trials", height=800, width=1000)
    fig.update_xaxes(title_text="F1 (minimize)")
    fig.update_yaxes(title_text="F2 (minimize)")
    fig.write_html(str(out / "specialization.html"))
    fig.write_image(str(out / "specialization.png"), scale=2)
    print(f"  Saved: {out / 'specialization.png'}")

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="MF+Robust benchmark suite")
    parser.add_argument("--run", action="store_true", help="Run configs")
    parser.add_argument("--compare", action="store_true", help="Text comparison")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--configs", default=",".join(ALL_BASE),
                        help="Comma-separated config names (default: base configs)")
    parser.add_argument("--all", action="store_true", help="Run all configs including specializations")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    if args.all:
        configs = ALL_BASE + ALL_SPEC
    else:
        configs = [c.strip() for c in args.configs.split(",")]

    if not args.run and not args.compare and not args.plot:
        args.run = True
        args.compare = True
        args.plot = True

    if args.run:
        # Run base configs first, then specs (which depend on base results)
        base = [c for c in configs if c in ALL_BASE]
        spec = [c for c in configs if c in ALL_SPEC]
        if base: run_configs(base, base_dir)
        if spec: run_configs(spec, base_dir)
    if args.compare:
        compare(configs, base_dir)
    if args.plot:
        plot_all(configs, base_dir)

if __name__ == "__main__":
    main()
