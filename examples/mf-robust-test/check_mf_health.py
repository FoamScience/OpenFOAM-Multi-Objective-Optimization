#!/usr/bin/env python3
"""MF run health-check. Usage: uv run python3 check_mf_health.py [config...]

Verifies per-config:
  (1) cost metric landed in experiment.lookup_data (tracking wiring)
  (2) fidelity distribution across trials
  (3) total cost vs. sum of expected per-fidelity cost
  (4) force-schedule quota met for target fidelity (if applicable)
  (5) HV from fidelity-1 observations
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

# Register foambo types so loading client_state works.
import foambo.metrics  # noqa: F401
import foambo.robustness  # noqa: F401

from ax.api.client import Client

RESULTS = Path(__file__).parent / "results"
CONFIGS = ["mf", "mf_robust", "mf_robust_spec_n0", "mf_robust_spec_n3"]
TARGET_FID = 1
FID_COST = {0: 0.7, 1: 1.0}
QUOTA = 0.20   # match configs' multi_fidelity.force_schedule.quotas[1.0]


def check(name: str):
    out = {"config": name}
    art = RESULTS / name / "artifacts"
    csvs = sorted(art.glob("*_report.csv"))
    if not csvs:
        return {**out, "error": "no report.csv"}
    df = pd.read_csv(csvs[-1])

    # (1) cost metric presence in trial report
    out["cost_tracked"] = "cost" in set(df.metric_name)

    # (2) fidelity distribution (per-trial, deduped)
    one_row = df.drop_duplicates("trial_index")
    fid_counts = one_row.fidelity.value_counts().to_dict()
    out["fid_counts"] = {int(k): int(v) for k, v in fid_counts.items()}
    n_target = int(fid_counts.get(TARGET_FID, 0))
    n_total = int(len(one_row))

    # (3) observed cost total vs. expected
    cost_obs = df[df.metric_name == "cost"]["mean"].sum()
    cost_expect = sum(
        FID_COST[int(f)] * int(c) for f, c in fid_counts.items()
        if int(f) in FID_COST
    )
    out["cost_total_obs"] = round(float(cost_obs), 2)
    out["cost_total_expected"] = round(float(cost_expect), 2)
    out["cost_ok"] = abs(cost_obs - cost_expect) < 0.1

    # (4) quota check (against BO trials only — approximate by subtracting
    #     5 initial Sobol trials; conservative)
    n_bo = max(n_total - 5, 0)
    expected_target = math.ceil(QUOTA * n_bo)
    out["target_fid_count"] = n_target
    out["target_fid_expected"] = expected_target
    out["quota_met"] = n_target >= expected_target

    # (5) HV over target-fidelity Pareto (quick)
    target_df = one_row[one_row.fidelity == TARGET_FID]
    if len(target_df) and "F1" in df.metric_name.unique():
        # attach F1/F2 per trial
        f1 = df[df.metric_name == "F1"].set_index("trial_index")["mean"]
        f2 = df[df.metric_name == "F2"].set_index("trial_index")["mean"]
        pts = [(f1[i], f2[i]) for i in target_df.trial_index if i in f1 and i in f2]
        REF = (200.0, 70.0)
        pts.sort()
        hv, prev = 0.0, REF[1]
        for x, y in pts:
            if x >= REF[0] or y >= REF[1]:
                continue
            if y < prev:
                hv += (REF[0] - x) * (prev - y)
                prev = y
        out["hv_target_fid"] = round(hv, 1)
    else:
        out["hv_target_fid"] = None

    return out


def main(configs):
    rows = [check(c) for c in configs]
    # Pretty print
    cols = ["config", "n_total", "fid_counts", "cost_tracked", "cost_ok",
            "target_fid_count", "target_fid_expected", "quota_met",
            "hv_target_fid"]
    for r in rows:
        r["n_total"] = sum(r.get("fid_counts", {}).values())
    df = pd.DataFrame(rows)
    # Keep available cols
    df = df[[c for c in cols if c in df.columns]]
    print(df.to_string(index=False))

    # Red flags
    bad = []
    for r in rows:
        if not r.get("cost_tracked"):
            bad.append(f"{r['config']}: cost NOT tracked — _update_cost_state will see empty df")
        if not r.get("cost_ok", True):
            bad.append(f"{r['config']}: observed cost != expected "
                       f"({r.get('cost_total_obs')} vs {r.get('cost_total_expected')})")
        if not r.get("quota_met", True):
            bad.append(f"{r['config']}: force-schedule quota miss "
                       f"({r.get('target_fid_count')}/{r.get('target_fid_expected')})")
    if bad:
        print("\nRED FLAGS:")
        for b in bad:
            print(" -", b)
    else:
        print("\nAll checks green.")


if __name__ == "__main__":
    main(sys.argv[1:] or CONFIGS)
