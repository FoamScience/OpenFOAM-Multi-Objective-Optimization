#!/usr/bin/env python3

"""foamBO REST API server.

Runs in a background daemon thread alongside the optimizer.  All reads from
the Ax ``Client`` are protected by a ``threading.Lock`` so the optimizer
thread and the uvicorn worker never touch the experiment concurrently.

ETag-based caching avoids serialization when the
dashboard already has current data.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger(__name__)


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles NaN, Inf, sets, numpy types, and unknown objects."""
    def default(self, o):
        if isinstance(o, set):
            return list(o)
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return None if np.isnan(o) or np.isinf(o) else float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        # Last resort: stringify unknown objects
        try:
            return str(o)
        except Exception:
            return f"<unserializable: {type(o).__name__}>"


def _safe_json(data: Any) -> str:
    return json.dumps(data, cls=_SafeEncoder)


from starlette.responses import Response

class SafeJSONResponse(Response):
    media_type = "application/json"
    def __init__(self, content: Any = None, status_code: int = 200,
                 headers: dict | None = None, **kwargs):
        body = _safe_json(content).encode("utf-8")
        super().__init__(content=body, status_code=status_code,
                         headers=headers, media_type=self.media_type)

class ParameterInfo(BaseModel):
    name: str
    type: str
    bounds: Optional[list] = None
    values: Optional[list] = None
    parameter_type: Optional[str] = None
    fixed: bool = False

class ObjectiveInfo(BaseModel):
    name: str
    minimize: bool

class ExperimentResponse(BaseModel):
    name: str
    description: str
    parameters: List[ParameterInfo]
    objectives: List[ObjectiveInfo]
    outcome_constraints: List[str]
    objective_thresholds: List[str]
    max_trials: int
    parallelism: int
    poll_interval: float
    early_stopping: Optional[dict] = None
    dimensionality_reduction: Optional[dict] = None
    dependency_rules: List[dict]

class TrialDetail(BaseModel):
    index: int
    status: str
    parameters: dict
    gen_node: str
    case_path: Optional[str] = None
    dependencies: List[dict]
    metrics: dict

class TrialsResponse(BaseModel):
    trials: List[TrialDetail]
    counts: Dict[str, int]

class ObjectiveValues(BaseModel):
    minimize: bool
    values: List[dict]
    best: Optional[dict] = None
    best_so_far: List[float]

class ObjectivesResponse(BaseModel):
    objectives: Dict[str, ObjectiveValues]

class StreamingResponse_(BaseModel):
    metrics: dict
    thresholds: dict

class GenerationResponse(BaseModel):
    current_node: str
    node_counts: Dict[str, int]
    has_model: bool
    model_type: Optional[str] = None

class PredictRequest(BaseModel):
    parameters: dict

class PredictResponse(BaseModel):
    predictions: dict

class ParetoResponse(BaseModel):
    frontier: List[dict]
    hypervolume: Optional[float] = None
    reference_point: Optional[dict] = None
    model_predictions_used: bool

class StatusResponse(BaseModel):
    running: bool
    uptime_s: float
    last_callback_s_ago: float
    trials_completed: int
    trials_running: int
    trials_total: int
    model_fitted: bool

class ConfigFieldSchema(BaseModel):
    path: str
    type: str
    default: Any = None
    description: str = ""
    mutable: bool = False

class ConfigSchemaResponse(BaseModel):
    fields: List[ConfigFieldSchema]

class ConfigPatchRequest(BaseModel):
    model_config = {"extra": "allow"}

class ConfigPatchResponse(BaseModel):
    updated: List[str]
    rejected: List[dict]


# Mutable config fields (safe to change mid-run)
MUTABLE_PREFIXES = {
    "orchestration_settings.max_trials",
    "orchestration_settings.timeout_hours",
    "orchestration_settings.ttl_seconds_for_trials",
    "orchestration_settings.tolerated_trial_failure_rate",
    "orchestration_settings.dimensionality_reduction.",
    "orchestration_settings.early_stopping_strategy.",
}

def _is_mutable(path: str) -> bool:
    for prefix in MUTABLE_PREFIXES:
        if path == prefix or path.startswith(prefix):
            return True
    return False

# Server state — shared between endpoints and the optimizer callback
class _ApiState:
    """Holds references and cached snapshots for the API endpoints."""

    def __init__(self):
        self.client = None
        self.raw_cfg = None
        self.orch_cfg = None
        self.lock = threading.Lock()
        self.start_time: float = time.time()
        self.last_callback: float = time.time()
        self.callback_seq: int = 0

        # Per-endpoint version counters for ETag caching
        self._versions: Dict[str, int] = {
            "experiment": 0,
            "trials": 0,
            "objectives": 0,
            "streaming": 0,
            "generation": 0,
            "pareto": 0,
            "config": 0,
        }

    def bump(self, *endpoints: str):
        for ep in endpoints:
            self._versions[ep] = self._versions.get(ep, 0) + 1

    def etag(self, endpoint: str) -> str:
        return f'"{endpoint}-{self._versions.get(endpoint, 0)}"'

    def update(self, client):
        """Called from the optimizer callback to refresh state."""
        with self.lock:
            self.client = client
            self.last_callback = time.time()
            self.callback_seq += 1
            # Bump all poll-driven endpoints
            self.bump(
                "trials", "objectives", "streaming",
                "generation", "pareto", "config",
            )


_state = _ApiState()

# FastAPI application
app = FastAPI(
    title="foamBO API",
    version="1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    import traceback
    log.error(f"API {request.method} {request.url.path} failed: {exc}\n{traceback.format_exc()}")
    return SafeJSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/", response_class=Response)
def serve_dashboard():
    """Serve the web dashboard HTML."""
    import importlib.resources
    try:
        html = importlib.resources.files("foambo").joinpath("templates/dashboard.html").read_text()
    except Exception:
        import os
        path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
        with open(path) as f:
            html = f.read()
    return Response(content=html, media_type="text/html")


def _check_etag(endpoint: str, if_none_match: Optional[str]) -> Optional[Response]:
    """Return a 304 response if the ETag matches, else None."""
    current = _state.etag(endpoint)
    if if_none_match and if_none_match == current:
        return Response(status_code=304,
                        headers={"ETag": current,
                                 "X-Callback-Seq": str(_state.callback_seq)})
    return None


def _headers(endpoint: str) -> dict:
    return {
        "ETag": _state.etag(endpoint),
        "X-Callback-Seq": str(_state.callback_seq),
    }


# ---- helpers to extract data under lock ----

def _get_experiment_info() -> dict:
    """Extract experiment metadata under lock."""
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        exp = client._experiment
        opt_config = exp.optimization_config

        params = []
        for pname, p in list(exp.search_space.parameters.items()):
            from ax.core.parameter import FixedParameter, RangeParameter, ChoiceParameter
            info = {"name": pname, "type": "range", "fixed": isinstance(p, FixedParameter)}
            if isinstance(p, FixedParameter):
                info["type"] = "fixed"
                info["bounds"] = [p.value, p.value]
            elif isinstance(p, RangeParameter):
                info["type"] = "range"
                info["bounds"] = [p.lower, p.upper]
                info["parameter_type"] = p.parameter_type.name.lower()
            elif isinstance(p, ChoiceParameter):
                info["type"] = "choice"
                info["values"] = list(p.values)
            param_groups = getattr(exp.runner, '_parameter_groups', {})
            info["groups"] = param_groups.get(pname, [])
            params.append(info)

        param_groups = getattr(exp.runner, '_parameter_groups', {})
        objectives = []
        if hasattr(opt_config.objective, 'objectives'):
            for obj in opt_config.objective.objectives:
                for mn in obj.metric_names:
                    objectives.append({"name": mn, "minimize": obj.minimize})
        else:
            for mn in opt_config.objective.metric_names:
                objectives.append({"name": mn, "minimize": opt_config.objective.minimize})

        constraints = []
        if opt_config.outcome_constraints:
            for c in opt_config.outcome_constraints:
                constraints.append(str(c))

        thresholds = []
        if hasattr(opt_config, '_objective_thresholds') and opt_config._objective_thresholds:
            for ot in opt_config._objective_thresholds:
                op_str = ">=" if str(ot.op) == "ComparisonOp.GEQ" else "<="
                thresholds.append(f"{ot.metric.name} {op_str} {ot.bound}")

        orch = _state.orch_cfg
        es_dict = None
        if orch and hasattr(orch, 'early_stopping_strategy') and orch.early_stopping_strategy:
            es_dict = _serialize_early_stopping(orch.early_stopping_strategy)

        dr_dict = None
        if orch and hasattr(orch, 'dimensionality_reduction'):
            dr = orch.dimensionality_reduction
            dr_dict = {
                "enabled": dr.enabled,
                "after_trials": dr.after_trials,
                "min_importance": dr.min_importance,
                "fix_at": dr.fix_at,
                "max_fix_fraction": dr.max_fix_fraction,
            }

        dep_rules = []
        runner = exp.runner
        if hasattr(runner, 'trial_dependencies') and runner.trial_dependencies:
            for dep in runner.trial_dependencies:
                dep_rules.append({
                    "name": dep.name,
                    "source": dep.source,
                    "phase": dep.phase if hasattr(dep, 'phase') else None,
                })

        # Extract ES thresholds in flat format for the dashboard
        es_thresholds = {}
        if es_dict:
            from .analysis import _extract_es_thresholds
            _extract_es_thresholds(es_dict, es_thresholds)

        return {
            "name": exp.name,
            "description": exp.description or "",
            "parameters": params,
            "objectives": objectives,
            "outcome_constraints": constraints,
            "objective_thresholds": thresholds,
            "max_trials": orch.max_trials if orch else 0,
            "parallelism": orch.parallelism if orch else 1,
            "poll_interval": orch.initial_seconds_between_polls if orch else 1,
            "early_stopping": es_dict,
            "es_thresholds": es_thresholds,
            "dimensionality_reduction": dr_dict,
            "dependency_rules": dep_rules,
            "parameter_groups": param_groups,
        }


def _serialize_early_stopping(strategy) -> dict | None:
    """Recursively serialize an early stopping strategy to a dict."""
    if strategy is None:
        return None
    from ax.early_stopping.strategies import (
        AndEarlyStoppingStrategy, OrEarlyStoppingStrategy, ThresholdEarlyStoppingStrategy,
    )
    from ax.api.client import PercentileEarlyStoppingStrategy
    if isinstance(strategy, OrEarlyStoppingStrategy):
        return {
            "type": "or",
            "left": _serialize_early_stopping(strategy.left),
            "right": _serialize_early_stopping(strategy.right),
        }
    if isinstance(strategy, AndEarlyStoppingStrategy):
        return {
            "type": "and",
            "left": _serialize_early_stopping(strategy.left),
            "right": _serialize_early_stopping(strategy.right),
        }
    if isinstance(strategy, ThresholdEarlyStoppingStrategy):
        return {
            "type": "threshold",
            "metric_signatures": list(strategy.metric_signatures) if strategy.metric_signatures else [],
            "metric_threshold": strategy.metric_threshold if hasattr(strategy, 'metric_threshold') else None,
            "min_progression": strategy.min_progression if hasattr(strategy, 'min_progression') else None,
        }
    if isinstance(strategy, PercentileEarlyStoppingStrategy):
        return {
            "type": "percentile",
            "metric_signatures": list(strategy.metric_signatures) if strategy.metric_signatures else [],
            "percentile_threshold": strategy.percentile_threshold if hasattr(strategy, 'percentile_threshold') else None,
            "min_progression": strategy.min_progression if hasattr(strategy, 'min_progression') else None,
        }
    return {"type": type(strategy).__name__}


def _get_trials() -> dict:
    """Extract all trial data under lock."""
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        exp = client._experiment
        from ax.core.base_trial import TrialStatus as AxTrialStatus

        trials = []
        counts: Dict[str, int] = {}
        for tidx, trial in list(exp.trials.items()):
            status = trial.status.name
            counts[status] = counts.get(status, 0) + 1
            params = trial.arm.parameters if trial.arm else {}
            gen_node = ""
            gr = getattr(trial, "generator_run", None) or (
                trial.generator_runs[0] if hasattr(trial, "generator_runs")
                and trial.generator_runs else None)
            if gr:
                gen_node = getattr(gr, "_generation_node_name", "") or ""
            case_path = (trial.run_metadata.get("case_path")
                         or trial.run_metadata.get("job", {}).get("case_path"))

            deps = []
            runner = exp.runner
            if hasattr(runner, 'trial_registry') and tidx in runner.trial_registry:
                reg = runner.trial_registry[tidx]
                if "dependencies" in reg:
                    deps = reg["dependencies"]

            metrics = {}
            if trial.status in (AxTrialStatus.COMPLETED, AxTrialStatus.EARLY_STOPPED):
                try:
                    data = exp.lookup_data(trial_indices=[tidx])
                    df = data.df if hasattr(data, 'df') else data
                    if not df.empty:
                        # Keep only non-streaming rows (step is NaN or missing)
                        import math
                        for _, row in df.iterrows():
                            step = row.get("step", None)
                            if step is None or (isinstance(step, float) and math.isnan(step)):
                                metrics[row["metric_name"]] = row["mean"]
                        # If no non-streaming rows, take the last value per metric
                        if not metrics and "step" in df.columns:
                            for mname in df["metric_name"].unique():
                                mdf = df[df["metric_name"] == mname].sort_values("step" if "step" in df.columns else "trial_index")
                                if not mdf.empty:
                                    metrics[mname] = float(mdf.iloc[-1]["mean"])
                except Exception:
                    pass

            trials.append({
                "index": tidx,
                "status": status,
                "parameters": params,
                "gen_node": gen_node,
                "case_path": case_path,
                "dependencies": deps,
                "metrics": metrics,
            })
        return {"trials": trials, "counts": counts}


def _get_single_trial(index: int) -> dict:
    """Extract a single trial's data under lock."""
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        exp = client._experiment
        if index not in exp.trials:
            raise HTTPException(404, f"Trial {index} not found")

        trial = exp.trials[index]
        from ax.core.base_trial import TrialStatus as AxTrialStatus

        params = trial.arm.parameters if trial.arm else {}
        gen_node = ""
        if trial.generator_run and trial.generator_run.generator_run_type:
            gen_node = trial.generator_run.generator_run_type
        case_path = (trial.run_metadata.get("case_path")
                     or trial.run_metadata.get("job", {}).get("case_path"))

        deps = []
        runner = exp.runner
        if hasattr(runner, 'trial_registry') and index in runner.trial_registry:
            reg = runner.trial_registry[index]
            if "dependencies" in reg:
                deps = reg["dependencies"]

        metrics = {}
        if trial.status == AxTrialStatus.COMPLETED:
            try:
                data = exp.lookup_data(trial_indices=[index])
                df = data.df if hasattr(data, 'df') else data
                if not df.empty:
                    for _, row in df.iterrows():
                        if row.get("step") is None or (hasattr(row["step"], '__float__') and row["step"] != row["step"]):
                            metrics[row["metric_name"]] = row["mean"]
            except Exception:
                pass

        run_metadata = dict(trial.run_metadata) if trial.run_metadata else {}
        # Try reading log tail — find most recently modified log.* or *.log file
        log_tail = ""
        log_file_name = ""
        if case_path:
            import os, glob as _glob
            candidates = _glob.glob(os.path.join(case_path, "log.*")) + \
                         _glob.glob(os.path.join(case_path, "*.log"))
            if candidates:
                newest = max(candidates, key=os.path.getmtime)
                log_file_name = os.path.basename(newest)
                try:
                    with open(newest) as f:
                        lines = f.readlines()
                    log_tail = "".join(lines[-50:])
                except Exception:
                    pass

        return {
            "index": index,
            "status": trial.status.name,
            "parameters": params,
            "gen_node": gen_node,
            "case_path": case_path,
            "dependencies": deps,
            "metrics": metrics,
            "run_metadata": {k: v for k, v in run_metadata.items()
                             if isinstance(v, (str, int, float, bool, type(None)))},
            "log_tail": log_tail,
            "log_file": log_file_name,
        }


def _get_objectives() -> dict:
    """Extract per-objective values, best, and best-so-far under lock."""
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        exp = client._experiment
        opt_config = exp.optimization_config
        data = exp.lookup_data()
        df = data.df if hasattr(data, 'df') else data
        # Filter out streaming rows (keep only final metric values where step is NaN/missing)
        if not df.empty and "step" in df.columns:
            import pandas as pd
            filtered = df[pd.isna(df["step"])]
            if not filtered.empty:
                df = filtered
            # else: no non-streaming rows, keep all (take last per trial+metric below)

        objectives_out = {}
        obj_list = []
        if hasattr(opt_config.objective, 'objectives'):
            for obj in opt_config.objective.objectives:
                for mn in obj.metric_names:
                    obj_list.append((mn, obj.minimize))
        else:
            for mn in opt_config.objective.metric_names:
                obj_list.append((mn, opt_config.objective.minimize))

        for metric_name, minimize in obj_list:
            mdf = df[df["metric_name"] == metric_name].sort_values("trial_index")
            # Deduplicate: keep last row per trial (handles streaming fallback)
            mdf = mdf.drop_duplicates(subset=["trial_index"], keep="last")
            values = [{"trial": int(r["trial_index"]), "value": float(r["mean"])}
                      for _, r in mdf.iterrows()]
            best = None
            best_so_far = []
            running_best = None
            for v in values:
                val = v["value"]
                if running_best is None:
                    running_best = val
                elif (minimize and val < running_best) or (not minimize and val > running_best):
                    running_best = val
                best_so_far.append(running_best)
                if best is None:
                    best = v
                elif (minimize and val < best["value"]) or (not minimize and val > best["value"]):
                    best = v

            objectives_out[metric_name] = {
                "minimize": minimize,
                "values": values,
                "best": best,
                "best_so_far": best_so_far,
            }

        return {"objectives": objectives_out}


def _get_streaming() -> dict:
    """Extract streaming data under lock."""
    with _state.lock:
        client = _state.client
        raw_cfg = _state.raw_cfg
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        from .analysis import compute_streaming_data
        data = compute_streaming_data(client, raw_cfg)
        # Serialize trial indices as strings for JSON keys
        metrics_out = {}
        for metric_name, per_trial in data.get("metrics", {}).items():
            metrics_out[metric_name] = {
                str(tidx): td for tidx, td in per_trial.items()
            }
        return {
            "metrics": metrics_out,
            "thresholds": data.get("thresholds", {}),
        }


def _get_generation() -> dict:
    """Extract generation strategy state under lock."""
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        exp = client._experiment
        try:
            gs = client._generation_strategy
        except Exception:
            gs = None

        if gs is None:
            return {
                "current_node": "",
                "node_counts": {},
                "node_targets": {},
                "has_model": False,
                "model_type": None,
            }

        current_node = gs._curr.name if gs._curr else ""

        # Build per-node counts from each trial's generator_run attribution
        node_counts: Dict[str, int] = {}
        node_targets: Dict[str, int | None] = {}

        # Initialize all strategy nodes with 0 counts and extract targets
        for node in gs._nodes:
            node_counts[node.name] = 0
            target = None
            for tc in node.transition_criteria:
                if hasattr(tc, "threshold") and hasattr(tc, "block_transition_if_unmet"):
                    target = int(tc.threshold)
                    break
            node_targets[node.name] = target

        # Count by inspecting each trial's generator_run node name
        for trial in list(exp.trials.values()):
            node_name = None
            gr = getattr(trial, "generator_run", None) or (
                trial.generator_runs[0] if hasattr(trial, "generator_runs")
                and trial.generator_runs else None)
            if gr:
                node_name = getattr(gr, "_generation_node_name", None)
            if node_name and node_name in node_counts:
                node_counts[node_name] += 1
            elif node_name == "baseline" or (
                exp.status_quo is not None
                and hasattr(trial, "arm") and trial.arm == exp.status_quo):
                node_counts.setdefault("baseline", 0)
                node_counts["baseline"] += 1
                node_targets.setdefault("baseline", 1)
            elif node_name:
                # Node from a different strategy (e.g. baseline ManualGenerationNode)
                node_counts.setdefault(node_name, 0)
                node_counts[node_name] += 1
                node_targets.setdefault(node_name, 1)

        has_model = False
        model_type = None
        try:
            adapter = gs.adapter
            if adapter is not None:
                # Check that the surrogate model has actually been fitted on data
                surr = getattr(adapter.generator, "surrogate", None)
                if surr is not None:
                    model = getattr(surr, "model", None)
                    has_model = model is not None
                else:
                    has_model = False
                if has_model:
                    model_type = type(adapter.generator).__name__
        except Exception:
            pass

        return {
            "current_node": current_node,
            "node_counts": node_counts,
            "node_targets": node_targets,
            "has_model": has_model,
            "model_type": model_type,
        }


def _get_pareto(use_model: bool) -> dict:
    """Compute Pareto frontier under lock."""
    import warnings
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")

        # Need at least 2 completed trials for Pareto
        from ax.core.base_trial import TrialStatus as _TS
        n_completed = sum(1 for t in list(client._experiment.trials.values()) if t.status == _TS.COMPLETED)
        if n_completed < 2:
            return {"frontier": [], "hypervolume": None, "hypervolume_trace": [],
                    "reference_point": None, "model_predictions_used": False}

        from ax.api.client import MultiObjective
        opt_config = client._experiment.optimization_config
        if not isinstance(opt_config.objective, MultiObjective):
            raise HTTPException(400, "Pareto frontier requires multi-objective optimization")

        model_used = False
        front = None
        # Suppress Ax winsorize warnings about missing objective thresholds
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*objective thresholds.*")
            if use_model:
                try:
                    front = client.get_pareto_frontier(use_model_predictions=True)
                    model_used = True
                except Exception:
                    pass
            if front is None:
                try:
                    front = client.get_pareto_frontier(use_model_predictions=False)
                except Exception as e:
                    raise HTTPException(503, f"Cannot compute Pareto frontier: {e}")

        frontier = []
        for params, predictions, trial_idx, arm_name in front:
            metrics = {k: v[0] if isinstance(v, (list, tuple)) else v
                       for k, v in predictions.items()}
            frontier.append({"parameters": params, "metrics": metrics})

        # Reference point from thresholds
        ref_point = {}
        if hasattr(opt_config, '_objective_thresholds') and opt_config._objective_thresholds:
            for ot in opt_config._objective_thresholds:
                ref_point[ot.metric.name] = ot.bound

        # Compute hypervolume trace over completed trials
        hv_trace = []
        try:
            from ax.plot.pareto_utils import compute_hypervolume
            exp = client._experiment
            completed = sorted(
                [t for t in list(exp.trials.values())
                 if t.status.name == "COMPLETED"],
                key=lambda t: t.index,
            )
            if completed and ref_point:
                obj_names = list(opt_config.objective.metric_names)
                minimize_flags = {
                    obj.metric.name: obj.minimize
                    for obj in opt_config.objective.objectives
                }
                data = exp.lookup_data().df
                for i in range(1, len(completed) + 1):
                    subset_indices = {t.index for t in completed[:i]}
                    subset_data = data[data["trial_index"].isin(subset_indices)]
                    try:
                        points = []
                        for tidx in subset_indices:
                            tdf = subset_data[subset_data["trial_index"] == tidx]
                            point = []
                            for obj_name in obj_names:
                                mdf = tdf[tdf["metric_name"] == obj_name]
                                if mdf.empty:
                                    break
                                val = float(mdf["mean"].iloc[-1])
                                # Negate maximized objectives for HV computation
                                if not minimize_flags.get(obj_name, True):
                                    val = -val
                                point.append(val)
                            if len(point) == len(obj_names):
                                points.append(point)
                        if points:
                            import numpy as np
                            ref = [
                                -ref_point[n] if not minimize_flags.get(n, True) else ref_point[n]
                                for n in obj_names
                            ]
                            hv = compute_hypervolume(
                                np.array(points),
                                np.array(ref),
                            )
                            hv_trace.append({"trial": completed[i-1].index, "value": float(hv)})
                    except Exception:
                        pass
        except Exception:
            pass

        return {
            "frontier": frontier,
            "hypervolume": hv_trace[-1]["value"] if hv_trace else None,
            "hypervolume_trace": hv_trace,
            "reference_point": ref_point or None,
            "model_predictions_used": model_used,
        }



def _do_predict(parameters: dict) -> dict:
    """Run model prediction under lock."""
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        gs = None
        try:
            gs = client._generation_strategy
        except Exception:
            pass
        if gs is None or gs.adapter is None:
            raise HTTPException(503, "Model not fitted yet")

        try:
            prediction = gs.adapter.predict([parameters])
            # prediction is (means_dict, covariances_dict)
            means = prediction[0]
            sems = prediction[1] if len(prediction) > 1 else {}
            result = {}
            for metric_name in means:
                result[metric_name] = {
                    "mean": float(means[metric_name][0]) if isinstance(means[metric_name], list) else float(means[metric_name]),
                    "sem": float(sems.get(metric_name, [0])[0]) if isinstance(sems.get(metric_name, [0]), list) else float(sems.get(metric_name, 0)),
                }
            return {"predictions": result}
        except Exception as e:
            raise HTTPException(503, f"Prediction failed: {e}")


def _get_config_flat() -> dict:
    """Return the active config as a flat dict."""
    with _state.lock:
        raw_cfg = _state.raw_cfg
        if raw_cfg is None:
            raise HTTPException(503, "Optimizer not initialized")
        from omegaconf import OmegaConf
        container = OmegaConf.to_container(raw_cfg, resolve=True)

    def _flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
        return out

    return _flatten(container)


def _get_config_schema() -> dict:
    """Auto-generate config schema from Pydantic models."""
    from .orchestrate import FoamBOConfig
    fields = []
    _walk_model(FoamBOConfig, "", fields)
    return {"fields": fields}


def _walk_model(model_cls, prefix: str, fields: list):
    """Recursively walk Pydantic model fields to build schema entries."""
    for name, field_info in model_cls.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        annotation = field_info.annotation

        # Check if annotation is itself a Pydantic BaseModel subclass
        is_model = False
        try:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                is_model = True
        except TypeError:
            pass

        if is_model:
            _walk_model(annotation, path, fields)
        else:
            type_name = _type_name(annotation)
            fields.append({
                "path": path,
                "type": type_name,
                "default": _safe_default(field_info.default),
                "description": field_info.description or "",
                "mutable": _is_mutable(path),
            })


def _type_name(annotation) -> str:
    """Convert a type annotation to a simple string."""
    if annotation is None:
        return "any"
    origin = getattr(annotation, '__origin__', None)
    if origin is not None:
        import typing
        if origin is list or origin is typing.List:
            return "list"
        if origin is dict or origin is typing.Dict:
            return "dict"
        if origin is typing.Union:
            args = [a for a in annotation.__args__ if a is not type(None)]
            if len(args) == 1:
                return _type_name(args[0])
            return "any"
        return "any"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "bool"
    if annotation is str:
        return "str"
    name = getattr(annotation, '__name__', None)
    if name:
        return name.lower()
    return "any"


def _safe_default(value) -> Any:
    """Return a JSON-serializable default or None."""
    if value is None:
        return None
    if callable(value) and not isinstance(value, (int, float, str, bool)):
        # factory default
        return None
    try:
        import json
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _coerce_value(raw_cfg, path: str, value):
    """Coerce a value to match the OmegaConf node type at path."""
    from omegaconf import OmegaConf
    try:
        current = OmegaConf.select(raw_cfg, path)
        if current is not None:
            if isinstance(current, bool):
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            if isinstance(current, int):
                return int(value)
            if isinstance(current, float):
                return float(value)
    except Exception:
        pass
    return value


def _patch_config(updates: dict) -> dict:
    """Apply mutable config changes. Returns {updated, rejected}."""
    updated = []
    rejected = []
    with _state.lock:
        raw_cfg = _state.raw_cfg
        orch_cfg = _state.orch_cfg
        if raw_cfg is None or orch_cfg is None:
            raise HTTPException(503, "Optimizer not initialized")

        from omegaconf import OmegaConf

        for path, value in updates.items():
            if not _is_mutable(path):
                rejected.append({"path": path, "reason": "immutable"})
                continue
            # Apply to OmegaConf raw_cfg
            try:
                coerced = _coerce_value(raw_cfg, path, value)
                OmegaConf.update(raw_cfg, path, coerced)
                # Also update the typed orch_cfg if the path starts with orchestration_settings
                if path.startswith("orchestration_settings."):
                    sub_path = path[len("orchestration_settings."):]
                    parts = sub_path.split(".")
                    obj = orch_cfg
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], coerced)
                updated.append(path)
            except Exception as e:
                rejected.append({"path": path, "reason": str(e)})

    if updated:
        _state.bump("config", "experiment")

    return {"updated": updated, "rejected": rejected}


# Endpoints

@app.get("/api/v1/experiment")
def get_experiment(if_none_match: Optional[str] = Header(None)):
    cached = _check_etag("experiment", if_none_match)
    if cached:
        return cached
    try:
        data = _get_experiment_info()
        return SafeJSONResponse(content=data, headers=_headers("experiment"))
    except Exception as e:
        log.error(f"GET /experiment failed: {e}", exc_info=True)
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/v1/trials")
def get_trials(if_none_match: Optional[str] = Header(None)):
    cached = _check_etag("trials", if_none_match)
    if cached:
        return cached
    data = _get_trials()
    return SafeJSONResponse(content=data, headers=_headers("trials"))


@app.get("/api/v1/trials/{index}")
def get_trial(index: int):
    data = _get_single_trial(index)
    return SafeJSONResponse(content=data, headers=_headers("trials"))


@app.get("/api/v1/objectives")
def get_objectives(if_none_match: Optional[str] = Header(None)):
    cached = _check_etag("objectives", if_none_match)
    if cached:
        return cached
    data = _get_objectives()
    return SafeJSONResponse(content=data, headers=_headers("objectives"))


@app.get("/api/v1/streaming")
def get_streaming(if_none_match: Optional[str] = Header(None)):
    cached = _check_etag("streaming", if_none_match)
    if cached:
        return cached
    data = _get_streaming()
    return SafeJSONResponse(content=data, headers=_headers("streaming"))


@app.get("/api/v1/generation")
def get_generation(if_none_match: Optional[str] = Header(None)):
    cached = _check_etag("generation", if_none_match)
    if cached:
        return cached
    data = _get_generation()
    return SafeJSONResponse(content=data, headers=_headers("generation"))


@app.get("/api/v1/pareto/pick")
def pick_pareto_point(objective: str = Query(...)):
    """Pick the most interesting Pareto point for a given objective."""
    import warnings
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        from ax.api.client import MultiObjective
        opt_config = client._experiment.optimization_config
        if not isinstance(opt_config.objective, MultiObjective):
            raise HTTPException(400, "Requires multi-objective optimization")
        # Find minimize flag for this objective
        minimize = True
        for obj in opt_config.objective.objectives:
            if obj.metric.name == objective:
                minimize = obj.minimize
                break
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*objective thresholds.*")
            try:
                front = client.get_pareto_frontier(use_model_predictions=True)
            except Exception:
                try:
                    front = client.get_pareto_frontier(use_model_predictions=False)
                except Exception as e:
                    raise HTTPException(503, f"Cannot compute Pareto frontier: {e}")
        # Pick interesting point: best value, then lowest SEM among similar values
        best = None
        best_val = float("inf") if minimize else -float("inf")
        best_sem = float("inf")
        best_preds = {}
        for params, preds, _, _ in front:
            if objective not in preds:
                continue
            mean, sem = preds[objective][0], preds[objective][1]
            is_better = (minimize and mean < best_val) or (not minimize and mean > best_val)
            tolerance = abs(best_val * 0.01) if best_val != 0 else 0.01
            is_similar = abs(mean - best_val) <= tolerance
            if is_better or (is_similar and sem < best_sem):
                best_val = mean
                best_sem = sem
                best = params
                best_preds = {k: {"mean": v[0], "sem": v[1]} for k, v in preds.items()}
        if best is None and front:
            best = front[0][0]
            best_preds = {k: {"mean": v[0], "sem": v[1]} for k, v in front[0][1].items()}
    return SafeJSONResponse(content={
        "parameters": best or {},
        "predictions": best_preds,
        "objective": objective,
        "minimize": minimize,
    })


@app.post("/api/v1/predict")
def predict(req: PredictRequest):
    data = _do_predict(req.parameters)
    return SafeJSONResponse(content=data)


@app.get("/api/v1/pareto")
def get_pareto(use_model: bool = Query(True),
               if_none_match: Optional[str] = Header(None)):
    cached = _check_etag("pareto", if_none_match)
    if cached:
        return cached
    data = _get_pareto(use_model)
    return SafeJSONResponse(content=data, headers=_headers("pareto"))


@app.get("/api/v1/status")
def get_status():
    with _state.lock:
        client = _state.client
    if client is None:
        return SafeJSONResponse(content={
            "running": False, "uptime_s": 0, "last_callback_s_ago": 0,
            "trials_completed": 0, "trials_running": 0, "trials_total": 0,
            "model_fitted": False,
        })

    with _state.lock:
        exp = client._experiment
        from ax.core.base_trial import TrialStatus as AxTrialStatus
        completed = sum(1 for t in exp.trials.values() if t.status == AxTrialStatus.COMPLETED)
        running = sum(1 for t in exp.trials.values() if t.status == AxTrialStatus.RUNNING)
        total = len(exp.trials)
        has_model = False
        try:
            gs = client._generation_strategy
            has_model = gs is not None and gs.adapter is not None
        except Exception:
            pass

    return SafeJSONResponse(content={
        "running": True,
        "uptime_s": round(time.time() - _state.start_time, 1),
        "last_callback_s_ago": round(time.time() - _state.last_callback, 1),
        "trials_completed": completed,
        "trials_running": running,
        "trials_total": total,
        "model_fitted": has_model,
    })


@app.get("/api/v1/config")
def get_config(if_none_match: Optional[str] = Header(None)):
    cached = _check_etag("config", if_none_match)
    if cached:
        return cached
    data = _get_config_flat()
    return SafeJSONResponse(content=data, headers=_headers("config"))


@app.get("/api/v1/config/schema")
def get_config_schema():
    data = _get_config_schema()
    return SafeJSONResponse(content=data)


@app.patch("/api/v1/config")
def patch_config(request: Request):
    import asyncio
    # Read body synchronously-safe
    loop = asyncio.new_event_loop()
    body = loop.run_until_complete(request.json())
    loop.close()
    if not isinstance(body, dict):
        raise HTTPException(400, "Request body must be a JSON object")
    result = _patch_config(body)
    if result["rejected"] and not result["updated"]:
        return SafeJSONResponse(content=result, status_code=400)
    return SafeJSONResponse(content=result)



# Analysis endpoints (on-demand, POST)

def _ensure_model_fitted():
    """Refit the model if not already fitted. Called before analysis endpoints."""
    client = _state.client
    if client is None:
        raise HTTPException(503, "Optimizer not initialized")
    gs = None
    try:
        gs = client._generation_strategy
    except Exception:
        pass
    if gs is not None and gs.adapter is not None:
        return  # already fitted
    # Refit by generating (and abandoning) a trial
    log.info("Refitting model for analysis...")
    try:
        result = client.get_next_trials(max_trials=1)
        trials = result if isinstance(result, dict) else result[0]
        from ax.core.base_trial import TrialStatus
        for tidx in trials:
            client._experiment.trials[tidx]._status = TrialStatus.ABANDONED
        log.info("Model refitted successfully")
    except Exception as e:
        raise HTTPException(503, f"Model fitting failed: {e}")


def _compute_ax_analysis(analysis_cls, **kwargs) -> dict:
    """Run an Ax analysis under lock and return Plotly figure JSON."""
    import json as _json
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        _ensure_model_fitted()

        import logging
        ax_logger = logging.getLogger("ax")
        prev_level = ax_logger.level
        ax_logger.setLevel(logging.CRITICAL)
        try:
            cards = client.compute_analyses(
                analyses=[analysis_cls(**kwargs)],
                display=False,
            )
        finally:
            ax_logger.setLevel(prev_level)

        from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
        from ax.core.analysis_card import AnalysisCardGroup
        results = []
        for card in cards:
            if isinstance(card, AnalysisCardGroup):
                for sub in card.cards:
                    if isinstance(sub, PlotlyAnalysisCard):
                        results.append({
                            "figure": _json.loads(sub.blob),
                            "title": sub.title,
                            "subtitle": sub.subtitle,
                        })
            elif isinstance(card, PlotlyAnalysisCard):
                results.append({
                    "figure": _json.loads(card.blob),
                    "title": card.title,
                    "subtitle": card.subtitle,
                })
        if results:
            return results[0] if len(results) == 1 else {"cards": results}
        raise HTTPException(500, "Analysis produced no plotly card")


def _compute_ax_healthcheck(analysis_cls, **kwargs) -> dict:
    """Run an Ax healthcheck analysis and return markdown cards."""
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")

        import logging
        ax_logger = logging.getLogger("ax")
        prev_level = ax_logger.level
        ax_logger.setLevel(logging.CRITICAL)
        try:
            cards = client.compute_analyses(
                analyses=[analysis_cls(**kwargs)],
                display=False,
            )
        finally:
            ax_logger.setLevel(prev_level)

        from ax.core.analysis_card import AnalysisCardGroup
        results = []
        for card in cards:
            if isinstance(card, AnalysisCardGroup):
                for sub in card.cards:
                    results.append({"title": sub.title, "subtitle": sub.subtitle,
                                    "blob": sub.blob, "level": getattr(sub, "level", "INFO")})
            else:
                results.append({"title": card.title, "subtitle": card.subtitle,
                                "blob": card.blob, "level": getattr(card, "level", "INFO")})
        return {"cards": results}


@app.post("/api/v1/analysis/sensitivity")
def post_analysis_sensitivity(request_body: dict):
    """Sobol sensitivity analysis via Ax."""
    metric = request_body.get("metric")
    top_k = request_body.get("top_k", 10)
    try:
        from ax.analysis.plotly.sensitivity import SensitivityAnalysisPlot
        return SafeJSONResponse(content=_compute_ax_analysis(
            SensitivityAnalysisPlot, metric_name=metric, top_k=top_k))
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/v1/analysis/parallel-coordinates")
def post_analysis_parallel_coords(request_body: dict):
    """Parallel coordinates plot via Ax."""
    metric = request_body.get("metric")
    try:
        from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
        return SafeJSONResponse(content=_compute_ax_analysis(
            ParallelCoordinatesPlot, metric_name=metric))
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/v1/analysis/cross-validation")
def post_analysis_cross_validation(request_body: dict = {}):
    """Cross-validation observed vs predicted plot via Ax."""
    try:
        from ax.analysis.plotly.cross_validation import CrossValidationPlot
        return SafeJSONResponse(content=_compute_ax_analysis(CrossValidationPlot))
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/v1/analysis/contour")
def post_analysis_contour(request_body: dict):
    """Contour / response surface plot via Ax."""
    metric = request_body.get("metric")
    try:
        from ax.analysis.plotly.surface.contour import ContourPlot
        with _state.lock:
            client = _state.client
            if client is None:
                raise HTTPException(503, "Optimizer not initialized")
            params = [p for p in client._experiment.search_space.parameters
                      if hasattr(client._experiment.search_space.parameters[p], 'lower')]
        if len(params) < 2:
            return SafeJSONResponse(content={"error": "Need at least 2 range parameters for contour"}, status_code=400)
        # Use TopSurfacesAnalysis which auto-picks the best parameter pairs
        from ax.analysis.plotly.top_surfaces import TopSurfacesAnalysis
        return SafeJSONResponse(content=_compute_ax_analysis(
            TopSurfacesAnalysis, metric_name=metric, top_k=3))
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/v1/analysis/search-space")
def post_analysis_search_space(request_body: dict = {}):
    """Search space coverage analysis."""
    try:
        with _state.lock:
            client = _state.client
            if client is None:
                raise HTTPException(503, "Optimizer not initialized")
            completed = [idx for idx, t in client._experiment.trials.items()
                         if t.status.name == "COMPLETED"]
            if not completed:
                raise HTTPException(503, "No completed trials")
            trial_index = max(completed)
        from ax.analysis.healthcheck.search_space_analysis import SearchSpaceAnalysis
        return SafeJSONResponse(content=_compute_ax_healthcheck(
            SearchSpaceAnalysis, trial_index=trial_index))
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/v1/analysis/insights")
def post_analysis_insights(request_body: dict = {}):
    """Ax InsightsAnalysis diagnostic cards."""
    try:
        from ax.analysis.insights import InsightsAnalysis
        return SafeJSONResponse(content=_compute_ax_healthcheck(InsightsAnalysis))
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/v1/analysis/healthchecks")
def post_analysis_healthchecks(request_body: dict = {}):
    """TestOfNoEffect + RegressionAnalysis diagnostics."""
    all_cards = []
    for cls_path in [
        ("ax.analysis.healthcheck.no_effects_analysis", "TestOfNoEffectAnalysis"),
        ("ax.analysis.healthcheck.regression_analysis", "RegressionAnalysis"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(cls_path[0])
            cls = getattr(mod, cls_path[1])
            result = _compute_ax_healthcheck(cls)
            all_cards.extend(result.get("cards", []))
        except Exception as e:
            all_cards.append({"title": cls_path[1], "subtitle": "Error",
                              "blob": str(e), "level": "ERROR"})
    return SafeJSONResponse(content={"cards": all_cards})


@app.post("/api/v1/analysis/group-sensitivity")
def post_group_sensitivity(request_body: dict):
    """Group-level sensitivity: sum first-order Sobol indices by parameter group."""
    metric = request_body.get("metric")
    try:
        with _state.lock:
            client = _state.client
            if client is None:
                raise HTTPException(503, "Not initialized")
            _ensure_model_fitted()
            gs = client._generation_strategy
            exp = client._experiment

            import torch
            from ax.utils.sensitivity.sobol_measures import SobolSensitivityGPMean
            surr = gs.adapter.generator.surrogate
            model = surr.model
            param_names = [p for p in exp.search_space.parameters
                           if hasattr(exp.search_space.parameters[p], 'lower')]
            n_params = len(param_names)
            bounds = torch.zeros(2, n_params, dtype=torch.float64)
            bounds[1] = 1.0
            sobol = SobolSensitivityGPMean(model=model, bounds=bounds, num_mc_samples=1000)
            first_order = sobol.first_order_indices().detach().cpu().numpy()
            if first_order.ndim > 1:
                first_order = first_order.mean(axis=0)

            param_groups = getattr(exp.runner, '_parameter_groups', {})
            group_scores = {}
            for i, pname in enumerate(param_names):
                groups = param_groups.get(pname, ["ungrouped"])
                for g in groups:
                    group_scores[g] = group_scores.get(g, 0.0) + float(first_order[i])

        return SafeJSONResponse(content={"groups": group_scores, "metric": metric})
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/v1/analysis/group-conditional-best")
def post_group_conditional_best(request_body: dict):
    """Find the best trial among those matching fixed group parameter values."""
    group = request_body.get("group")
    values = request_body.get("values", {})
    try:
        with _state.lock:
            client = _state.client
            if client is None:
                raise HTTPException(503, "Not initialized")
            exp = client._experiment
            param_groups = getattr(exp.runner, '_parameter_groups', {})
            group_params = [n for n, gs in param_groups.items() if group in gs]

            from ax.core.base_trial import TrialStatus
            matches = []
            for idx, trial in exp.trials.items():
                if trial.status != TrialStatus.COMPLETED:
                    continue
                params = trial.arm.parameters
                if all(params.get(p) == values.get(p) for p in group_params if p in values):
                    matches.append((idx, params))

            if not matches:
                return SafeJSONResponse(content={"match_count": 0, "error": "No matching trials"})

            opt_config = exp.optimization_config
            obj_list = []
            if hasattr(opt_config.objective, 'objectives'):
                for obj in opt_config.objective.objectives:
                    for mn in obj.metric_names:
                        obj_list.append((mn, obj.minimize))
            else:
                for mn in opt_config.objective.metric_names:
                    obj_list.append((mn, opt_config.objective.minimize))

            data = exp.lookup_data()
            df = data.df if hasattr(data, 'df') else data
            import pandas as pd
            if not df.empty and "step" in df.columns:
                filtered = df[pd.isna(df["step"])]
                if not filtered.empty:
                    df = filtered

            best_idx = None
            best_val = None
            primary_metric, primary_minimize = obj_list[0]
            for idx, params in matches:
                mdf = df[(df["trial_index"] == idx) & (df["metric_name"] == primary_metric)]
                if mdf.empty:
                    continue
                val = float(mdf.iloc[-1]["mean"])
                if best_val is None or (primary_minimize and val < best_val) or (not primary_minimize and val > best_val):
                    best_val = val
                    best_idx = idx

            if best_idx is None:
                return SafeJSONResponse(content={"match_count": len(matches), "error": "No metric data"})

            best_params = dict(exp.trials[best_idx].arm.parameters)
            non_group = {k: v for k, v in best_params.items() if k not in group_params}

            best_objectives = {}
            for mn, _ in obj_list:
                mdf = df[(df["trial_index"] == best_idx) & (df["metric_name"] == mn)]
                if not mdf.empty:
                    best_objectives[mn] = float(mdf.iloc[-1]["mean"])

        return SafeJSONResponse(content={
            "match_count": len(matches),
            "best_trial": best_idx,
            "best_objectives": best_objectives,
            "best_params": best_params,
            "non_group_params": non_group,
            "group": group,
        })
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/v1/analysis/group-interactions")
def post_group_interactions(request_body: dict):
    """Group interaction heatmap from second-order Sobol indices."""
    metric = request_body.get("metric")
    try:
        with _state.lock:
            client = _state.client
            if client is None:
                raise HTTPException(503, "Not initialized")
            _ensure_model_fitted()
            gs = client._generation_strategy
            exp = client._experiment

            import torch
            import numpy as np
            from ax.utils.sensitivity.sobol_measures import SobolSensitivityGPMean
            surr = gs.adapter.generator.surrogate
            model = surr.model
            param_names = [p for p in exp.search_space.parameters
                           if hasattr(exp.search_space.parameters[p], 'lower')]
            n = len(param_names)
            bounds = torch.zeros(2, n, dtype=torch.float64)
            bounds[1] = 1.0
            sobol = SobolSensitivityGPMean(model=model, bounds=bounds, num_mc_samples=1000)
            second_order = sobol.second_order_indices().detach().cpu().numpy()
            if second_order.ndim > 2:
                second_order = second_order.mean(axis=0)

            param_groups = getattr(exp.runner, '_parameter_groups', {})

            def get_primary_group(pname):
                gs = param_groups.get(pname, ["ungrouped"])
                return gs[0]

            group_names = sorted(set(get_primary_group(p) for p in param_names))
            g_idx = {g: i for i, g in enumerate(group_names)}
            ng = len(group_names)
            matrix = np.zeros((ng, ng))

            for i in range(n):
                for j in range(n):
                    gi = g_idx[get_primary_group(param_names[i])]
                    gj = g_idx[get_primary_group(param_names[j])]
                    matrix[gi][gj] += second_order[i][j]

            matrix = (matrix + matrix.T) / 2

        return SafeJSONResponse(content={
            "groups": group_names,
            "matrix": matrix.tolist(),
            "metric": metric,
        })
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


# Visualization endpoints

_DANGEROUS_PATTERNS = [
    "subprocess", "os.system", "os.popen", "os.remove", "os.unlink",
    "shutil.rmtree", "shutil.move", "eval(", "exec(", "__import__",
    "socket", "http.client", "urllib", "requests.",
]


def _validate_pvpython_script(source: str) -> str | None:
    """Validate a pvpython script. Returns error message or None if OK."""
    if "sys.argv[1]" not in source:
        return "Script must use sys.argv[1] for case path"
    if "sys.argv[2]" not in source:
        return "Script must use sys.argv[2] for screenshot filename"
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in source:
            return f"Script contains disallowed pattern: '{pattern}'"
    return None


@app.post("/api/v1/paraview-state")
async def upload_paraview_state(request: Request):
    """Upload a pvpython script for trial visualization.

    The script receives two arguments:
      sys.argv[1] — case path (absolute path to the OpenFOAM case directory)
      sys.argv[2] — screenshot filename (write screenshot to this file inside the case folder)

    Example script:
        import sys, os, paraview.simple as pvs
        case = sys.argv[1]; out = os.path.join(case, sys.argv[2])
        pvs.OpenFOAMReader(FileName=case + '/' + case.split('/')[-1] + '.foam')
        # ... set up pipeline, coloring, camera ...
        pvs.SaveScreenshot(out, pvs.GetActiveView(), ImageResolution=[1200, 800])
    """
    # Check if uploads are allowed
    orch = _state.orch_cfg
    if orch and not getattr(orch, "allow_pvpython_upload", True):
        raise HTTPException(403, "pvpython script upload is disabled "
                            "(set orchestration_settings.allow_pvpython_upload: true to enable)")
    import os, tempfile
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty request body")
    # Validate script content
    source = body.decode("utf-8", errors="replace")
    err = _validate_pvpython_script(source)
    if err:
        raise HTTPException(400, f"Script rejected: {err}")
    state_dir = os.path.join(tempfile.gettempdir(), "foambo_pv_state")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, "user_viz.py")
    with open(state_path, "wb") as f:
        f.write(body)
    _state.pv_state_path = state_path
    log.info(f"ParaView visualization script uploaded: {state_path} ({len(body)} bytes)")
    return SafeJSONResponse(content={"status": "ok", "path": state_path, "size": len(body)})


@app.delete("/api/v1/paraview-state")
def delete_paraview_state():
    """Remove the uploaded ParaView state file."""
    import os
    path = getattr(_state, "pv_state_path", None)
    if path and os.path.exists(path):
        os.unlink(path)
    _state.pv_state_path = None
    return SafeJSONResponse(content={"status": "ok"})


def _render_with_paraview_state(case_path: str, script_path: str) -> dict:
    """Render a trial by running the user's pvpython script.

    The script receives:
      sys.argv[1] — case_path (absolute path to OpenFOAM case directory)
      sys.argv[2] — screenshot filename (script writes to case_path/filename)
    """
    import subprocess, base64, os

    screenshot_name = ".foambo_screenshot.png"
    screenshot_path = os.path.join(case_path, screenshot_name)

    try:
        import shutil
        pvpython = shutil.which("pvpython")
        if pvpython is None:
            return {"error": "pvpython not found in PATH", "image": None}
        log.info(f"Running: {pvpython} {script_path} {case_path} {screenshot_name}")
        # Sandboxed environment: only expose case_path and essential ParaView vars
        safe_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "DISPLAY": os.environ.get("DISPLAY", ""),
            "MESA_GL_VERSION_OVERRIDE": os.environ.get("MESA_GL_VERSION_OVERRIDE", ""),
            "TMPDIR": case_path,
        }
        # Preserve ParaView/VTK/Python/foamBO-related env vars
        for k, v in os.environ.items():
            if k.startswith(("PV_", "VTK_", "PARAVIEW_", "LD_LIBRARY", "PYTHONPATH", "PYTHON", "FOAMBO_")):
                safe_env[k] = v
        # Use bubblewrap for OS-level isolation when available
        bwrap = shutil.which("bwrap")
        if bwrap:
            cmd = [
                bwrap,
                "--ro-bind", "/", "/",              # read-only root filesystem
                "--bind", case_path, case_path,     # read-write case folder
                "--bind", "/tmp", "/tmp",           # ParaView needs tmp
                "--unshare-net",                    # no network access
                "--die-with-parent",                # kill if server dies
                pvpython, script_path, case_path, screenshot_name,
            ]
        else:
            cmd = [pvpython, script_path, case_path, screenshot_name]
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=120,
            cwd=case_path,
            env=safe_env,
        )
        if result.stdout.strip():
            log.info(f"pvpython stdout: {result.stdout.strip()[:300]}")
        if result.stderr.strip():
            log.warning(f"pvpython stderr: {result.stderr.strip()[:300]}")
        if result.returncode != 0:
            return {"error": f"pvpython exited with code {result.returncode}: {result.stderr.strip()[:500]}", "image": None}

        if not os.path.exists(screenshot_path) or os.path.getsize(screenshot_path) == 0:
            return {"error": f"pvpython completed but no screenshot at {screenshot_path}", "image": None}

        with open(screenshot_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        return {"image": img_data, "case_path": case_path, "renderer": "paraview"}
    except FileNotFoundError:
        return {"error": "pvpython not found — install ParaView to use script rendering", "image": None}
    except subprocess.TimeoutExpired:
        return {"error": "pvpython timed out after 120s", "image": None}


@app.get("/api/v1/trials/{trial_index}/visualization")
def get_trial_visualization(trial_index: int):
    """Render an OpenFOAM case with PyVista (or ParaView state) and return base64 PNG."""
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        trial = client._experiment.trials.get(trial_index)
        if trial is None:
            raise HTTPException(404, f"Trial {trial_index} not found")
        run_meta = trial.run_metadata or {}
        case_path = run_meta.get("case_path") or (run_meta.get("job") or {}).get("case_path", "")

    if not case_path:
        raise HTTPException(404, "No case path for this trial")

    import os

    # If a ParaView script is uploaded, use pvpython instead of PyVista
    pvsm = getattr(_state, "pv_state_path", None)
    if pvsm and os.path.exists(pvsm):
        log.info(f"Rendering trial {trial_index} with pvpython script: {pvsm}")
        result = _render_with_paraview_state(case_path, pvsm)
        if result.get("image"):
            log.info(f"Trial {trial_index} rendered successfully")
        else:
            log.warning(f"Trial {trial_index} render failed: {result.get('error')}")
        status = 200 if result.get("image") else 500
        return SafeJSONResponse(content=result, status_code=status)

    # Check if the case has a mesh (polyMesh/points or constant/polyMesh/points)
    poly_mesh = os.path.join(case_path, "constant", "polyMesh", "points")
    if not os.path.exists(poly_mesh):
        poly_mesh = os.path.join(case_path, "polyMesh", "points")
    if not os.path.exists(poly_mesh):
        return SafeJSONResponse(
            content={"error": "No mesh found — trial may still be running or meshing hasn't completed", "image": None},
            status_code=400)

    foam_file = os.path.join(case_path, f"{os.path.basename(case_path)}.foam")
    if not os.path.exists(foam_file):
        try:
            with open(foam_file, "w") as f:
                f.write("")
        except Exception:
            raise HTTPException(404, f"Cannot create .foam file at {foam_file}")

    try:
        import pyvista as pv
        import base64
        import tempfile
        pv.OFF_SCREEN = True
        reader = pv.OpenFOAMReader(foam_file)
        if reader.number_time_points == 0:
            return SafeJSONResponse(
                content={"error": "No time steps available yet", "image": None},
                status_code=400)
        reader.set_active_time_point(reader.number_time_points - 1)
        mesh = reader.read()
        internal = mesh.get("internalMesh") if hasattr(mesh, "get") else mesh

        if internal is None or internal.n_points == 0:
            return SafeJSONResponse(content={"error": "Empty mesh — solver may not have started yet", "image": None})

        plotter = pv.Plotter(off_screen=True, window_size=[1200, 800])
        plotter.add_mesh(internal, show_edges=False, opacity=1.0)
        plotter.view_isometric()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plotter.screenshot(tmp.name)
        plotter.close()

        with open(tmp.name, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(tmp.name)

        return SafeJSONResponse(content={
            "image": img_data,
            "case_path": case_path,
            "n_points": internal.n_points,
            "n_cells": internal.n_cells,
        })
    except ImportError:
        raise HTTPException(501, "PyVista not installed")
    except Exception as e:
        log.error(f"Trial visualization failed: {e}")
        return SafeJSONResponse(content={"error": str(e), "image": None}, status_code=500)


# Server lifecycle

def start_api_server(client, raw_cfg, orch_cfg, host: str = "127.0.0.1",
                     port: int = 8098) -> threading.Thread | None:
    """Start the API server in a background daemon thread.

    Args:
        client: The Ax Client instance (shared, not copied).
        raw_cfg: The OmegaConf DictConfig.
        orch_cfg: The typed ConfigOrchestratorOptions.
        host: Bind address.
        port: Bind port. 0 to disable.

    Returns:
        The daemon thread, or None if port is 0.
    """
    if port == 0:
        log.info("API server disabled (api_port=0)")
        return None

    _state.client = client
    _state.raw_cfg = raw_cfg
    _state.orch_cfg = orch_cfg
    _state.start_time = time.time()
    _state.last_callback = time.time()

    # Security checks at startup
    import shutil
    if getattr(orch_cfg, "allow_pvpython_upload", True):
        if shutil.which("bwrap") is None:
            log.warning(
                "bubblewrap (bwrap) not found — pvpython scripts will run without OS-level sandboxing. "
                "To disable script uploads entirely, set orchestration_settings.allow_pvpython_upload: false"
            )

    import uvicorn
    import socket

    # If the requested port is in use, pick a random available one
    actual_port = port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            s.bind((host, 0))
            actual_port = s.getsockname()[1]
            log.warning(f"Port {port} in use, using {actual_port} instead")

    config = uvicorn.Config(app, host=host, port=actual_port, log_level="warning",
                            access_log=False)
    server = uvicorn.Server(config)
    _state._uvicorn_server = server

    thread = threading.Thread(target=server.run, daemon=True, name="foambo-api")
    thread.start()
    log.info(f"API server started at http://{host}:{actual_port}/api/v1/")
    return thread


def stop_api_server():
    """Signal the uvicorn server to shut down gracefully."""
    server = getattr(_state, '_uvicorn_server', None)
    if server:
        server.should_exit = True
        log.info("API server shutdown requested")


def update_api_state(client):
    """Called from the optimizer callback to refresh cached data.

    Bumps version counters so ETag-based caching works correctly.
    """
    _state.update(client)
