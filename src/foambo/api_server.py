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


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None before JSON encoding."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj

def _safe_json(data: Any) -> str:
    return json.dumps(_sanitize(data), cls=_SafeEncoder)


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
    execution_time_s: Optional[float] = None

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
    context_point: Optional[dict] = None  # specific context, or None for all + aggregated

class PredictResponse(BaseModel):
    predictions: dict

class ParetoResponse(BaseModel):
    frontier: List[dict]
    hypervolume: Optional[float] = None
    reference_point: Optional[dict] = None
    model_predictions_used: bool

class SweepRequest(BaseModel):
    base_parameters: dict
    sweep_params: list[str]
    n_points: int = 25
    context_point: Optional[dict] = None  # specific context, or None for all + aggregated

class GroupSweepRequest(BaseModel):
    frozen_group: str
    base_parameters: dict
    n_points: int = 25

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
    "orchestration_settings.trial_timeout",
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

        # Push-based remote runner state (consumed by FoamJobRunner.poll_trial)
        self.event_log: list[dict] = []  # timestamped events (dashboard feed)
        self.trial_status_overrides: dict[int, str] = {}  # trial_idx -> "completed"/"failed"
        self.trial_pushed_metrics: dict[int, list] = {}  # trial_idx -> [{metrics, step, ts}]
        self.trial_heartbeats: dict[int, float] = {}

        self.standalone = False  # True when launched via --config-builder

        # Unique session ID so ETags never match across server restarts
        self._session_id = hex(int(time.time() * 1000))[-8:]
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
        return f'"{self._session_id}-{endpoint}-{self._versions.get(endpoint, 0)}"'

    def log_event(self, trial: int, event_type: str, detail: str = "", ts: float | None = None):
        self.event_log.append({
            "ts": ts or time.time(),
            "trial": trial,
            "type": event_type,
            "detail": detail,
        })

    def update(self, client):
        """Called from the optimizer callback to refresh state."""
        with self.lock:
            # Detect newly completed/failed trials for event logging
            old_terminal: set[int] = set()
            if self.client is not None:
                try:
                    from ax.core.base_trial import TrialStatus as _TS
                    for idx, t in self.client._experiment.trials.items():
                        if t.status in (_TS.COMPLETED, _TS.FAILED, _TS.EARLY_STOPPED):
                            old_terminal.add(idx)
                except Exception:
                    pass

            self.client = client
            self.last_callback = time.time()
            self.callback_seq += 1
            # Bump all poll-driven endpoints
            self.bump(
                "trials", "objectives", "streaming",
                "generation", "pareto", "config",
            )

            # Log newly completed/failed trials
            try:
                from ax.core.base_trial import TrialStatus as _TS
                for idx, t in client._experiment.trials.items():
                    if idx not in old_terminal and t.status in (_TS.COMPLETED, _TS.FAILED, _TS.EARLY_STOPPED):
                        self.log_event(idx, f"trial.{t.status.name.lower()}", "")
            except Exception:
                pass


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
    if isinstance(exc, HTTPException):
        raise exc  # let FastAPI handle HTTP exceptions normally
    import traceback
    log.error(f"API {request.method} {request.url.path} failed: {exc}\n{traceback.format_exc()}")
    return SafeJSONResponse(content={"error": str(exc)}, status_code=500)


def _resolve_template(filename: str) -> str:
    """Resolve a template file path, trying importlib.resources first."""
    import importlib.resources
    try:
        return str(importlib.resources.files("foambo").joinpath(f"templates/{filename}"))
    except Exception:
        import os
        return os.path.join(os.path.dirname(__file__), "templates", filename)


@app.get("/", response_class=Response)
def serve_dashboard():
    """Serve the web dashboard HTML."""
    path = _resolve_template("dashboard.html")
    with open(path) as f:
        html = f.read()
    return Response(content=html, media_type="text/html")


_STATIC_TYPES = {
    ".css": "text/css",
    ".js": "application/javascript",
}


@app.get("/static/{filename}")
def serve_static(filename: str):
    """Serve static assets (CSS/JS) from templates dir."""
    import os
    ext = os.path.splitext(filename)[1]
    if ext not in _STATIC_TYPES:
        raise HTTPException(404, "Not found")
    path = _resolve_template(filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Not found")
    with open(path) as f:
        content = f.read()
    return Response(content=content, media_type=_STATIC_TYPES[ext])


@app.get("/config-builder", response_class=Response)
def serve_config_builder():
    """Serve the config builder single-page app."""
    path = _resolve_template("config_builder.html")
    with open(path) as f:
        html = f.read()
    return Response(content=html, media_type="text/html")


@app.get("/api/v1/config/json-schema")
def config_json_schema():
    """Return the raw Pydantic JSON Schema for FoamBOConfig (used by config builder)."""
    from foambo.orchestrate import FoamBOConfig
    return SafeJSONResponse(content=FoamBOConfig.model_json_schema())


@app.post("/api/v1/config/validate")
async def config_validate(request: Request):
    """Validate a config dict against FoamBOConfig."""
    from foambo.orchestrate import FoamBOConfig
    body = await request.json()
    try:
        FoamBOConfig.model_validate(body)
        return SafeJSONResponse(content={"valid": True, "errors": []})
    except Exception as exc:
        errors = []
        if hasattr(exc, 'errors'):
            errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", str(e))} for e in exc.errors()]
        else:
            errors = [{"loc": [], "msg": str(exc)}]
        return SafeJSONResponse(content={"valid": False, "errors": errors}, status_code=422)


@app.post("/api/v1/config/preflight")
async def config_preflight(request: Request):
    """Run preflight checks on a config dict."""
    from omegaconf import DictConfig
    from foambo.preflight import static_checks
    body = await request.json()
    try:
        cfg = DictConfig(body)
        result = static_checks(cfg)
        checks = [{"name": name, "status": status, "detail": detail or ""}
                  for status, name, detail in result.checks]
        return SafeJSONResponse(content={"ok": result.ok, "checks": checks})
    except Exception as exc:
        return SafeJSONResponse(
            content={"ok": False, "checks": [{"name": "Preflight", "status": "FAIL", "detail": str(exc)}]},
            status_code=500,
        )


@app.get("/api/v1/config/docs")
def config_docs():
    """Return all config docs: field descriptions, concepts, and tutorials."""
    from foambo.default_config import get_config_docs
    return SafeJSONResponse(content=get_config_docs())


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


def _parse_trial_deps(runner, trial_index: int, trial) -> list:
    """Parse dependency info from trial_registry (live) or run_metadata (persisted)."""
    raw_deps = None
    # Try live trial_registry first
    if hasattr(runner, 'trial_registry') and trial_index in runner.trial_registry:
        reg = runner.trial_registry[trial_index]
        if "dependencies" in reg:
            raw_deps = reg["dependencies"]
    # Fallback to persisted run_metadata
    if raw_deps is None and trial.run_metadata:
        raw_deps = trial.run_metadata.get("dependencies")
    if not raw_deps:
        return []
    if isinstance(raw_deps, list):
        return raw_deps
    # Convert dict format to list
    deps = []
    for name, info in raw_deps.items():
        if name.startswith("_"):
            continue
        if isinstance(info, dict) and "source_trial_index" in info:
            deps.append({
                "name": name,
                "source_index": info["source_trial_index"],
                "source_path": info.get("source_case_path", ""),
                "actions_applied": info.get("actions_applied", 0),
                "phased_actions": info.get("phased_actions", []),
            })
    return deps


def _get_trials() -> dict:
    """Extract all trial data under lock."""
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        exp = client._experiment
        from ax.core.base_trial import TrialStatus as AxTrialStatus

        # Single bulk data lookup for all completed trials
        completed_indices = [idx for idx, t in exp.trials.items()
                             if t.status in (AxTrialStatus.COMPLETED, AxTrialStatus.EARLY_STOPPED)]
        all_metrics: Dict[int, Dict[str, float]] = {}
        if completed_indices:
            try:
                import pandas as pd
                data = exp.lookup_data(trial_indices=completed_indices)
                full_df = data.full_df if hasattr(data, 'full_df') else (data.df if hasattr(data, 'df') else data)
                if full_df is not None and not full_df.empty:
                    # First pass: non-streaming rows (final metric values)
                    if "step" in full_df.columns:
                        non_stream = full_df[pd.isna(full_df["step"])]
                        if not non_stream.empty:
                            for _, row in non_stream.iterrows():
                                tidx_r = int(row["trial_index"])
                                val = row["mean"]
                                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                                    all_metrics.setdefault(tidx_r, {})[row["metric_name"]] = float(val)
                    # Second pass: for trials with missing metrics, use last streaming value
                    if "step" in full_df.columns:
                        streaming = full_df[pd.notna(full_df["step"])].sort_values("step")
                        for _, row in streaming.drop_duplicates(
                                subset=["trial_index", "metric_name"], keep="last").iterrows():
                            tidx_r = int(row["trial_index"])
                            mname = row["metric_name"]
                            if tidx_r not in all_metrics or mname not in all_metrics.get(tidx_r, {}):
                                val = row["mean"]
                                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                                    all_metrics.setdefault(tidx_r, {})[mname] = float(val)
            except Exception as e:
                log.debug(f"Bulk metric lookup failed: {e}")

        # Ensure all completed/ES trials have entries for all known metric names (null if missing)
        all_metric_names = set()
        for m in all_metrics.values():
            all_metric_names.update(m.keys())
        for tidx in completed_indices:
            trial_m = all_metrics.get(tidx, {})
            for mname in all_metric_names:
                if mname not in trial_m:
                    trial_m[mname] = None
            all_metrics[tidx] = trial_m

        # Build completion-time map from event log
        _terminal_event_types = {"trial.completed", "trial.failed", "trial.early_stopped", "trial.killed"}
        completion_ts: Dict[int, float] = {}
        for ev in _state.event_log:
            if ev.get("type") in _terminal_event_types:
                completion_ts[ev["trial"]] = ev["ts"]

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

            deps = _parse_trial_deps(exp.runner, tidx, trial)
            metrics = all_metrics.get(tidx, {})

            # Check if subprocess already exited while Ax hasn't polled yet
            job_id = trial.run_metadata.get("job_id") if trial.run_metadata else None
            if status == "RUNNING" and job_id is not None:
                from foambo.metrics import job_finish_times
                finish_info = job_finish_times.get(job_id)  # (exit_time, returncode) or None
                if finish_info is not None:
                    status = "COMPLETED" if finish_info[1] == 0 else "FAILED"

            dispatch_time = trial.run_metadata.get("dispatch_time") if trial.run_metadata else None
            exec_time = None
            if dispatch_time is not None:
                if status in ("COMPLETED", "FAILED", "EARLY_STOPPED"):
                    end_ts = completion_ts.get(tidx)
                    # Fall back to Ax's trial.time_completed when event log is unavailable (e.g. --no-opt reload)
                    if end_ts is None and trial.time_completed is not None:
                        end_ts = trial.time_completed.timestamp()
                    # Fall back to actual subprocess exit time (before Ax polls)
                    if end_ts is None and job_id is not None:
                        from foambo.metrics import job_finish_times
                        fi = job_finish_times.get(job_id)
                        if fi is not None:
                            end_ts = fi[0]
                    if end_ts is not None:
                        exec_time = end_ts - dispatch_time or None
                elif status == "RUNNING":
                    exec_time = time.time() - dispatch_time

            trials.append({
                "index": tidx,
                "status": status,
                "parameters": params,
                "gen_node": gen_node,
                "case_path": case_path,
                "dependencies": deps,
                "metrics": metrics,
                "execution_time_s": exec_time,
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

        deps = _parse_trial_deps(exp.runner, index, trial)

        metrics = {}
        if trial.status == AxTrialStatus.COMPLETED:
            try:
                data = exp.lookup_data(trial_indices=[index])
                df = data.df if hasattr(data, 'df') else data
                if not df.empty:
                    for _, row in df.iterrows():
                        if row.get("step") is None or (hasattr(row["step"], '__float__') and row["step"] != row["step"]):
                            metrics[row["metric_name"]] = row["mean"]
            except Exception as e:
                log.debug(f"Metric lookup failed for trial {index}: {e}")

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

        # Check if subprocess already exited while Ax hasn't polled yet
        status = trial.status.name
        job_id = trial.run_metadata.get("job_id") if trial.run_metadata else None
        if status == "RUNNING" and job_id is not None:
            from foambo.metrics import job_finish_times
            finish_info = job_finish_times.get(job_id)
            if finish_info is not None:
                status = "COMPLETED" if finish_info[1] == 0 else "FAILED"

        return {
            "index": index,
            "status": status,
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
        from ax.core.base_trial import TrialStatus
        completed_indices = [idx for idx, t in exp.trials.items()
                             if t.status in (TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED)]
        data = exp.lookup_data(trial_indices=completed_indices) if completed_indices else exp.lookup_data()
        df = data.df if hasattr(data, 'df') else data
        # Filter to only non-streaming rows (step is NaN) for objective metrics
        if df is not None and not df.empty and "step" in df.columns:
            import pandas as pd
            non_streaming = df[pd.isna(df["step"])]
            if not non_streaming.empty:
                df = non_streaming

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

        # First pass: initialize strategy nodes with 0 counts. Target computation
        # is deferred to a second pass so we can subtract external trials
        # (baseline, seeded) from `use_all_trials_in_exp` MinTrials budgets.
        max_trials = _state.orch_cfg.max_trials if _state.orch_cfg else 0
        node_raw_tc: Dict[str, Any] = {}
        for node in gs._nodes:
            node_counts[node.name] = 0
            node_targets[node.name] = None
            node_raw_tc[node.name] = None
            for tc in node.transition_criteria:
                if hasattr(tc, "threshold") and isinstance(getattr(tc, "threshold", None), (int, float)):
                    node_raw_tc[node.name] = ("min_trials", tc)
                    break
                if tc.criterion_class == "AutoTransitionAfterGen":
                    node_raw_tc[node.name] = ("auto", tc)
                    break

        # Count by inspecting each trial's generator_run node name.
        # External counts (baseline, seeded, or any trial whose origin node is
        # not in the current strategy) are tracked separately because they DO
        # count toward downstream MinTrials(use_all_trials_in_exp=True) budgets.
        external_total = 0
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
                external_total += 1
            elif node_name:
                # Node from a different strategy (e.g. baseline ManualGenerationNode)
                node_counts.setdefault(node_name, 0)
                node_counts[node_name] += 1
                node_targets.setdefault(node_name, 1)
                external_total += 1

        # Second pass: compute per-node targets.
        # For MinTrials with `use_all_trials_in_exp=True`, the threshold is
        # measured against the whole experiment, so a node's *own* budget is
        # threshold minus whatever was already committed before it (external
        # + upstream strategy nodes). For node-local MinTrials, the threshold
        # is the budget as-is.
        running = external_total
        allocated = 0
        for node in gs._nodes:
            kind_tc = node_raw_tc.get(node.name)
            target: int | None
            if kind_tc is None:
                target = None
            else:
                kind, tc = kind_tc
                if kind == "auto":
                    target = 1
                else:
                    thresh = int(tc.threshold)
                    if getattr(tc, "use_all_trials_in_exp", False):
                        target = max(0, thresh - running)
                    else:
                        target = thresh
            node_targets[node.name] = target
            if target is not None:
                running += target
                allocated += target
        # Last node (usually MBM) gets remaining budget from max_trials.
        last_node = gs._nodes[-1].name if gs._nodes else None
        if last_node and node_targets.get(last_node) is None and max_trials > 0:
            node_targets[last_node] = max_trials - allocated - external_total

        # Reorder: put "baseline" first if present so the dashboard shows it
        # at the start of the generation strategy pane.
        if "baseline" in node_counts:
            node_counts = {"baseline": node_counts["baseline"],
                           **{k: v for k, v in node_counts.items() if k != "baseline"}}
            node_targets = {"baseline": node_targets.get("baseline"),
                            **{k: v for k, v in node_targets.items() if k != "baseline"}}

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
            import torch
            from botorch.utils.multi_objective.hypervolume import Hypervolume
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
                # BoTorch Hypervolume assumes maximization: negate minimized objectives
                ref_t = torch.tensor([
                    -ref_point[n] if minimize_flags.get(n, True) else ref_point[n]
                    for n in obj_names
                ], dtype=torch.double)
                hv_calc = Hypervolume(ref_point=ref_t)
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
                                # Negate minimized objectives for maximization convention
                                if minimize_flags.get(obj_name, True):
                                    val = -val
                                point.append(val)
                            if len(point) == len(obj_names):
                                points.append(point)
                        if points:
                            pts_t = torch.tensor(points, dtype=torch.double)
                            hv = hv_calc.compute(pts_t)
                            hv_trace.append({"trial": completed[i-1].index, "value": float(hv)})
                    except Exception as e:
                        log.debug("HV computation failed at trial %d: %s",
                                  completed[i-1].index if i > 0 else -1, e)
        except Exception as e:
            log.debug("HV trace setup failed: %s", e)

        return {
            "frontier": frontier,
            "hypervolume": hv_trace[-1]["value"] if hv_trace else None,
            "hypervolume_trace": hv_trace,
            "reference_point": ref_point or None,
            "model_predictions_used": model_used,
        }



def _disable_context_transforms(gs) -> list[tuple]:
    """Temporarily disable SubstituteContextFeatures on all surrogate models.

    Returns a list of (transform, original_flag) tuples to restore later.
    """
    disabled = []
    try:
        from foambo.robustness import SubstituteContextFeatures
        surr = getattr(gs.adapter.generator, "surrogate", None)
        if surr is None or surr.model is None:
            return disabled
        models = getattr(surr.model, "models", [surr.model])
        for m in models:
            tf = getattr(m, "input_transform", None)
            if tf is None:
                continue
            # ChainedInputTransform stores transforms as named children
            transforms = getattr(tf, "items", lambda: [(None, tf)])()
            for name, t in transforms:
                if isinstance(t, SubstituteContextFeatures):
                    disabled.append((t, t.transform_on_eval))
                    t.transform_on_eval = False
    except Exception:
        pass
    return disabled


def _restore_context_transforms(disabled: list[tuple]) -> None:
    """Restore SubstituteContextFeatures flags from _disable_context_transforms."""
    for transform, original_flag in disabled:
        transform.transform_on_eval = original_flag


def _do_predict(parameters: dict, context_point: dict | None = None) -> dict:
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

        def _extract_float(v):
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, dict):
                return float(v.get("mean", v.get("value", 0)))
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return float(v[0])
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        try:
            from ax.core.observation import ObservationFeatures
            params = dict(parameters)
            if context_point:
                params.update(context_point)
            obs = [ObservationFeatures(parameters=params)]
            _disabled_transforms = _disable_context_transforms(gs)
            try:
                means, sems = gs.adapter.predict(obs)
            finally:
                _restore_context_transforms(_disabled_transforms)
            result = {}
            for metric_name in means:
                result[metric_name] = {
                    "mean": _extract_float(means[metric_name]),
                    "sem": _extract_float(sems.get(metric_name, 0)),
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
        from omegaconf import OmegaConf, DictConfig
        container = OmegaConf.to_container(raw_cfg, resolve=True) if isinstance(raw_cfg, DictConfig) else raw_cfg

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

        # Resolve lazily-typed fields (e.g. robust_optimization: Any → RobustOptimizationConfig)
        if not is_model and annotation is Any:
            try:
                resolved = _resolve_lazy_model(name)
                if resolved is not None:
                    is_model = True
                    annotation = resolved
            except Exception:
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


def _resolve_lazy_model(field_name: str):
    """Resolve lazily-imported Pydantic models for schema introspection."""
    if field_name == "robust_optimization":
        from foambo.robustness import RobustOptimizationConfig
        return RobustOptimizationConfig
    return None


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
        # Propagate trial_timeout to the runner if it was changed
        if any(p == "orchestration_settings.trial_timeout" for p in updated):
            try:
                runner = _state.client._experiment.runner
                runner._trial_timeout = orch_cfg.trial_timeout
            except Exception:
                pass

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
    except HTTPException:
        raise
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
    data = _do_predict(req.parameters, context_point=req.context_point)
    return SafeJSONResponse(content=data)


@app.post("/api/v1/predict/robust")
def predict_robust(req: PredictRequest):
    """Predict at each context scenario. Returns per-scenario predictions,
    nominal (mean context) prediction, CVaR, std, and percentile rank
    vs all completed trials."""
    import numpy as np
    raw_cfg = _state.raw_cfg
    if not raw_cfg or not raw_cfg.get("robust_optimization"):
        return SafeJSONResponse(content={"error": "Not in robust mode"}, status_code=400)

    rc = raw_cfg["robust_optimization"]
    alpha = max(rc.get("robustness", 0.5), 0.05)

    # Get context points
    ctx_points = rc.get("context_points")
    if ctx_points is None:
        try:
            from foambo.robustness import resolve_context_points, RobustOptimizationConfig
            from foambo.orchestrate import ExperimentOptions
            robust_cfg = RobustOptimizationConfig.model_validate(dict(rc))
            exp_cfg = ExperimentOptions.model_validate(dict(raw_cfg["experiment"]))
            resolve_context_points(robust_cfg, exp_cfg)
            ctx_points = robust_cfg.context_points
        except Exception:
            ctx_points = []
    if not ctx_points:
        return SafeJSONResponse(content={"error": "No context points"}, status_code=500)

    # Get context param names
    ctx_groups = rc.get("context_groups", [])
    ctx_param_names = []
    for p in raw_cfg.get("experiment", {}).get("parameters", []):
        p_groups = p.get("groups", []) if hasattr(p, "get") else getattr(p, "groups", [])
        if any(g in ctx_groups for g in (p_groups or [])):
            ctx_param_names.append(p.get("name", "") if hasattr(p, "get") else getattr(p, "name", ""))

    # Compute nominal context (mean of context points)
    nominal_ctx = {}
    for cn in ctx_param_names:
        nominal_ctx[cn] = float(np.mean([cp[cn] for cp in ctx_points]))

    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        gs = client._generation_strategy
        if gs is None or gs.adapter is None:
            raise HTTPException(503, "Model not fitted yet")

        from ax.core.observation import ObservationFeatures
        params = dict(req.parameters)

        # Predict at each context scenario
        obs_list = []
        for cp in ctx_points:
            p = dict(params)
            p.update(cp)
            obs_list.append(ObservationFeatures(parameters=p))

        # Also predict at nominal
        nom_params = dict(params)
        nom_params.update(nominal_ctx)
        obs_list.append(ObservationFeatures(parameters=nom_params))

        _disabled = _disable_context_transforms(gs)
        try:
            means, sems = gs.adapter.predict(obs_list)
        finally:
            _restore_context_transforms(_disabled)

        n_ctx = len(ctx_points)
        scenarios = []
        metric_stats = {}

        for i, cp in enumerate(ctx_points):
            sc = {"context": cp, "predictions": {}}
            for metric in means:
                sc["predictions"][metric] = {
                    "mean": float(means[metric][i]),
                    "sem": float(sems.get(metric, {}).get(metric, [0]*len(obs_list))[i])
                          if isinstance(sems.get(metric), dict) else 0.0,
                }
            scenarios.append(sc)

        # Nominal prediction (last obs)
        nominal_preds = {}
        for metric in means:
            nominal_preds[metric] = {
                "mean": float(means[metric][n_ctx]),
                "sem": float(sems.get(metric, {}).get(metric, [0]*len(obs_list))[n_ctx])
                      if isinstance(sems.get(metric), dict) else 0.0,
            }

        # Per-metric stats across contexts
        for metric in means:
            vals = np.array([float(means[metric][i]) for i in range(n_ctx)])
            sorted_vals = np.sort(vals)
            n_tail = max(1, int(len(sorted_vals) * (1 - alpha)))
            cvar = float(np.mean(sorted_vals[:n_tail]))
            metric_stats[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "cvar": cvar,
                "nominal": nominal_preds[metric]["mean"],
                "gap": float(nominal_preds[metric]["mean"] - cvar),
            }

        # Percentile rank: how does this design's CVaR compare to completed trials?
        from ax.core.base_trial import TrialStatus as AxTrialStatus
        from ax.core.trial import Trial
        completed = [t for t in client._experiment.trials.values()
                     if t.status == AxTrialStatus.COMPLETED and isinstance(t, Trial)]

        percentile_rank = {}
        if len(completed) >= 3:
            # Sample a few trials to compute CVaR distribution
            sample = completed[-min(30, len(completed)):]
            for metric in means:
                trial_cvars = []
                for trial in sample:
                    arm = trial.arm
                    if arm is None:
                        continue
                    t_obs = []
                    for cp in ctx_points:
                        p = dict(arm.parameters)
                        p.update(cp)
                        t_obs.append(ObservationFeatures(parameters=p))
                    try:
                        t_means, _ = gs.adapter.predict(t_obs)
                        t_vals = np.array([float(t_means[metric][j]) for j in range(n_ctx)])
                        t_sorted = np.sort(t_vals)
                        t_cvar = float(np.mean(t_sorted[:n_tail]))
                        trial_cvars.append(t_cvar)
                    except Exception:
                        continue
                if trial_cvars:
                    my_cvar = metric_stats[metric]["cvar"]
                    rank = float(np.mean([1 if my_cvar >= tc else 0 for tc in trial_cvars]) * 100)
                    percentile_rank[metric] = round(rank, 1)

    return SafeJSONResponse(content={
        "scenarios": scenarios,
        "nominal": nominal_preds,
        "nominal_context": nominal_ctx,
        "stats": metric_stats,
        "percentile_rank": percentile_rank,
        "alpha": alpha,
        "context_params": ctx_param_names,
        "n_contexts": n_ctx,
    })


def _do_sweep(base_params: dict, sweep_params: list[str], n_points: int = 25,
              context_point: dict | None = None) -> dict:
    """Sweep each param across its bounds."""
    import numpy as np
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

        exp = client._experiment
        from ax.core.observation import ObservationFeatures
        from ax.core.parameter import ChoiceParameter

        def _build_obs(base, pname, xs):
            obs = []
            for xval in xs:
                p = dict(base)
                p[pname] = xval
                obs.append(ObservationFeatures(parameters=p))
            return obs

        def _val(v, idx):
            if isinstance(v, (list, tuple)):
                return float(v[idx]) if idx < len(v) else 0.0
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        params = dict(base_params)
        if context_point:
            params.update(context_point)

        all_obs = []
        sweep_meta = []
        for pname in sweep_params:
            param = exp.search_space.parameters.get(pname)
            if param is None or isinstance(param, ChoiceParameter):
                continue
            lo, hi = param.lower, param.upper
            if param.parameter_type.name == "INT":
                xs = np.unique(np.linspace(lo, hi, n_points).astype(int)).tolist()
            else:
                xs = np.linspace(lo, hi, n_points).tolist()
            start = len(all_obs)
            all_obs.extend(_build_obs(params, pname, xs))
            sweep_meta.append((pname, xs, start, len(xs)))

        base_idx = len(all_obs)
        all_obs.append(ObservationFeatures(parameters=params))

        # Temporarily disable SubstituteContextFeatures so predictions
        # return one result per input (not n_w per input).
        _disabled_transforms = _disable_context_transforms(gs)
        try:
            means, sems = gs.adapter.predict(all_obs)
        finally:
            _restore_context_transforms(_disabled_transforms)

        curves = []
        for pname, xs, start, count in sweep_meta:
            preds = {}
            for metric in means:
                m_means = [_val(means[metric], start + i) for i in range(count)]
                cov_row = sems.get(metric, {})
                m_sems_raw = cov_row.get(metric, 0) if isinstance(cov_row, dict) else cov_row
                m_sems = [_val(m_sems_raw, start + i) for i in range(count)]
                preds[metric] = {"mean": m_means, "sem": m_sems}
            curves.append({"param_name": pname, "x_values": xs, "predictions": preds})

        base_preds = {}
        for metric in means:
            base_preds[metric] = {
                "mean": _val(means[metric], base_idx),
                "sem": _val(sems.get(metric, {}).get(metric, 0) if isinstance(sems.get(metric), dict) else sems.get(metric, 0), base_idx),
            }
        return {"curves": curves, "base_predictions": base_preds}


@app.post("/api/v1/predict/sweep")
def predict_sweep(req: SweepRequest):
    data = _do_sweep(req.base_parameters, req.sweep_params, req.n_points,
                     context_point=req.context_point)
    return SafeJSONResponse(content=data)


@app.post("/api/v1/predict/group-sweep")
def predict_group_sweep(req: GroupSweepRequest):
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        param_groups = getattr(client._experiment.runner, '_parameter_groups', {})
    # Find params NOT in the frozen group
    frozen_params = {n for n, gs in param_groups.items() if req.frozen_group in gs}
    all_params = list(req.base_parameters.keys())
    sweep_params = [p for p in all_params if p not in frozen_params]
    if not sweep_params:
        raise HTTPException(400, f"No unfrozen parameters to sweep (all belong to '{req.frozen_group}')")
    data = _do_sweep(req.base_parameters, sweep_params, req.n_points)
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
        exp = client._experiment
        from ax.core.base_trial import TrialStatus as AxTrialStatus
        completed = sum(1 for t in list(exp.trials.values()) if t.status == AxTrialStatus.COMPLETED)
        running = sum(1 for t in list(exp.trials.values()) if t.status == AxTrialStatus.RUNNING)
        total = len(exp.trials)
        has_model = False
        try:
            adapter = client._generation_strategy.adapter
            if adapter is not None:
                surr = getattr(adapter.generator, "surrogate", None)
                has_model = surr is not None and getattr(surr, "model", None) is not None
        except Exception:
            pass

    # Derive robust mode info from raw config
    robust_info = None
    raw_cfg = _state.raw_cfg
    if raw_cfg and raw_cfg.get("robust_optimization"):
        rc = raw_cfg["robust_optimization"]
        # Get context param names from config
        ctx_groups = rc.get("context_groups", [])
        ctx_params = []
        for p in raw_cfg.get("experiment", {}).get("parameters", []):
            if hasattr(p, "get"):
                p_groups = p.get("groups", [])
            else:
                p_groups = getattr(p, "groups", [])
            if any(g in ctx_groups for g in (p_groups or [])):
                ctx_params.append(p.get("name", "") if hasattr(p, "get") else getattr(p, "name", ""))
        robust_info = {
            "risk_measure": rc.get("risk_measure", "auto"),
            "robustness": rc.get("robustness", 0.5),
            "alpha": max(rc.get("robustness", 0.5), 0.05),
            "context_groups": ctx_groups,
            "context_params": ctx_params,
            "context_samples": rc.get("context_samples", 10),
        }

    return SafeJSONResponse(content={
        "running": True,
        "uptime_s": round(time.time() - _state.start_time, 1),
        "last_callback_s_ago": round(time.time() - _state.last_callback, 1),
        "trials_completed": completed,
        "trials_running": running,
        "trials_total": total,
        "model_fitted": has_model,
        "timing": getattr(_state, '_timing', None),
        "robust": robust_info,
    })


@app.get("/api/v1/robust/context-points")
def get_robust_context_points():
    """Return the context scenarios used for robust optimization."""
    raw_cfg = _state.raw_cfg
    if not raw_cfg or not raw_cfg.get("robust_optimization"):
        return SafeJSONResponse(content={"points": []})
    rc = raw_cfg["robust_optimization"]
    points = rc.get("context_points")
    if points is None:
        # Context points are auto-generated — try to reconstruct from config
        try:
            import logging as _logging
            from foambo.robustness import resolve_context_points, RobustOptimizationConfig
            from foambo.orchestrate import ExperimentOptions
            robust_cfg = RobustOptimizationConfig.model_validate(dict(rc))
            exp_cfg = ExperimentOptions.model_validate(dict(raw_cfg["experiment"]))
            # Suppress logging during context point generation to avoid
            # format errors from custom log formatters outside optimize()
            _rob_log = _logging.getLogger("ax.foambo.robustness")
            _old_level = _rob_log.level
            _rob_log.setLevel(_logging.WARNING)
            try:
                resolve_context_points(robust_cfg, exp_cfg)
            finally:
                _rob_log.setLevel(_old_level)
            points = robust_cfg.context_points
        except Exception:
            points = []
    return SafeJSONResponse(content={"points": points or []})


@app.get("/api/v1/robust/pareto-robustness")
def get_pareto_robustness():
    """For each Pareto point, predict across all context scenarios.

    Returns per-point, per-objective stats (mean, std, min, max, cvar)
    and the raw predictions for box plots.
    """
    import numpy as np
    raw_cfg = _state.raw_cfg
    if not raw_cfg or not raw_cfg.get("robust_optimization"):
        return SafeJSONResponse(content={"points": []})

    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        gs = client._generation_strategy
        if gs is None or gs.adapter is None:
            raise HTTPException(503, "Model not fitted yet")

        # Get Pareto frontier
        from ax.core.objective import MultiObjective
        is_moo = isinstance(
            client._experiment.optimization_config.objective, MultiObjective)
        if not is_moo:
            return SafeJSONResponse(content={"points": []})

        try:
            front = client.get_pareto_frontier(use_model_predictions=False)
        except Exception as e:
            return SafeJSONResponse(content={"error": str(e)}, status_code=500)

        # Get context points
        rc = raw_cfg["robust_optimization"]
        ctx_groups = rc.get("context_groups", [])
        ctx_params = []
        for p in raw_cfg.get("experiment", {}).get("parameters", []):
            p_groups = p.get("groups", []) if hasattr(p, "get") else getattr(p, "groups", [])
            if any(g in ctx_groups for g in (p_groups or [])):
                ctx_params.append(p.get("name", "") if hasattr(p, "get") else getattr(p, "name", ""))

        ctx_points = rc.get("context_points")
        if ctx_points is None:
            try:
                from foambo.robustness import resolve_context_points, RobustOptimizationConfig
                from foambo.orchestrate import ExperimentOptions
                import logging as _logging
                _rob_log = _logging.getLogger("ax.foambo.robustness")
                _old = _rob_log.level
                _rob_log.setLevel(_logging.WARNING)
                try:
                    robust_cfg = RobustOptimizationConfig.model_validate(dict(rc))
                    exp_cfg = ExperimentOptions.model_validate(dict(raw_cfg["experiment"]))
                    resolve_context_points(robust_cfg, exp_cfg)
                    ctx_points = robust_cfg.context_points
                finally:
                    _rob_log.setLevel(_old)
            except Exception:
                ctx_points = []
        if not ctx_points:
            return SafeJSONResponse(content={"points": []})

        alpha = max(rc.get("robustness", 0.5), 0.05)

        # For each Pareto point, predict at each context scenario
        from ax.core.observation import ObservationFeatures
        _disabled = _disable_context_transforms(gs)
        try:
            results = []
            for params, preds, trial_idx, arm_name in front:
                obs_list = []
                for cp in ctx_points:
                    p = dict(params)
                    p.update(cp)
                    obs_list.append(ObservationFeatures(parameters=p))
                means, sems = gs.adapter.predict(obs_list)

                obj_stats = {}
                for metric in means:
                    vals = np.array([float(means[metric][i]) for i in range(len(ctx_points))])
                    sorted_vals = np.sort(vals)
                    n_tail = max(1, int(len(sorted_vals) * (1 - alpha)))
                    cvar = float(np.mean(sorted_vals[:n_tail]))
                    obj_stats[metric] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "cvar": cvar,
                        "values": vals.tolist(),
                    }
                results.append({
                    "trial_index": trial_idx,
                    "arm_name": arm_name,
                    "parameters": params,
                    "observed": {k: v if isinstance(v, (int, float)) else v[0] if isinstance(v, tuple) else v
                                 for k, v in preds.items()},
                    "robustness": obj_stats,
                })
        finally:
            _restore_context_transforms(_disabled)

    return SafeJSONResponse(content={"points": results, "alpha": alpha,
                                     "context_params": ctx_params,
                                     "n_contexts": len(ctx_points)})


@app.get("/api/v1/robust/risk-trace")
def get_risk_trace():
    """CVaR trace over completed trials — shows whether optimization finds
    increasingly robust designs over time."""
    import numpy as np
    raw_cfg = _state.raw_cfg
    if not raw_cfg or not raw_cfg.get("robust_optimization"):
        return SafeJSONResponse(content={"traces": {}})

    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        gs = client._generation_strategy
        if gs is None or gs.adapter is None:
            raise HTTPException(503, "Model not fitted yet")

        rc = raw_cfg["robust_optimization"]
        alpha = max(rc.get("robustness", 0.5), 0.05)

        # Get context points
        ctx_points = rc.get("context_points")
        if ctx_points is None:
            try:
                from foambo.robustness import resolve_context_points, RobustOptimizationConfig
                from foambo.orchestrate import ExperimentOptions
                import logging as _logging
                _rob_log = _logging.getLogger("ax.foambo.robustness")
                _old = _rob_log.level
                _rob_log.setLevel(_logging.WARNING)
                try:
                    robust_cfg = RobustOptimizationConfig.model_validate(dict(rc))
                    exp_cfg = ExperimentOptions.model_validate(dict(raw_cfg["experiment"]))
                    resolve_context_points(robust_cfg, exp_cfg)
                    ctx_points = robust_cfg.context_points
                finally:
                    _rob_log.setLevel(_old)
            except Exception:
                ctx_points = []
        if not ctx_points:
            return SafeJSONResponse(content={"traces": {}})

        from ax.core.observation import ObservationFeatures
        from ax.core.base_trial import TrialStatus as AxTrialStatus
        from ax.core.trial import Trial

        completed = sorted(
            [t for t in client._experiment.trials.values()
             if t.status == AxTrialStatus.COMPLETED and isinstance(t, Trial)],
            key=lambda t: t.index)

        _disabled = _disable_context_transforms(gs)
        try:
            traces = {}  # metric → [cvar_per_trial]
            trial_indices = []
            for trial in completed:
                arm = trial.arm
                if arm is None:
                    continue
                params = arm.parameters
                obs_list = []
                for cp in ctx_points:
                    p = dict(params)
                    p.update(cp)
                    obs_list.append(ObservationFeatures(parameters=p))
                try:
                    means, _ = gs.adapter.predict(obs_list)
                except Exception:
                    continue
                trial_indices.append(trial.index)
                for metric in means:
                    vals = np.array([float(means[metric][i]) for i in range(len(ctx_points))])
                    sorted_vals = np.sort(vals)
                    n_tail = max(1, int(len(sorted_vals) * (1 - alpha)))
                    cvar = float(np.mean(sorted_vals[:n_tail]))
                    traces.setdefault(metric, []).append(cvar)
        finally:
            _restore_context_transforms(_disabled)

    return SafeJSONResponse(content={
        "traces": traces,
        "trial_indices": trial_indices,
        "alpha": alpha,
    })


@app.get("/api/v1/robust/context-sensitivity")
def get_context_sensitivity():
    """Heatmap data: for each objective × context param, how much does
    the objective vary when sweeping that context param?

    Uses the best Pareto point (or best SOO point) as the base design.
    """
    import numpy as np
    raw_cfg = _state.raw_cfg
    if not raw_cfg or not raw_cfg.get("robust_optimization"):
        return SafeJSONResponse(content={"matrix": {}})

    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        gs = client._generation_strategy
        if gs is None or gs.adapter is None:
            raise HTTPException(503, "Model not fitted yet")

        rc = raw_cfg["robust_optimization"]
        ctx_groups = rc.get("context_groups", [])
        ctx_params = []
        for p in raw_cfg.get("experiment", {}).get("parameters", []):
            p_groups = p.get("groups", []) if hasattr(p, "get") else getattr(p, "groups", [])
            if any(g in ctx_groups for g in (p_groups or [])):
                ctx_params.append(p.get("name", "") if hasattr(p, "get") else getattr(p, "name", ""))

        if not ctx_params:
            return SafeJSONResponse(content={"matrix": {}})

        # Get base design (best point or first Pareto point)
        exp = client._experiment
        from ax.core.objective import MultiObjective
        try:
            if isinstance(exp.optimization_config.objective, MultiObjective):
                front = client.get_pareto_frontier(use_model_predictions=False)
                base_params = front[0][0] if front else None
            else:
                base_params, _, _, _ = client.get_best_parameterization(
                    use_model_predictions=False)
        except Exception:
            base_params = None
        if base_params is None:
            return SafeJSONResponse(content={"matrix": {}})

        from ax.core.observation import ObservationFeatures
        from ax.core.parameter import ChoiceParameter

        # Only include objective metrics (not tracking metrics)
        obj_names = set()
        from ax.core.objective import MultiObjective
        opt_config = exp.optimization_config
        if isinstance(opt_config.objective, MultiObjective):
            obj_names = {o.metric.name for o in opt_config.objective.objectives}
        else:
            obj_names = {opt_config.objective.metric.name}

        n_sweep = 20
        _disabled = _disable_context_transforms(gs)
        try:
            matrix = {}  # metric → {ctx_param → variation}
            for cp_name in ctx_params:
                param = exp.search_space.parameters.get(cp_name)
                if param is None or isinstance(param, ChoiceParameter):
                    continue
                lo, hi = param.lower, param.upper
                xs = np.linspace(lo, hi, n_sweep).tolist()
                obs_list = []
                for xval in xs:
                    p = dict(base_params)
                    p[cp_name] = xval
                    obs_list.append(ObservationFeatures(parameters=p))
                means, _ = gs.adapter.predict(obs_list)
                for metric in means:
                    if obj_names and metric not in obj_names:
                        continue
                    vals = np.array([float(means[metric][i]) for i in range(len(means[metric]))])
                    variation = float(np.std(vals))
                    matrix.setdefault(metric, {})[cp_name] = {
                        "std": variation,
                        "range": float(np.max(vals) - np.min(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "mean": float(np.mean(vals)),
                    }
        finally:
            _restore_context_transforms(_disabled)

    return SafeJSONResponse(content={
        "matrix": matrix,
        "context_params": ctx_params,
        "base_parameters": base_params,
    })


@app.get("/api/v1/convergence")
def get_convergence():
    """Part A: Per-objective max Probability of Improvement for convergence tracking."""
    with _state.lock:
        client = _state.client
        if client is None:
            return SafeJSONResponse(content={"estimates": {}, "error": "Not initialized"})
        gs = client._generation_strategy
        if gs is None or gs.adapter is None:
            return SafeJSONResponse(content={"estimates": {}, "error": "No fitted model"})
        # Get improvement_bar from config
        orch_cfg = _state.orch_cfg
        improvement_bar = 0.1
        if orch_cfg and hasattr(orch_cfg, 'global_stopping_strategy'):
            gss = orch_cfg.global_stopping_strategy
            if gss and hasattr(gss, 'improvement_bar'):
                improvement_bar = gss.improvement_bar
            elif isinstance(gss, dict):
                improvement_bar = gss.get('improvement_bar', 0.1)
        try:
            from .analysis import compute_convergence_pi
            result = compute_convergence_pi(client, gs, improvement_bar=improvement_bar)

            # Build improvement trace from best-so-far data (retrospective)
            # This shows how the normalized improvement rate decayed over trials
            import numpy as np
            from ax.core.objective import MultiObjective as _MO
            exp = client._experiment
            oc = exp.optimization_config
            obj_info = ({o.metric.name: o.minimize for o in oc.objective.objectives}
                        if isinstance(oc.objective, _MO)
                        else {oc.objective.metric.name: oc.objective.minimize})
            data = exp.lookup_data().df
            completed = sorted(
                [t for t in exp.trials.values() if t.status.name == "COMPLETED"],
                key=lambda t: t.index)
            improvement_traces = {}
            for metric, minimize in obj_info.items():
                best = None
                trace = []
                for trial in completed:
                    tdf = data[(data["trial_index"] == trial.index) & (data["metric_name"] == metric)]
                    if tdf.empty:
                        continue
                    val = float(tdf["mean"].iloc[-1])
                    if best is None:
                        best = val
                    elif (minimize and val < best) or (not minimize and val > best):
                        best = val
                    # Normalized improvement: how much room is left
                    # Use IQR as scale
                    all_vals = data[data["metric_name"] == metric]["mean"].dropna().values
                    if len(all_vals) > 2:
                        q75, q25 = np.percentile(all_vals, [75, 25])
                        iqr = max(q75 - q25, 1e-12)
                        trace.append(abs(val - best) / iqr)
                    else:
                        trace.append(1.0)
                if trace:
                    improvement_traces[metric] = trace

            result["improvement_trace"] = improvement_traces
            return SafeJSONResponse(content=result)
        except Exception as e:
            return SafeJSONResponse(content={"estimates": {}, "error": str(e)}, status_code=500)


@app.get("/api/v1/specialization-cost")
def get_specialization_cost():
    """Part B: Per-objective PI at fixed context (no robustness)."""
    # Return cached result if available (computed at startup or previous call)
    if hasattr(_state, '_specialization_cache') and _state._specialization_cache:
        return SafeJSONResponse(content=_state._specialization_cache)

    # Heavy GP simulation — only run in --no-opt dashboard mode.
    # During an active optimization, this would compete with trial scheduling.
    if not getattr(_state, 'no_opt', False):
        return SafeJSONResponse(content={
            "estimates": {},
            "error": "Specialization cost is only computed in --no-opt mode",
        })

    import numpy as np
    raw_cfg = _state.raw_cfg
    if not raw_cfg or not raw_cfg.get("robust_optimization"):
        return SafeJSONResponse(content={"estimates": {}, "error": "Not in robust mode"})

    with _state.lock:
        client = _state.client
        if client is None:
            return SafeJSONResponse(content={"estimates": {}, "error": "Not initialized"})
        gs = client._generation_strategy
        if gs is None or gs.adapter is None:
            return SafeJSONResponse(content={"estimates": {}, "error": "No fitted model"})

        # Compute nominal context (mean of context points)
        rc = raw_cfg["robust_optimization"]
        ctx_groups = rc.get("context_groups", [])
        ctx_param_names = []
        for p in raw_cfg.get("experiment", {}).get("parameters", []):
            p_groups = p.get("groups", []) if hasattr(p, "get") else getattr(p, "groups", [])
            if any(g in ctx_groups for g in (p_groups or [])):
                ctx_param_names.append(p.get("name", "") if hasattr(p, "get") else getattr(p, "name", ""))

        ctx_points = rc.get("context_points")
        if ctx_points is None:
            try:
                from foambo.robustness import resolve_context_points, RobustOptimizationConfig
                from foambo.orchestrate import ExperimentOptions
                robust_cfg = RobustOptimizationConfig.model_validate(dict(rc))
                exp_cfg = ExperimentOptions.model_validate(dict(raw_cfg["experiment"]))
                resolve_context_points(robust_cfg, exp_cfg)
                ctx_points = robust_cfg.context_points
            except Exception:
                ctx_points = []

        if not ctx_points or not ctx_param_names:
            return SafeJSONResponse(content={"estimates": {}, "error": "No context points"})

        nominal_ctx = {cn: float(np.mean([cp[cn] for cp in ctx_points])) for cn in ctx_param_names}

        orch_cfg = _state.orch_cfg
        improvement_bar = 0.1
        if orch_cfg and hasattr(orch_cfg, 'global_stopping_strategy'):
            gss = orch_cfg.global_stopping_strategy
            if gss and hasattr(gss, 'improvement_bar'):
                improvement_bar = gss.improvement_bar
            elif isinstance(gss, dict):
                improvement_bar = gss.get('improvement_bar', 0.1)

        try:
            from .analysis import compute_specialization_cost
            # Compute for each context point + nominal
            all_ctx = [nominal_ctx] + list(ctx_points)
            all_labels = ["nominal (mean)"] + [
                ", ".join(f"{k}={v:.4g}" for k, v in cp.items()) for cp in ctx_points
            ]
            per_context = []
            for ctx, label in zip(all_ctx, all_labels):
                result = compute_specialization_cost(
                    client, gs, context_point=ctx, improvement_bar=improvement_bar,
                )
                per_context.append({
                    "context_point": ctx,
                    "label": label,
                    "estimates": result.get("estimates", {}),
                })
            # Build summary: worst-case (max trials) across all context points per objective
            summary = {}
            all_metrics = set()
            for pc in per_context:
                all_metrics.update(pc["estimates"].keys())
            for metric in all_metrics:
                max_trials = 0
                max_pi = 0.0
                worst_ctx = ""
                for pc in per_context:
                    est = pc["estimates"].get(metric, {})
                    t = est.get("estimated_specialized_trials") or 0
                    pi = est.get("max_pi_specialized") or 0.0
                    if t > max_trials:
                        max_trials = t
                        max_pi = pi
                        worst_ctx = pc["label"]
                summary[metric] = {
                    "max_trials": max_trials,
                    "max_pi": round(max_pi, 4),
                    "worst_context": worst_ctx,
                }
            result = {
                "summary": summary,
                "per_context": per_context,
                "pi_threshold": improvement_bar,
                "context_params": ctx_param_names,
            }
            _state._specialization_cache = result
            return SafeJSONResponse(content=result)
        except Exception as e:
            return SafeJSONResponse(content={"estimates": {}, "error": str(e)}, status_code=500)


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
    cls_name = getattr(analysis_cls, '__name__', str(analysis_cls))
    log.info("Computing analysis: %s(%s)", cls_name,
             ', '.join(f'{k}={v!r}' for k, v in kwargs.items()) if kwargs else '')
    with _state.lock:
        client = _state.client
        if client is None:
            raise HTTPException(503, "Optimizer not initialized")
        _ensure_model_fitted()

        import logging
        ax_logger = logging.getLogger("ax")
        cv_logger = logging.getLogger("ax.adapter.cross_validation")
        prev_level = ax_logger.level
        prev_cv_level = cv_logger.level
        ax_logger.setLevel(logging.CRITICAL)
        cv_logger.setLevel(logging.CRITICAL)
        # Disable SubstituteContextFeatures for Ax analyses — they do dense
        # grid predictions (contour, CV) that blow up with n_w expansion.
        # Our custom robust endpoints handle context separately.
        gs = client._generation_strategy
        _disabled = _disable_context_transforms(gs)
        try:
            cards = client.compute_analyses(
                analyses=[analysis_cls(**kwargs)],
                display=False,
            )
        finally:
            _restore_context_transforms(_disabled)
            ax_logger.setLevel(prev_level)
            cv_logger.setLevel(prev_cv_level)
        log.info("Analysis %s completed (%d cards)", cls_name, len(cards) if cards else 0)

        from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
        from ax.core.analysis_card import AnalysisCardGroup
        results = []
        for card in cards:
            if isinstance(card, AnalysisCardGroup):
                for sub in card.flatten():
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

        import logging, warnings
        ax_loggers = [logging.getLogger(n) for n in ("ax", "ax.analysis", "ax.analysis.analysis")]
        prev_levels = {l.name: l.level for l in ax_loggers}
        for l in ax_loggers:
            l.setLevel(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cards = client.compute_analyses(
                    analyses=[analysis_cls(**kwargs)],
                    display=False,
                )
        finally:
            for l in ax_loggers:
                l.setLevel(prev_levels.get(l.name, logging.WARNING))

        from ax.core.analysis_card import AnalysisCardGroup
        results = []
        for card in cards:
            sub_cards = card.flatten() if isinstance(card, AnalysisCardGroup) else [card]
            for sub in sub_cards:
                level = str(getattr(sub, "level", "INFO"))
                if "ERROR" in level:
                    log.debug(f"Skipping failed card: {sub.title}")
                    continue
                blob = sub.blob
                # Skip cards with non-meaningful content (raw JSON, status codes)
                if not blob or len(blob.strip()) < 10:
                    continue
                try:
                    import json as _j
                    parsed = _j.loads(blob)
                    if isinstance(parsed, dict) and "status" in parsed:
                        continue  # skip raw status blobs
                    # Plotly figure JSON → return as figure, not raw blob
                    if isinstance(parsed, dict) and "data" in parsed:
                        results.append({"title": sub.title, "subtitle": sub.subtitle,
                                        "figure": parsed, "level": level})
                        continue
                except (ValueError, TypeError):
                    pass  # not JSON, keep it
                results.append({"title": sub.title, "subtitle": sub.subtitle,
                                "blob": blob, "level": level})
        return {"cards": results}


@app.post("/api/v1/analysis/sensitivity")
def post_analysis_sensitivity(request_body: dict):
    """Sobol sensitivity analysis via Ax.

    In robust mode, context parameter bars are colored differently from
    design parameter bars so users can see at a glance which sensitivity
    comes from environmental variation vs design choices.
    """
    metric = request_body.get("metric")
    top_k = request_body.get("top_k", 10)
    try:
        from ax.analysis.plotly.sensitivity import SensitivityAnalysisPlot
        result = _compute_ax_analysis(
            SensitivityAnalysisPlot, metric_name=metric, top_k=top_k)

        # Annotate context params in robust mode
        raw_cfg = _state.raw_cfg
        if raw_cfg and raw_cfg.get("robust_optimization"):
            ctx_groups = raw_cfg["robust_optimization"].get("context_groups", [])
            ctx_params = set()
            for p in raw_cfg.get("experiment", {}).get("parameters", []):
                p_groups = p.get("groups", []) if hasattr(p, "get") else getattr(p, "groups", [])
                if any(g in ctx_groups for g in (p_groups or [])):
                    ctx_params.add(p.get("name", "") if hasattr(p, "get") else getattr(p, "name", ""))
            if ctx_params:
                result = _annotate_sensitivity_context(result, ctx_params)

        return SafeJSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        return SafeJSONResponse(content={"error": str(e)}, status_code=500)


def _annotate_sensitivity_context(result: dict, ctx_params: set) -> dict:
    """Color context parameter bars in the sensitivity Plotly figure."""
    for card in result.get("cards", []):
        fig = card.get("figure")
        if not fig or "data" not in fig:
            continue
        for trace in fig["data"]:
            if trace.get("type") not in ("bar",):
                continue
            labels = trace.get("y") or trace.get("x") or []
            colors = []
            for label in labels:
                name = label.split(" ")[-1] if isinstance(label, str) else str(label)
                if name in ctx_params:
                    colors.append("rgba(234, 88, 12, 0.8)")  # orange for context
                else:
                    colors.append("rgba(59, 130, 246, 0.8)")  # blue for design
            if colors:
                trace.setdefault("marker", {})["color"] = colors
        # Add legend annotation
        fig.setdefault("layout", {}).setdefault("annotations", []).extend([
            {"text": "<b style='color:rgba(59,130,246,0.8)'>\u25a0</b> Design",
             "xref": "paper", "yref": "paper", "x": 1.0, "y": 1.05,
             "showarrow": False, "font": {"size": 11}},
            {"text": "<b style='color:rgba(234,88,12,0.8)'>\u25a0</b> Context",
             "xref": "paper", "yref": "paper", "x": 1.0, "y": 1.12,
             "showarrow": False, "font": {"size": 11}},
        ])
    return result


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
        top_k = min(3, len(params) * (len(params) - 1) // 2)
        from ax.analysis.plotly.top_surfaces import TopSurfacesAnalysis
        result = _compute_ax_analysis(TopSurfacesAnalysis, metric_name=metric, top_k=top_k)
        return SafeJSONResponse(content=result)
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
            log.debug(f"{cls_path[1]} failed (skipped): {e}")
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

        return SafeJSONResponse(content={
            "groups": group_scores, "metric": metric,
        })
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


# Push endpoints (event-driven orchestration)

@app.post("/api/v1/trials/{trial_index}/push/status")
def push_trial_status(trial_index: int, request_body: dict):
    """Trial pushes completion/failure status."""
    status = request_body.get("status", "")
    exit_code = request_body.get("exit_code")
    session = request_body.get("session_id", "")

    # Validate session to detect rogue jobs from crashed runs
    if session and session != _state._session_id:
        log.warning(f"Trial {trial_index} push from stale session {session} (current: {_state._session_id})")
        return SafeJSONResponse(content={"error": "stale session", "expected": _state._session_id}, status_code=409)

    if exit_code is not None:
        status = "completed" if exit_code == 0 else "failed"

    with _state.lock:
        _state.trial_status_overrides[trial_index] = status
        _state.trial_heartbeats[trial_index] = time.time()
        _state.log_event(trial_index, f"trial.{status}",
                         request_body.get("message", ""))
    _state.bump("trials")
    return SafeJSONResponse(content={"ok": True})


@app.post("/api/v1/trials/{trial_index}/push/metrics")
def push_trial_metrics(trial_index: int, request_body: dict):
    """Trial pushes streaming metric values."""
    metrics = request_body.get("metrics", {})
    step = request_body.get("step")
    session = request_body.get("session_id", "")

    if session and session != _state._session_id:
        return SafeJSONResponse(content={"error": "stale session"}, status_code=409)

    with _state.lock:
        _state.trial_pushed_metrics.setdefault(trial_index, []).append({
            "metrics": metrics, "step": step, "ts": time.time(),
        })
        _state.trial_heartbeats[trial_index] = time.time()
        for mn, val in metrics.items():
            _state.log_event(trial_index, "trial.streaming",
                             f"{mn} step={step} value={val}")
    _state.bump("streaming", "trials")
    return SafeJSONResponse(content={"ok": True})


@app.post("/api/v1/trials/{trial_index}/push/heartbeat")
def push_trial_heartbeat(trial_index: int):
    """Trial signals it's alive."""
    with _state.lock:
        _state.trial_heartbeats[trial_index] = time.time()
    return SafeJSONResponse(content={"ok": True})


@app.get("/api/v1/events")
def get_events(if_none_match: Optional[str] = Header(None)):
    """Return the timestamped event log."""
    with _state.lock:
        return SafeJSONResponse(content={"events": _state.event_log[-500:]})


# Store actual port on _state for event-driven runner API endpoint injection

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
        # Serve cached screenshot if it already exists
        if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
            log.info(f"Serving cached screenshot for {case_path}")
            with open(screenshot_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            return {"image": img_data, "case_path": case_path, "renderer": "paraview"}
        # Sandboxed environment: only expose case_path and essential ParaView vars
        safe_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "MESA_GL_VERSION_OVERRIDE": os.environ.get("MESA_GL_VERSION_OVERRIDE", ""),
            "TMPDIR": case_path,
        }
        # Preserve ParaView/VTK/Python/foamBO-related env vars
        for k, v in os.environ.items():
            if k.startswith(("PV_", "VTK_", "PARAVIEW_", "LD_LIBRARY", "PYTHONPATH", "PYTHON", "FOAMBO_")):
                safe_env[k] = v

        # Provide a display for X11-backed ParaView builds (vtkXOpenGLRenderWindow).
        # Prefer Xvfb (no auth issues, isolated). Fall back to host DISPLAY + XAUTHORITY.
        xvfb_proc = None
        xvfb = shutil.which("Xvfb")
        if xvfb:
            import random
            disp_num = random.randint(50, 200)
            disp_str = f":{disp_num}"
            try:
                xvfb_proc = subprocess.Popen(
                    [xvfb, disp_str, "-screen", "0", "1280x1024x24", "-ac", "-nolisten", "tcp"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                import time as _time; _time.sleep(0.5)
                safe_env["DISPLAY"] = disp_str
                log.info(f"Started Xvfb on display {disp_str} (pid {xvfb_proc.pid})")
            except Exception as e:
                log.warning(f"Failed to start Xvfb: {e}")
                xvfb_proc = None
        if "DISPLAY" not in safe_env:
            host_display = os.environ.get("DISPLAY", "")
            if host_display:
                safe_env["DISPLAY"] = host_display
                xauth = os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
                if os.path.exists(xauth):
                    safe_env["XAUTHORITY"] = xauth
                log.info(f"Using host display {host_display}")
            else:
                log.warning("No Xvfb and no DISPLAY — pvpython may fail on X11 builds")

        cmd = [pvpython, "--force-offscreen-rendering", script_path, case_path, screenshot_name]
        log.info(f"Running: {' '.join(cmd)}")
        bwrap = shutil.which("bwrap")
        if bwrap:
            cmd = [
                bwrap,
                "--ro-bind", "/", "/",              # read-only root filesystem
                "--dev", "/dev",                    # expose /dev (needed for /dev/urandom, DRI)
                "--bind", case_path, case_path,     # read-write case folder
                "--bind", "/tmp", "/tmp",           # ParaView needs tmp
                "--unshare-net",                    # no network access
                "--die-with-parent",                # kill if server dies
                pvpython, "--force-offscreen-rendering", script_path, case_path, screenshot_name,
            ]
            log.info(f"Running (sandboxed): {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=120,
                cwd=case_path,
                env=safe_env,
            )
        finally:
            if xvfb_proc is not None:
                xvfb_proc.terminate()
                log.info(f"Terminated Xvfb (pid {xvfb_proc.pid})")
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
                     port: int = 8098, no_opt: bool = False) -> threading.Thread | None:
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
    _state.no_opt = no_opt
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
    _state._actual_port = actual_port
    _state._actual_host = host

    thread = threading.Thread(target=server.run, daemon=True, name="foambo-api")
    thread.start()
    log.info(f"=================================================")
    log.info(f"Dashboard: http://{host}:{actual_port}/  API docs: http://{host}:{actual_port}/api/docs")
    log.info(f"=================================================")
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
