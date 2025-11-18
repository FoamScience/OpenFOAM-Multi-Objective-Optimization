#!/usr/bin/env python3

""" Web UI to visualize and interact with an Ax experiment state.

Capabilities:
- Load client + experiment state from the provided config.
- Fetch Pareto frontier and pick the most interesting point for
  a chosen objective.
- Run a trial using selected parameters (manual generation node).
- Sensitivity exploration via model predictions, when available.
"""

from __future__ import annotations

import webbrowser
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .common import *
from .orchestrate import (
    StoreOptions,
    ConfigOrchestratorOptions,
    OptimizationOptions,
    ManualGenerationNode,
)
from .analysis import plot_pareto_frontier

from ax.api.client import Client, Orchestrator
from ax.generation_strategy.generation_strategy import GenerationStrategy

from logging import Logger
from ax.utils.common.logger import get_logger
log: Logger = get_logger(__name__)


class AppState:
    """App-level state"""
    cfg: Optional[DictConfig] = None
    client: Optional[Client] = None
    orch_opts: Optional[ConfigOrchestratorOptions] = None
    opt_opts: Optional[OptimizationOptions] = None
    viz_settings: dict = None


state = AppState()
state.viz_settings = {
    'time_step': 'latest',
    'field': 'auto',
    'mesh_parts': 'internal',
    'camera_angle': 'isometric',
    'decompose_polyhedra': True,
    'cell_to_point': True,
    'skip_zero_time': True
}
app = FastAPI(title="FoamBO Optimization Visualizer")


@app.on_event("startup")
async def _auto_open_browser_on_startup():
    if getattr(state, "open_browser", False):
        import threading
        import time
        url = f"http://{getattr(state, 'host', '127.0.0.1')
                        }:{getattr(state, 'port', 8099)}"

        def _open_later():
            time.sleep(0.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open_later, daemon=True).start()


class RunTrialRequest(BaseModel):
    parameters: Dict[str, Any]


class SensitivityRequest(BaseModel):
    base_parameters: Dict[str, Any]
    variations: Dict[str, float]


def ensure_loaded():
    if not state.cfg:
        raise HTTPException(
            status_code=400, detail="Config not loaded. Start UI with visualizer_ui(cfg)")
    if not state.client:
        raise HTTPException(
            status_code=400, detail="Client not loaded from store config")


def get_objective_minimize_map(client: Client) -> Dict[str, bool]:
    oc = client._experiment.optimization_config
    name_to_min = {}
    if hasattr(oc, "objective") and hasattr(oc.objective, "objectives"):
        for obj in oc.objective.objectives:
            for m in obj.metric_names:
                name_to_min[m] = obj.minimize
    else:
        name_to_min[oc.objective.metric.name] = oc.objective.minimize
    return name_to_min


def pick_interesting_point(front, objective: str, minimize: bool) -> Dict[str, Any]:
    """
    front is assumed to be iterable of (parameters, predictions, arm, model_predictions)
    Select point that:
    1. Has the best value according to objective direction (minimize/maximize)
    2. Among points with similar values, prefer the one with smallest SEM (uncertainty)
    """

    best = None
    best_val = float("inf") if minimize else -float("inf")
    best_sem = float("inf")
    for params, preds, _, _ in front:
        if objective not in preds:
            continue
        mean, sem = preds[objective][0], preds[objective][1]
        is_better_value = (minimize and mean < best_val) or (
            not minimize and mean > best_val)
        value_tolerance = abs(best_val * 0.01) if best_val != 0 else 0.01
        is_similar_value = abs(mean - best_val) <= value_tolerance
        is_lower_uncertainty = sem < best_sem
        if is_better_value or (is_similar_value and is_lower_uncertainty):
            best_val = mean
            best_sem = sem
            best = params
    return best or (front[0][0] if front else {})


def ui_index_html() -> str:
    """Load the visualizer HTML template."""
    try:
        from importlib.resources import files
        template_path = files('foambo').joinpath('templates/visualizer.html')
        return template_path.read_text(encoding='utf-8')
    except (ImportError, AttributeError):
        from pkg_resources import resource_string
        return resource_string('foambo', 'templates/visualizer.html').decode('utf-8')


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=ui_index_html(), status_code=200)


@app.get("/api/experiment")
def api_experiment():
    ensure_loaded()
    exp = state.client._experiment
    params = []
    for name, p in exp.parameters.items():
        if hasattr(p, "values") and p.values is not None:
            params.append({"name": name, "type": "choice",
                          "values": list(p.values)})
        else:
            bounds = getattr(p, "bounds", None) or [getattr(
                p, "lower", None), getattr(p, "upper", None)]
            params.append({"name": name, "type": "range",
                          "bounds": bounds, "step": getattr(p, "step", None)})
    objectives_with_direction = []
    oc = exp.optimization_config
    if hasattr(oc, "objective") and hasattr(oc.objective, "objectives"):
        for obj in oc.objective.objectives:
            for metric_name in obj.metric_names:
                direction = "Minimize" if obj.minimize else "Maximize"
                objectives_with_direction.append(
                    {"name": metric_name, "direction": direction})
    else:
        metric_name = oc.objective.metric.name
        direction = "Minimize" if oc.objective.minimize else "Maximize"
        objectives_with_direction.append(
            {"name": metric_name, "direction": direction})
    if not objectives_with_direction:
        try:
            for metric in exp.metrics:
                objectives_with_direction.append(
                    {"name": metric, "direction": "Unknown"})
        except Exception:
            pass
    seen = set()
    uniq_objectives = []
    for obj in objectives_with_direction:
        if obj["name"] not in seen:
            seen.add(obj["name"])
            uniq_objectives.append(obj)
    max_trials = None
    if state.opt_opts and hasattr(state.opt_opts, 'max_trials'):
        max_trials = state.opt_opts.max_trials
    return {
        "name": exp.name,
        "trial_count": len(exp.trials),
        "max_trials": max_trials,
        "parameters": params,
        "metrics": list(exp.metrics),
        "objectives": uniq_objectives,
    }


@app.get("/api/pareto")
def api_pareto(objective: str = Query(..., description="Objective metric name to focus on")):
    ensure_loaded()
    try:
        front = state.client.get_pareto_frontier(use_model_predictions=True)
        if len(front) == 0:
            front = state.client.get_pareto_frontier(use_model_predictions=False)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        log.error(f"Failed to compute Pareto frontier:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Failed to compute Pareto frontier: {
                            e}\n\nSee server logs for full traceback.")

    if not front or len(front) == 0:
        raise HTTPException(
            status_code=404,
            detail="No Pareto frontier points found. The model may not be fitted yet, or there are no completed trials with predictions. Try running 'Fit data to model' first."
        )

    name_to_min = get_objective_minimize_map(state.client)
    minimize = name_to_min.get(objective, True)
    params = pick_interesting_point(front, objective, minimize)

    if not params or len(params) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No valid parameters found for objective '{objective}'. The Pareto frontier may not have predictions for this objective. Try a different objective or run more trials."
        )

    log.info(f"Selected Pareto point for objective '{objective}': {params}")
    return {"objective": objective, "minimize": minimize, "parameters": params}


# Track running trial status
trial_status = {}


@app.post("/api/run_trial")
def api_run_trial(req: RunTrialRequest):
    ensure_loaded()
    import hashlib
    import json
    param_str = json.dumps(req.parameters, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()

    for idx in state.client._experiment.trials:
        trial = state.client._experiment.trials[idx]
        if trial.arm and trial.arm.parameters:
            trial_param_str = json.dumps(trial.arm.parameters, sort_keys=True)
            trial_param_hash = hashlib.md5(
                trial_param_str.encode()).hexdigest()

            if param_hash == trial_param_hash:
                trial_status[idx] = trial.status.name if hasattr(
                    trial, 'status') else 'existing'
                return {"status": "existing", "trial_index": idx, "message": f"Trial {idx} already exists with these parameters"}

    trial_index = 0 if len(state.client._experiment._trials) == 0 else max(
        state.client._experiment._trials.keys()) + 1
    state.orch_opts.max_trials += 1
    state.client.configure_runner(**state.opt_opts.to_runner_dict())
    scheduler = Orchestrator(
        experiment=state.client._experiment,
        generation_strategy=GenerationStrategy(
            name=f"manual_pareto",
            nodes=[ManualGenerationNode(
                node_name="manual_pareto", parameters=req.parameters)],
        ),
        options=state.orch_opts.to_scheduler_options(),
        db_settings=None,
    )

    try:
        import threading
        trial_status[trial_index] = "running"

        def _run():
            try:
                scheduler.run_n_trials(
                    max_trials=1, timeout_hours=state.orch_opts.timeout_hours)
                trial_status[trial_index] = "completed"
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                log.error(f"Trial {trial_index} failed:\n{error_trace}")
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                trial_status[trial_index] = f"failed: {error_msg}"
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"status": "queued", "trial_index": trial_index}
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        log.error(f"Failed to start trial {trial_index}:\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Failed to start trial {
                            trial_index}: {str(e)}")


@app.get("/api/trial_status/{trial_index}")
def api_trial_status(trial_index: int):
    """Get the current status of a running trial."""
    if trial_index in trial_status:
        return {"trial_index": trial_index, "status": trial_status[trial_index]}
    else:
        ensure_loaded()
        exp = state.client._experiment
        if trial_index in exp.trials:
            trial = exp.trials[trial_index]
            return {"trial_index": trial_index, "status": trial.status.name if hasattr(trial, 'status') else 'Unknown'}
        return {"trial_index": trial_index, "status": "not_found"}


@app.post("/api/fit_model")
def api_fit_model():
    ensure_loaded()
    try:
        _ = state.client.get_next_trials(max_trials=1)
        return {"status": "fit step executed"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fit model: {e}")


@app.post("/api/sensitivity")
def api_sensitivity(req: SensitivityRequest):
    ensure_loaded()
    log.info(f"Sensitivity request received: base_parameters={req.base_parameters}, variations={req.variations}")
    try:
        predictions = state.client.predict([req.base_parameters])
        if not predictions or len(predictions) == 0:
            raise RuntimeError("No predictions returned from model")
        pred_dict = predictions[0]
        means_dict = {metric: float(values[0])
                      for metric, values in pred_dict.items()}
        sems_dict = {metric: float(values[1])
                     for metric, values in pred_dict.items()}

        response = {"predicted_means": means_dict, "predicted_sems": sems_dict}

        # Check if custom visualization callback is configured
        if state.cfg.get('visualizer', {}).get('sensitivity_callback'):
            callback_path = state.cfg['visualizer']['sensitivity_callback']
            try:
                import importlib
                import importlib.util
                import sys
                import os

                # Add current working directory to sys.path to allow imports from experiment directory
                cwd = os.getcwd()
                if cwd not in sys.path:
                    sys.path.insert(0, cwd)

                # Split module path and function name
                module_path, func_name = callback_path.rsplit('.', 1)

                try:
                    module = importlib.import_module(module_path)
                except ModuleNotFoundError:
                    file_path = os.path.join(cwd, *module_path.split('.')) + '.py'
                    if not os.path.exists(file_path):
                        raise ModuleNotFoundError(f"Cannot find module '{module_path}' or file '{file_path}'")
                    spec = importlib.util.spec_from_file_location(module_path, file_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_path] = module
                    spec.loader.exec_module(module)

                callback_func = getattr(module, func_name)

                # Call the callback with the parameters
                image_base64 = callback_func(req.base_parameters)

                # Validate that it's a string (base64 image)
                if not isinstance(image_base64, str):
                    raise TypeError(f"Callback must return a base64-encoded image string, got {type(image_base64)}")

                # Include the base64 image in response
                response["visualization"] = image_base64
                log.info(f"Successfully generated custom visualization using {callback_path}")

            except Exception as viz_error:
                import traceback
                viz_trace = traceback.format_exc()
                log.warning(f"Custom visualization failed, but predictions succeeded:\n{viz_trace}")
                response["visualization_error"] = f"Visualization failed: {str(viz_error)}"

        return response

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        log.error(f"Sensitivity analysis failed:\n{error_trace}")
        error_msg = str(e)
        if "UnsupportedError" in type(e).__name__ or "not predictive" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail="Model not ready for predictions. The optimization may still be in exploration phase. Try running more trials."
            )
        elif "adapter" in error_msg.lower() or "AssertionError" in type(e).__name__:
            raise HTTPException(
                status_code=503,
                detail="Generation strategy adapter not available. The current node may not support predictions yet."
            )
        else:
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {error_msg}\n\nSee server logs for full traceback.")


@app.get("/api/pareto_html")
def api_pareto_html():
    ensure_loaded()
    try:
        _ = plot_pareto_frontier(cfg=state.cfg, client=state.client, open_html=True)
        return {"status": "opened"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to render Pareto HTML: {e}")


@app.get("/api/insights")
def api_insights():
    """Generate experiment insights using Ax InsightsAnalysis."""
    ensure_loaded()
    try:
        from ax.analysis.insights import InsightsAnalysis
        from ax.analysis.diagnostics import DiagnosticAnalysis
        from ax.plot.pareto_frontier import scatter_plot_with_hypervolume_trace_plotly
        import plotly.io as pio

        pio.templates.default = "plotly_dark"
        hv_trace = scatter_plot_with_hypervolume_trace_plotly(
            experiment=state.client._experiment)
        cards = state.client.compute_analyses(
            analyses=[DiagnosticAnalysis(), InsightsAnalysis()])
        if not cards or len(cards) == 0:
            raise HTTPException(
                status_code=500, detail="No insights generated")
        html = hv_trace.to_html(full_html=False, include_plotlyjs='cdn') + \
            cards[0]._to_html(depth=0) + cards[1]._to_html(depth=0)
        if 'plotly' in html.lower() or 'Plotly.newPlot' in html:
            html = f'<div class="insights-plotly-container">{html}</div>'
        return {"html": html}
    except ImportError:
        raise HTTPException(
            status_code=500, detail="InsightsAnalysis not available. Update Ax to latest version.")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        log.error(f"Failed to generate insights:\n{error_trace}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate insights: {str(e)}")


@app.get("/api/trial_list")
def api_trial_list():
    """Get list of all trials with their status."""
    ensure_loaded()
    exp = state.client._experiment
    trials = []
    for idx in exp.trials:
        trial = exp.trials[idx]
        trials.append({
            "index": idx,
            "status": trial.status.name if hasattr(trial, 'status') else 'Unknown',
            "has_case_path": 'case_path' in (trial.run_metadata or {})
        })
    return {"trials": trials}


@app.get("/api/viz_settings")
def api_get_viz_settings():
    """Get current visualization settings."""
    return state.viz_settings


@app.post("/api/viz_settings")
def api_update_viz_settings(settings: dict):
    """Update visualization settings."""
    state.viz_settings.update(settings)
    return {"status": "ok", "settings": state.viz_settings}


@app.get("/api/trial_visualization/{trial_index}")
def api_trial_visualization(trial_index: int):
    """Generate PyVista visualization for an OpenFOAM trial case using server-side settings."""
    time_step = state.viz_settings.get('time_step', 'latest')
    field = state.viz_settings.get('field', 'auto')
    mesh_parts = state.viz_settings.get('mesh_parts', 'internal')
    camera_angle = state.viz_settings.get('camera_angle', 'isometric')
    decompose_polyhedra = state.viz_settings.get('decompose_polyhedra', True)
    cell_to_point = state.viz_settings.get('cell_to_point', True)
    skip_zero_time = state.viz_settings.get('skip_zero_time', False)

    ensure_loaded()
    try:
        import pyvista as pv
        from pathlib import Path
        exp = state.client._experiment
        if trial_index >= len(exp.trials) or trial_index < 0:
            raise HTTPException(status_code=404, detail=f"Trial {
                                trial_index} not found")
        trial = exp.trials[trial_index]
        if not trial.run_metadata or 'case_path' not in trial.run_metadata:
            raise HTTPException(status_code=404, detail=f"Trial {
                                trial_index} has no case_path in run_metadata")
        case_path = Path(trial.run_metadata['case_path'])
        if not case_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Case directory does not exist: {case_path}")
        foam_file = case_path / f"{case_path.name}.foam"
        if not foam_file.exists():
            foam_file.touch()

        reader = pv.OpenFOAMReader(str(foam_file))
        if hasattr(reader, 'decompose_polyhedra'):
            reader.decompose_polyhedra = decompose_polyhedra
        if hasattr(reader, 'cell_to_point_data'):
            reader.cell_to_point_data = cell_to_point
        if hasattr(reader, 'skip_zero_time'):
            reader.skip_zero_time = skip_zero_time
        time_points = reader.time_values if hasattr(
            reader, 'time_values') else []
        if time_step and time_step != "latest":
            try:
                time_idx = int(time_step)
                if 0 <= time_idx < len(time_points):
                    reader.set_active_time_value(time_points[time_idx])
            except (ValueError, IndexError):
                pass

        reader.skip_zero_time = True
        mesh = reader.read()
        if not mesh or mesh.n_blocks == 0:
            raise HTTPException(
                status_code=500, detail="Failed to read mesh from OpenFOAM case")

        mesh_names = ["internal"]
        patch_names = []
        internal_mesh = None
        patches_dict = {}
        if mesh[0] is not None and hasattr(mesh[0], 'n_points') and mesh[0].n_points > 0:
            internal_mesh = mesh[0]

        if mesh.n_blocks > 1 and mesh[1] is not None:
            patches_block = mesh[1]
            if hasattr(patches_block, 'n_blocks'):
                for i in range(patches_block.n_blocks):
                    patch = patches_block[i]
                    if patch is not None and hasattr(patch, 'n_points') and patch.n_points > 0:
                        patch_name = patches_block.get_block_name(i) if hasattr(
                            patches_block, 'get_block_name') else f"patch_{i}"
                        patch_names.append(patch_name)
                        patches_dict[patch_name] = patch
                        mesh_names.append(patch_name)

        requested_parts = [p.strip() for p in mesh_parts.split(',')]
        selected_meshes = []
        for part in requested_parts:
            if part == "internal" and internal_mesh is not None:
                selected_meshes.append(internal_mesh)
            elif part == "all_patches":
                selected_meshes.extend(patches_dict.values())
            elif part in patches_dict:
                selected_meshes.append(patches_dict[part])
        if not selected_meshes and internal_mesh is not None:
            selected_meshes = [internal_mesh]
        if not selected_meshes:
            raise HTTPException(
                status_code=500, detail=f"No valid mesh found for selection: {mesh_parts}")

        available_arrays = list(
            selected_meshes[0].array_names) if selected_meshes[0].n_arrays > 0 else []
        scalar_name = None
        if field and field != "auto" and field in available_arrays:
            scalar_name = field
        elif available_arrays:
            scalar_name = available_arrays[0]
        plotter = pv.Plotter(off_screen=True, window_size=[1400, 1000])
        for i, mesh_obj in enumerate(selected_meshes):
            if scalar_name and scalar_name in mesh_obj.array_names:
                plotter.add_mesh(mesh_obj, scalars=scalar_name,
                                 cmap='viridis', show_edges=False, opacity=0.9)
            else:
                colors = ['lightblue', 'lightgreen',
                          'lightcoral', 'lightyellow', 'lightpink']
                color = colors[i % len(colors)]
                plotter.add_mesh(mesh_obj, color=color,
                                 show_edges=True, opacity=0.8)
        if camera_angle == "isometric":
            plotter.view_isometric()
        elif camera_angle == "xy":
            plotter.view_xy()
        elif camera_angle == "xz":
            plotter.view_xz()
        elif camera_angle == "yz":
            plotter.view_yz()
        elif camera_angle == "x":
            plotter.view_vector((1, 0, 0))
        elif camera_angle == "y":
            plotter.view_vector((0, 1, 0))
        elif camera_angle == "z":
            plotter.view_vector((0, 0, 1))
        else:
            plotter.view_isometric()  # Default
        plotter.add_axes()

        import base64
        from io import BytesIO
        from PIL import Image

        # Take a high-quality screenshot
        img_bytes = plotter.screenshot(
            return_img=True, window_size=[1400, 1000])
        plotter.close()
        img = Image.fromarray(img_bytes)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        time_info = f"Time: {time_points[int(
            time_step)] if time_step and time_step != 'latest' else 'Latest'}" if time_points else "Time: N/A"
        field_info = f"Field: {scalar_name}" if scalar_name else "Field: None"
        mesh_info = f"Mesh: {mesh_parts}" if len(requested_parts) <= 3 else f"Mesh: {
            len(requested_parts)} parts"
        camera_info = f"View: {camera_angle}"
        total_points = sum(m.n_points for m in selected_meshes)
        total_cells = sum(m.n_cells for m in selected_meshes)

        html_str = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background-color: #0f172a;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                }}
                .container {{
                    text-align: center;
                    max-width: 100%;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #334155;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                }}
                .info {{
                    color: #94a3b8;
                    margin-top: 10px;
                    font-family: monospace;
                    font-size: 14px;
                }}
                .settings {{
                    color: #64748b;
                    margin-top: 5px;
                    font-family: monospace;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <img src="data:image/png;base64,{img_base64}" alt="OpenFOAM Visualization">
                <div class="info">Trial {trial_index} - {total_points} points, {total_cells} cells</div>
                <div class="settings">{time_info} | {field_info} | {mesh_info} | {camera_info}</div>
            </div>
        </body>
        </html>
        """

        return {
            "html": html_str,
            "trial_index": trial_index,
            "case_path": str(case_path),
            "n_points": total_points,
            "n_cells": total_cells,
            "arrays": available_arrays,
            "time_points": [str(t) for t in time_points],
            "mesh_names": mesh_names
        }

    except ImportError:
        raise HTTPException(
            status_code=500, detail="PyVista not installed. Install with: pip install pyvista")


def visualizer_ui(cfg: DictConfig, host: str = "127.0.0.1", port: int = 8099, open_browser: bool = True):
    """Launch the FastAPI app and keep process alive.

    The caller (CLI) should pass the same config used for optimization.
    """
    state.cfg = cfg
    set_experiment_name(cfg["experiment"]["name"])
    state.orch_opts = instantiate_with_nested_fields(
        ConfigOrchestratorOptions, cfg["orchestration_settings"])
    state.orch_opts.global_stopping_strategy = None
    state.orch_opts.early_stopping_strategy = None
    state.opt_opts = instantiate_with_nested_fields(
        OptimizationOptions, cfg["optimization"])
    store_cfg = instantiate_with_nested_fields(StoreOptions, cfg["store"])
    state.client = store_cfg.load()

    if not hasattr(state.client, "_experiment"):
        raise RuntimeError(
            "Loaded client has no experiment. Run optimization first or configure experiment.")
    state.host = host
    state.open_browser = open_browser

    import uvicorn
    import socket

    max_retries = 3
    current_port = port
    for attempt in range(max_retries):
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.bind((host, current_port))
            test_socket.close()
            state.port = current_port
            uvicorn.run(app, host=host, port=current_port, log_level="info")
            break
        except OSError as e:
            if attempt < max_retries - 1:
                current_port += 1
            else:
                raise RuntimeError(
                    f"Failed to start server: ports {port} to {
                        current_port} are all in use. "
                    f"Please specify a different port or stop the process using these ports."
                ) from e
