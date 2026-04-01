from ax import Client
from ax.api.protocols.metric import IMetric
from ax.api.protocols.runner import IRunner, TrialStatus
from ax.api.types import TParameterization
from typing import Mapping, Any, Dict, Union, List
from omegaconf import DictConfig, DictKeyType, OmegaConf
from .common import preprocess_case, process_input_command, parse_outcome_for_metric, FoamBOBaseModel
from .common import SLURM_STATUS_MAP, case_preprocessor, CasePreprocessor
import inspect
from pydantic import Field
from foamlib import FoamCase
import subprocess as sb
import numpy as np
import re
import os, time, signal
from ax.storage.json_store.registry import CORE_ENCODER_REGISTRY, CORE_DECODER_REGISTRY, CORE_CLASS_DECODER_REGISTRY
from ax.storage.json_store.encoders import metric_to_dict
from ax.storage.json_store.encoders import runner_to_dict
from ax.storage.metric_registry import register_metrics, CORE_METRIC_REGISTRY
from ax.storage.runner_registry import register_runner
from ax.storage.json_store.encoders import metric_to_dict
from ax.storage.json_store.encoders import runner_to_dict

from logging import Logger
from ax.utils.common.logger import get_logger
log : Logger = get_logger(__name__)

# Subprocess timeout constants (seconds)
METRIC_EVAL_TIMEOUT = 600       # metric command evaluation (blocking, may be heavy)
REMOTE_QUERY_TIMEOUT = 60       # remote status query / remote kill
PROGRESSION_CMD_TIMEOUT = 30    # progression source command (runs each poll cycle)
DEPENDENCY_ACTION_TIMEOUT = 120 # trial dependency action (e.g. file copy, mapFields)
PROCESS_REAP_TIMEOUT = 5        # wait for killed process to exit

jobs : Dict[int, sb.Popen] = {}
trial_progression_step : Dict[int, Dict[str, int]] = {}

# Python callable registry for metrics: metric_name -> callable
# Populated by the fluent API; bypasses subprocess dispatch when set.
_fn_registry: Dict[str, Any] = {}       # metric_name -> fn(parameters, [case_path]) -> float | (float, float)
_progress_fn_registry: Dict[str, Any] = {}  # metric_name -> fn(parameters, [case_path, step]) -> float | (float, float)
def init_trial_progression(trial_index: int, metrics):
    if trial_index not in trial_progression_step:
        trial_progression_step[trial_index] = {}
    if not trial_progression_step[trial_index]:
        trial_progression_step[trial_index] = {metric: 0 for metric in metrics}

class FoamJob(FoamBOBaseModel):
    """An async OpenFOAM job scheduled on an HPC system."""

    id: int = Field(description="Job identifier (PID for local, SLURM job ID for remote, -1 if no runner)")
    parameters: Dict[str, Union[str, float, int, bool]] = Field(description="Trial parameterization values")
    mode: str = Field(description="Execution mode: 'local' or 'remote'")
    case_path: str = Field(description="Absolute path to the trial's OpenFOAM case directory")
    metadata: Dict[str, Any] = Field(description="Runtime metadata: command, pid, cwd, start_time")

    def __repr__(self):
        return self.case_path

    def to_dict(self):
        return self.model_dump()

    @classmethod
    def local_case_run(
            cls,
            parameters: Dict[str, Union[str, float, int, bool]],
            case_path: str,
            cfg: DictConfig,
            hook_env: Dict[str, str] | None = None,
    ):
        """
            Run shell command on local machine; asynchronously; and assume the job
            status queries and early stopping will be handled with a job ID
        """
        job_id = -1
        metadata={
            "command": [],
            "pid": -1,
            "cwd": case_path,
            "start_time": time.time(),
        }
        # Build subprocess environment with all FOAMBO_* variables
        run_env = os.environ.copy()
        run_env["FOAMBO_CASE_PATH"] = case_path
        run_env["FOAMBO_CASE_NAME"] = os.path.basename(case_path)
        if hook_env:
            run_env.update(hook_env)
        cmd = process_input_command(cfg.runner, FoamCase(case_path))
        if cmd:
            cmd = list(cmd)
            if cfg.log_runner == True:
                log_path = os.path.join(case_path, "log.runner")
                log_file = open(log_path, "w")
                proc = sb.Popen(cmd, cwd=case_path, env=run_env, stdout=log_file, stderr=sb.STDOUT, text=True, start_new_session=True)
                # Track the file handle on the proc for cleanup when process completes
                proc._log_file = log_file
            else:
                proc = sb.Popen(cmd, cwd=case_path, env=run_env, stdout=sb.DEVNULL, stderr=sb.PIPE, text=True, start_new_session=True)
                def stream_stderr(pipe):
                    for line in pipe:
                        log.error(f"[{proc.pid}] {line.rstrip()}")
                    pipe.close()
                import threading
                threading.Thread(target=stream_stderr, args=(proc.stderr,), daemon=True).start()
            job_id = proc.pid
            jobs[job_id] = proc
            metadata={
                "command": proc.args,
                "pid": proc.pid,
                "cwd": case_path,
                "start_time": time.time(),
            }
        job = FoamJob(
            id=job_id,
            parameters=parameters,
            mode=cfg['mode'],
            case_path=case_path,
            metadata=metadata,
        )
        return (job_id, job)

    @classmethod
    def local_metric(
            cls,
            metric: str,
            case: FoamCase,
            cfg: DictConfig
    ) -> tuple[int, float | tuple[float, float]]:
        """
            Run shell command on local machine; wait for completion and parse output into metrics
            For convinience, expected output formats map to Ax metric values as follows:
            <scalar>                        -> (0, <scalar>)
            (<mean>, <sem>)                 -> (0, (<mean>, <sem>))
        """
        out = sb.check_output(list(process_input_command(cfg['command'], case)), cwd=case.path, timeout=METRIC_EVAL_TIMEOUT)
        return parse_outcome_for_metric(out, metric)

    @classmethod
    def remote_case_run(
        cls,
        parameters: Dict[str, Union[str, float, int, bool]],
        case_path: str,
        cfg: DictConfig,
        hook_env: Dict[str, str] | None = None,
    ):
        """
            Runs a local command, but assumes the actual job is triggered remotely,
            so that Job status queries and early stopping get carried out properly
            without relying on a job ID
        """
        return cls.local_case_run(parameters=parameters, case_path=case_path, cfg=cfg, hook_env=hook_env)

    @classmethod
    def local_status_query(cls, job_id: int, cfg: DictConfig, metadata: Mapping[str, Any]):
        if job_id == -1:
            return TrialStatus.COMPLETED
        job = jobs[job_id]
        job.poll()
        if job.returncode is None:
            return TrialStatus.RUNNING
        # Process finished — clean up
        jobs.pop(job_id, None)
        if hasattr(job, '_log_file'):
            job._log_file.close()
        if job.returncode == 0:
            return TrialStatus.COMPLETED
        return TrialStatus.FAILED
    
    @classmethod
    def remote_status_query(cls, job_id, cfg, metadata):
        """
        Run a local command to check a remote job's status.
        Expected output is a single line:
        "<Status>"
        or:
        <Status>
        The status being a SLURM-like one https://slurm.schedmd.com/job_state_codes.html
        """
        out = sb.check_output(
            process_input_command(cfg['template_case']['remote_status_query'], FoamCase(metadata['case_path'])),
            cwd=metadata['case_path'], timeout=REMOTE_QUERY_TIMEOUT
        )
        if out.decode("utf-8") == "":
            return TrialStatus.FAILED
        status = out.decode("utf-8").splitlines()[-1].strip('"')
        return getattr(TrialStatus, SLURM_STATUS_MAP[status]) 

    @classmethod
    def local_kill(cls, job_id, cfg, metadata):
        try:
            os.killpg(os.getpgid(job_id), signal.SIGTERM)
        except ProcessLookupError:
            log.debug(f"Process {job_id} already exited, nothing to kill")
        except OSError as e:
            log.error(f"Failed to kill process group for PID {job_id}: {e}")
        # Reap the process to prevent zombies and close any tracked file handles
        proc = jobs.pop(job_id, None)
        if proc is not None:
            try:
                proc.wait(timeout=PROCESS_REAP_TIMEOUT)
            except Exception:
                pass
            if hasattr(proc, '_log_file'):
                proc._log_file.close()

    @classmethod
    def remote_kill(cls, job_id, cfg, metadata):
        """
        Executes a local command to kill a remote job
        """
        try:
            out = sb.check_output(
                process_input_command(cfg['template_case']['remote_early_stop'], case=FoamCase(metadata['case_path'])),
                cwd=metadata['case_path'], timeout=REMOTE_QUERY_TIMEOUT
            )
        except Exception as e:
            log.error(f"Early stop command failed: {cfg['template_case']['remote_early_stop']}\n"
                      f"Case: {metadata.get('case_path', 'unknown')}\n"
                      f"Error: {e}\n"
                      f"The trial may still be running on the remote system — manual cleanup may be needed.")

def _call_metric_fn(fn, parameters: dict, case_path: str | None = None, step: int | None = None):
    """Call a user-provided metric function, adapting to its signature.

    Supports: fn(parameters), fn(parameters, case_path), fn(parameters, case_path, step)
    Returns: float or (float, float)
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    args = [parameters]
    if len(params) >= 2:
        args.append(case_path)
    if len(params) >= 3 and step is not None:
        args.append(step)
    result = fn(*args)
    if isinstance(result, tuple):
        return result
    return float(result)


class FoamJobMetric(IMetric):
    """
    Fetch metric values from HPC jobs
    """
    cfg:  DictConfig
    lower_is_better: bool | None

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True
    dispatcher = {
        "local": FoamJob.local_metric,
    }
    def __init__(self, name: str, cfg):
        super().__init__(name=name)
        self.cfg = cfg
        self.lower_is_better = cfg['evaluate']['lower_is_better'] if 'lower_is_better' in cfg['evaluate'].keys() else None
    def fetch(self, trial_index: int, trial_metadata: Mapping[str, Any]) -> tuple[int, float | tuple[float, float]]:
        # Check for Python callable first (registered by fluent API)
        fn = _fn_registry.get(self.name)
        if fn is not None:
            params = trial_metadata.get("parameters", {})
            case_path = None
            if "job" in trial_metadata and trial_metadata["job"]:
                case_path = trial_metadata["job"].get("case_path")
            mean_sem = _call_metric_fn(fn, params, case_path)
        else:
            out = self.dispatcher[self.cfg['evaluate']['mode']](metric=self.name,
                        case=FoamCase(trial_metadata['job']['case_path']) if 'job' in trial_metadata and trial_metadata['job'] else None,
                        cfg=self.cfg['evaluate'])
            mean_sem = out[1]
        if not isinstance(mean_sem, tuple) and not np.isnan(mean_sem) and trial_index in trial_progression_step.keys() and self.name in trial_progression_step[trial_index].keys():
            trial_progression_step[trial_index][self.name] = trial_progression_step[trial_index][self.name]+1
            return (trial_progression_step[trial_index][self.name], mean_sem)
        return (0, mean_sem)


def encode_foam_metric(metric: FoamJobMetric) -> dict:
    return {
        "__class__": "FoamJobMetric",
        "name": metric.name,  # or however you construct it
    }

class LocalJobMetric(FoamBOBaseModel):
    """Configuration for a metric evaluation command."""
    name: str = Field(description="Metric name, must match what is referenced in `objective` or early stopping config",
                       examples=["metric"])
    command: str | List[str] | None = Field(default=None, description=(
        "Shell command to evaluate the metric at trial completion. "
        "Runs as a blocking op in the trial's CWD. Must print a `<scalar>` or `(<mean>, <sem>)` to stdout. "
        "Supports `$FOAMBO_CASE_NAME` and `$FOAMBO_CASE_PATH` substitution. "
        "Accepts a string (`'echo 0'`) or a list (`['echo', '0']`). "
        "Optional when using Python callables via the library API."
    ), examples=[["echo", "0"]])
    progress: str | List[str] | None = Field(default=None, description=(
        "Optional shell command to evaluate the metric while the trial is still running. "
        "Runs each poll cycle. Supports `$FOAMBO_STEP` substitution with the current progression step. "
        "Accepts a string or a list."
    ), examples=[["echo", "$FOAMBO_STEP"]])
    lower_is_better: bool | None = Field(default=None, description=(
        "Required only if the metric is used in early stopping but is not an objective. "
        "Defines the direction for 'worse' comparison in early stopping strategies."
    ))
    progression_source: str | None = Field(default=None, description=(
        "How to index progression steps for early stopping. Default (null) uses a poll counter.\n"
        "- `foam_time: log.<solver>` — parse latest `Time = ...` from the solver log\n"
        "- `command: <script>` — run a command that prints a number to stdout\n\n"
        "Use this when optimization parameters affect simulation speed (e.g. AMR/mesh refinement), "
        "so trials are compared at the same physical time rather than the same poll count."
    ))

    def to_metric(self):
        cfg = DictConfig({
            "evaluate": {
                "mode": "local",
                "progress": self.progress,
                "command": self.command,
                "lower_is_better": self.lower_is_better
            }
        })
        return FoamJobMetric(self.name, cfg)

_FOAM_TIME_RE = re.compile(r'^Time = (\S+)', re.MULTILINE)

def parse_foam_log_time(case_path: str, log_name: str) -> float | None:
    """Read the latest 'Time = ...' value from an OpenFOAM solver log.

    Reads only the last 64KB of the file for efficiency with large logs.
    """
    log_path = os.path.join(case_path, log_name)
    if not os.path.isfile(log_path):
        return None
    with open(log_path, 'rb') as f:
        try:
            f.seek(-65536, 2)
        except OSError:
            f.seek(0)
        chunk = f.read().decode('utf-8', errors='replace')
    last_match = None
    for m in _FOAM_TIME_RE.finditer(chunk):
        last_match = m
    if last_match:
        try:
            return float(last_match.group(1))
        except ValueError:
            return None
    return None


def resolve_progression(
    metric_cfg: "LocalJobMetric",
    trial_idx: int,
    case_path: str,
) -> float:
    """Resolve the progression value for a metric based on its progression_source.

    Returns:
        float: The progression value — either from the poll counter (default),
        the OpenFOAM solver log, or a user command.
    """
    source = metric_cfg.progression_source
    fallback = trial_progression_step.get(trial_idx, {}).get(metric_cfg.name, 0)

    if not source:
        return fallback

    if source.startswith("foam_time:"):
        log_name = source.split(":", 1)[1].strip()
        foam_time = parse_foam_log_time(case_path, log_name)
        if foam_time is not None:
            return foam_time
        log.debug(f"Could not parse foam time from {log_name} for trial {trial_idx}, "
                  f"falling back to poll count")
        return fallback

    if source.startswith("command:"):
        cmd_str = source.split(":", 1)[1].strip()
        try:
            cmd = process_input_command(cmd_str, FoamCase(case_path))
            out = sb.check_output(cmd, cwd=case_path, timeout=PROGRESSION_CMD_TIMEOUT)
            return float(out.decode('utf-8').strip())
        except Exception as e:
            log.debug(f"Progression command failed for {metric_cfg.name} trial {trial_idx}: {e}, "
                      f"falling back to poll count")
            return fallback

    log.warning(f"Unknown progression_source '{source}' for metric {metric_cfg.name}, "
                f"supported: 'foam_time: <log_file>', 'command: <shell_cmd>'. Using poll count.")
    return fallback


def streaming_metric(client: Client, opt_cfg: Dict):
    """
    Get progress of trial if progress tracking is on for some metrics.
    """
    # Parse objective metric names from objective string like "-m1, -m2, -m3"
    objective_names = set()
    try:
        objective_names = {s.strip().lstrip('+-') for s in opt_cfg["objective"].split(',') if s.strip()}
    except Exception:
        pass
    # Consider only metrics that are NOT part of the objective for streaming
    progress_metrics = [metric["name"] for metric in opt_cfg["metrics"]
                        if metric["name"] not in objective_names
                        and metric.get("progress") and metric["progress"] != "none"
                        and metric["progress"] != "" and metric["progress"] != []]
    init_metrics_tracking = lambda trial_index: init_trial_progression(trial_index, progress_metrics)
    trial_metrics_cfg = lambda trial_idx, metrics_cfg: [
        {
            **metric,
            "progress": (
                metric["progress"]
                    .replace("$FOAMBO_STEP", str(trial_progression_step[trial_idx][metric["name"]]))
                    .replace("$STEP", str(trial_progression_step[trial_idx][metric["name"]]))
                if "progress" in metric and isinstance(metric["progress"], str)
                else [
                    p.replace("$FOAMBO_STEP", str(trial_progression_step[trial_idx][metric["name"]]))
                     .replace("$STEP", str(trial_progression_step[trial_idx][metric["name"]]))
                    if isinstance(p, str) else p
                    for p in metric["progress"]
                ]
            )
        }
        for metric in metrics_cfg
        if metric["name"] not in objective_names
    ]
    # If we have no eligible progress items (command or callable), skip this
    has_progress_cmd = any(cfg.get("progress") and cfg["progress"] != "" and cfg["progress"] != "none"
        for cfg in opt_cfg["metrics"] if cfg["name"] not in objective_names)
    has_progress_fn = any(name in _progress_fn_registry
        for name in [cfg["name"] for cfg in opt_cfg["metrics"]] if name not in objective_names)
    running_trials = client._experiment.trials_by_status[TrialStatus.RUNNING]
    if (not has_progress_cmd and not has_progress_fn) or (len(running_trials) == 0):
        return
    # For running trials, stream metric value using the progress "command" or callable
    for trial in running_trials:
        init_metrics_tracking(trial.index)
        idx = trial.index
        trial_metrics = {}
        case_path = trial.run_metadata.get("case_path") or \
                    trial.run_metadata.get("job", {}).get("case_path", "")
        metadata = {
            "job" : {"case_path": case_path},
            "parameters": trial.arm.parameters if trial.arm else {},
        }
        metrics_cfg = trial_metrics_cfg(idx, opt_cfg["metrics"])
        gms = {metric["name"]: LocalJobMetric.model_validate(metric) for metric in metrics_cfg}
        for k,v in gms.items():
            try:
                progress_fn = _progress_fn_registry.get(k)
                if progress_fn is not None:
                    # Python callable progress
                    step = trial_progression_step.get(idx, {}).get(k, 0)
                    result = _call_metric_fn(progress_fn, metadata["parameters"], case_path, step)
                    if isinstance(result, tuple):
                        is_nan = bool(np.isnan(result[0]))
                    else:
                        is_nan = bool(np.isnan(result))
                    if is_nan:
                        log.debug(f"Attaching {k} data for streaming trial {trial.index} was skipped (NAN outcome)")
                    else:
                        trial_metrics[k] = result
                elif v.progress and v.progress != "" and v.progress != "none":
                    v.command = v.progress
                    metric_val = v.to_metric().fetch(idx, trial_metadata=metadata)
                    is_nan_outcome = False
                    if isinstance(metric_val[1], tuple):
                        is_nan_outcome = bool(np.isnan(metric_val[1][0]))
                    else:
                        is_nan_outcome = bool(np.isnan(metric_val[1]))
                    if is_nan_outcome:
                        log.debug(f"Attaching {k} data for streaming trial {trial.index} was skipped (NAN outcome)")
                    else:
                        trial_metrics[k] = metric_val[1]
            except Exception as e:
                step = trial_progression_step.get(idx, {}).get(k, 0)
                if step == 0:
                    log.debug(f"Streaming data for {k} not yet available at step 0 for trial {idx}: {e}")
                else:
                    log.warning(f"Failed to fetch streaming data for {k}\n"
                                f"Trial: {idx}, case: {case_path}\n"
                                f"Command: {v.progress}, step: {step}\n"
                                f"Error: {e}\n"
                                f"This step won't be recorded; early-stopping may be affected")
        if trial_metrics:
            for metric_name, metric_value in trial_metrics.items():
                metric_cfg = gms[metric_name]
                step = resolve_progression(metric_cfg, idx, metadata['job']['case_path'])
                log.debug(f"Streaming {metric_name} for trial {idx} at step={step} value={metric_value}")
                client.attach_data(
                    trial_index=idx,
                    raw_data={metric_name: metric_value},
                    progression=step,
                )

class FoamJobRunner(IRunner):
    jobs: Dict[int, FoamJob] = {}
    cfg:  DictConfig
    # Runtime-only registry of completed trials, populated by the orchestrator callback.
    # Maps trial_index -> {"case_path": str, "status": str, "parameters": dict}
    trial_registry: Dict[int, Dict[str, Any]] = {}
    # Trial dependency configs, set by optimize() after construction.
    trial_dependencies: List[Any] = []
    # Metric names for this experiment, set by optimize(). Used for caseless mode detection.
    _metric_names: List[str] = []

    @property
    def run_metadata_report_keys(self) -> list[str]:
        return ["case_path"]

    dispatcher = {
        "local": FoamJob.local_case_run,
        "remote": FoamJob.remote_case_run,
    }
    status_query = {
        "local": FoamJob.local_status_query,
        "remote": FoamJob.remote_status_query,
    }
    kill_job = {
        "local": FoamJob.local_kill,
        "remote": FoamJob.remote_kill,
    }
    def __init__(self, cfg: DictConfig, preprocessor: CasePreprocessor | None = None):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg['template_case']['mode']
        self.preprocessor = preprocessor or case_preprocessor

    @classmethod
    def serialize_init_args(cls, obj):
        """Only serialize cfg — runtime attributes are rebuilt on load."""
        return {"cfg": obj.cfg}

    def _resolve_source_trial(self, selector, target_params: dict) -> tuple[int, str] | None:
        """Resolve a source trial index and case path from the registry.

        Returns ``(trial_index, case_path)`` or ``None``.
        """
        completed = {k: v for k, v in self.trial_registry.items()
                     if v.get("status") == "COMPLETED" and v.get("case_path")}
        if not completed:
            return None

        strategy = selector.strategy
        if strategy == "baseline":
            entry = completed.get(0)
            return (0, entry["case_path"]) if entry else None
        if strategy == "by_index":
            entry = completed.get(selector.index)
            return (selector.index, entry["case_path"]) if entry else None
        if strategy == "latest":
            idx = max(completed.keys())
            return (idx, completed[idx]["case_path"])
        if strategy == "best":
            idx = min(completed, key=lambda k: completed[k].get("objective_value", float("inf")))
            return (idx, completed[idx]["case_path"])
        if strategy == "nearest":
            import math
            # Compute parameter ranges from completed trials for normalization
            ranges: dict[str, float] = {}
            for entry in completed.values():
                for k, v in entry.get("parameters", {}).items():
                    if isinstance(v, (int, float)):
                        if k not in ranges:
                            ranges[k] = [v, v]
                        else:
                            ranges[k][0] = min(ranges[k][0], v)
                            ranges[k][1] = max(ranges[k][1], v)
            bounds = {k: (hi - lo) if (hi - lo) > 0 else 1.0
                      for k, (lo, hi) in ranges.items()}
            def _dist(params_a, params_b):
                total = 0.0
                for key in params_a:
                    va, vb = params_a.get(key), params_b.get(key)
                    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                        rng = bounds.get(key, 1.0)
                        total += ((va - vb) / rng) ** 2
                return math.sqrt(total)
            idx = min(completed, key=lambda k: _dist(
                target_params, completed[k].get("parameters", {})))
            return (idx, completed[idx]["case_path"])
        if strategy == "custom":
            cmd = selector.command
            if isinstance(cmd, str):
                cmd = cmd.split()
            try:
                out = sb.check_output(cmd, timeout=PROGRESSION_CMD_TIMEOUT)
                idx = int(out.decode("utf-8").strip())
                entry = completed.get(idx)
                return (idx, entry["case_path"]) if entry else None
            except Exception as e:
                log.warning(f"Custom trial selector failed: {e}")
                return None
        return None

    # Well-known hook phases and their environment variable names
    HOOK_PHASES = ("pre_init", "pre_mesh", "pre_solve", "post_solve")

    def _substitute_cmd(self, cmd: str | List[str], source_path: str, target_path: str) -> str:
        """Apply $FOAMBO_SOURCE_TRIAL / $FOAMBO_TARGET_TRIAL substitution, return a single shell string."""
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        return (cmd
                .replace("$FOAMBO_SOURCE_TRIAL", source_path).replace("$SOURCE_TRIAL", source_path)
                .replace("$FOAMBO_TARGET_TRIAL", target_path).replace("$TARGET_TRIAL", target_path))

    def _execute_dependency_actions(self, dep, source_path: str, target_path: str) -> List[str]:
        """Execute a dependency's immediate actions, return list of action types applied."""
        applied = []
        for action in dep.actions:
            if action.phase != "immediate":
                continue
            if action.type == "run_command":
                cmd = self._substitute_cmd(action.command, source_path, target_path)
                try:
                    sb.check_call(cmd, shell=True, cwd=target_path, timeout=DEPENDENCY_ACTION_TIMEOUT)
                    applied.append("run_command")
                except Exception as e:
                    log.warning(f"Dependency '{dep.name}' action failed: {e}")
        return applied

    def _write_hook_scripts(self, target_path: str, phase_commands: Dict[str, List[str]]) -> Dict[str, str]:
        """Write per-phase hook scripts into the trial case directory.

        Returns a mapping of ``FOAMBO_<PHASE>`` env-var names to script paths.
        Phases with no commands get a no-op script so Allrun scripts never fail.
        """
        hook_env: Dict[str, str] = {}
        for phase in self.HOOK_PHASES:
            env_name = f"FOAMBO_{phase.upper()}"
            script_path = os.path.join(target_path, f".foambo_{phase}.sh")
            cmds = phase_commands.get(phase, [])
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"# foamBO hook: {phase}\n")
                f.write("set -e\n")
                if cmds:
                    for cmd in cmds:
                        f.write(f"{cmd}\n")
                else:
                    f.write("# (no actions for this phase)\n")
                    f.write("true\n")
            os.chmod(script_path, 0o755)
            hook_env[env_name] = script_path
        return hook_env

    def _resolve_dependencies(self, trial_index: int, target_path: str,
                              parameterization: dict) -> Dict[str, Any]:
        """Resolve all enabled trial dependencies.

        * ``immediate`` actions execute inline (before the runner starts).
        * Phased actions (``pre_mesh``, ``post_mesh``, ``pre_solve``) are
          written to hook scripts inside the trial case directory and exposed
          to the runner via ``$FOAMBO_PRE_MESH``, ``$FOAMBO_POST_MESH``,
          ``$FOAMBO_PRE_SOLVE`` environment variables.

        Returns a metadata dict with per-dependency info and a ``hook_env``
        mapping for the runner subprocess.
        """
        dep_meta: Dict[str, Any] = {}
        phase_commands: Dict[str, List[str]] = {}
        last_source_path: str | None = None

        for dep in self.trial_dependencies:
            if not dep.enabled:
                continue
            result = self._resolve_source_trial(dep.source, parameterization)
            if result is None:
                if dep.source.fallback == "error":
                    raise RuntimeError(
                        f"Dependency '{dep.name}': no source trial found and fallback is 'error'")
                log.debug(f"Dependency '{dep.name}': no source trial found, skipping")
                continue

            source_index, source_path = result
            last_source_path = source_path
            log.info(f"Trial {trial_index}: resolving dependency '{dep.name}' from trial {source_index} at {source_path}")

            # Execute immediate actions now
            applied = self._execute_dependency_actions(dep, source_path, target_path)

            # Collect phased actions for hook scripts
            phased = []
            for action in dep.actions:
                if action.phase != "immediate" and action.type == "run_command":
                    resolved_cmd = self._substitute_cmd(action.command, source_path, target_path)
                    phase_commands.setdefault(action.phase, []).append(resolved_cmd)
                    phased.append(action.phase)

            dep_meta[dep.name] = {
                "source_trial_index": source_index,
                "source_case_path": source_path,
                "actions_applied": applied,
                "phased_actions": phased,
            }

        # Always write hook scripts (no-op when empty) so env vars are always set
        hook_env = self._write_hook_scripts(target_path, phase_commands)
        # Expose source/target as env vars for the runner (empty string if no source found)
        hook_env["FOAMBO_SOURCE_TRIAL"] = last_source_path or ""
        hook_env["FOAMBO_TARGET_TRIAL"] = target_path
        dep_meta["_hook_env"] = hook_env

        return dep_meta

    def run_trial(self, trial_index: int, parameterization: TParameterization) -> dict[str, Any]:
        trial_progression_step[trial_index] = {}
        case_data = self.preprocessor.setup(parameterization, self.cfg)
        case_path = str(case_data['case'].path)
        # Resolve trial dependencies before dispatching
        dep_meta = self._resolve_dependencies(trial_index, case_path, dict(parameterization))
        # Extract hook env vars (always present, may be no-ops)
        hook_env = dep_meta.pop("_hook_env", {})
        # If all metrics are Python callables and no runner is set, skip subprocess dispatch
        all_metrics_are_fns = all(m in _fn_registry for m in self._metric_names) if self._metric_names else False
        has_runner = self.cfg['template_case'].get('runner') not in (None, "", "null", "None")
        if all_metrics_are_fns and not has_runner:
            log.info(f"Trial {trial_index}: caseless mode (Python callables)")
            meta = {"job_id": -1, "job": {"case_path": case_path}, "case_path": case_path,
                    "parameters": dict(parameterization)}
        else:
            (job_id, job) = self.dispatcher[self.mode](
                parameterization, case_path, self.cfg['template_case'], hook_env=hook_env)
            log.info(f"Dispatched trial {trial_index}: {case_path}")
            meta = {"job_id": job_id, "job": job.model_dump(), "case_path": case_path,
                    "parameters": dict(parameterization)}
        if dep_meta:
            meta["dependencies"] = dep_meta
        return meta
    # Stashed by optimize() so poll_trial can attach streaming data
    _streaming_cfg: Dict | None = None
    _streaming_client: Any = None

    def poll_trial(self, trial_index: int, trial_metadata: Mapping[str, Any]) -> TrialStatus:
        if not trial_metadata or 'job_id' not in trial_metadata:
            return TrialStatus.COMPLETED
        status = self.status_query[self.mode](trial_metadata['job_id'], self.cfg, trial_metadata)
        # Attach streaming data during poll so it's visible to early stopping
        if status == TrialStatus.RUNNING and self._streaming_client is not None and self._streaming_cfg is not None:
            try:
                streaming_metric(self._streaming_client, self._streaming_cfg)
            except Exception:
                pass
        return status
    def stop_trial(self, trial_index: int, trial_metadata: Mapping[str, Any]) -> dict[str, Any]:
        return self.kill_job[self.mode](trial_metadata['job_id'], self.cfg, trial_metadata)

### JSON serialize/deserialize

def config_to_dict(config: DictConfig) -> Union[Dict[DictKeyType, Any], List[Any], None, str, Any]:
    return OmegaConf.to_object(config)

def config_from_json(config: Dict[str, Any]) -> DictConfig:
    return DictConfig(content=config)

def foam_job_to_dict(job: FoamJob) -> Union[Dict[DictKeyType, Any], List[Any], None, str, Any]:
    return job.model_dump()

def foam_job_from_json(config: Dict[str, Any]) -> FoamJob:
    return FoamJob.model_validate(config)

CORE_CLASS_DECODER_REGISTRY["FoamJob"] = foam_job_from_json
CORE_ENCODER_REGISTRY[FoamJob] = foam_job_to_dict
CORE_CLASS_DECODER_REGISTRY["Type[DictConfig]"] = config_from_json
CORE_ENCODER_REGISTRY[DictConfig] = config_to_dict

CORE_ENCODER_REGISTRY[FoamJobRunner] = runner_to_dict;
CORE_DECODER_REGISTRY["FoamJobRunner"] = FoamJobRunner
register_runner(FoamJobRunner)

CORE_ENCODER_REGISTRY[FoamJobMetric] = metric_to_dict
CORE_DECODER_REGISTRY["FoamJobMetric"] = FoamJobMetric
register_metrics({FoamJobMetric: len(CORE_METRIC_REGISTRY.items())})

# Register Cast transform with Ax's JSON serialization (missing from Ax's default registry)
from ax.storage.json_store.decoders import REVERSE_TRANSFORM_REGISTRY
from ax.adapter.transforms.cast import Cast
if "Cast" not in REVERSE_TRANSFORM_REGISTRY:
    REVERSE_TRANSFORM_REGISTRY["Cast"] = Cast
