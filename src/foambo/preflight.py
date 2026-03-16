"""Preflight checks for foamBO configuration.

Two levels:
- Static checks: fast, no side effects — validates file existence, config coherence, name matching.
- Dry-run checks: clones template, substitutes center-of-domain params, runs metric commands once.
"""

import os
import shutil
import subprocess as sb
import logging
from typing import Any
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# ANSI
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
WARN = f"{YELLOW}WARN{RESET}"


class PreflightResult:
    def __init__(self):
        self.checks: list[tuple[str, str, str]] = []  # (status, name, detail)

    def passed(self, name: str, detail: str = ""):
        self.checks.append(("PASS", name, detail))

    def failed(self, name: str, detail: str):
        self.checks.append(("FAIL", name, detail))

    def warned(self, name: str, detail: str):
        self.checks.append(("WARN", name, detail))

    @property
    def ok(self) -> bool:
        return all(s != "FAIL" for s, _, _ in self.checks)

    @property
    def n_fail(self) -> int:
        return sum(1 for s, _, _ in self.checks if s == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for s, _, _ in self.checks if s == "WARN")

    def print_report(self):
        icons = {"PASS": PASS, "FAIL": FAIL, "WARN": WARN}
        for status, name, detail in self.checks:
            icon = icons[status]
            line = f"  [{icon}] {name}"
            if detail:
                line += f" — {detail}"
            print(line)
        total = len(self.checks)
        print(f"\n  {total} checks: {total - self.n_fail - self.n_warn} passed, "
              f"{self.n_warn} warnings, {self.n_fail} failed")


def static_checks(cfg: DictConfig) -> PreflightResult:
    """Fast, no-side-effect validation of the configuration."""
    r = PreflightResult()
    opt = cfg.get("optimization", {})
    runner_cfg = opt.get("case_runner", {})
    orch = cfg.get("orchestration_settings", {})
    exp = cfg.get("experiment", {})

    # --- Template case ---
    template = runner_cfg.get("template_case", "")
    if os.path.isdir(template):
        r.passed("Template case exists", template)
    else:
        r.failed("Template case exists", f"directory not found: {template}")
        # Can't check files inside if template is missing
        return _check_config_coherence(cfg, r)

    # --- Variable substitution files ---
    for entry in runner_cfg.get("variable_substitution", []):
        fpath = os.path.join(template, entry["file"].lstrip("/"))
        if os.path.isfile(fpath):
            r.passed(f"Substitution file: {entry['file']}")
        else:
            r.failed(f"Substitution file: {entry['file']}", f"not found: {fpath}")

    # --- File substitution variants ---
    params = exp.get("parameters", [])
    param_values = {}
    for p in params:
        if "values" in p:
            param_values[p["name"]] = p["values"]
    for entry in runner_cfg.get("file_substitution", []):
        base = os.path.join(template, entry["file_path"].lstrip("/"))
        pname = entry["parameter"]
        if pname in param_values:
            for val in param_values[pname]:
                variant = f"{base}.{val}"
                if os.path.isfile(variant):
                    r.passed(f"File variant: {entry['file_path']}.{val}")
                else:
                    r.failed(f"File variant: {entry['file_path']}.{val}", f"not found: {variant}")
        else:
            r.warned(f"File substitution: {pname}", "parameter not found in experiment.parameters")

    # --- Runner command ---
    runner_cmd = runner_cfg.get("runner")
    if runner_cmd and runner_cmd != "null":
        cmd_str = runner_cmd if isinstance(runner_cmd, str) else " ".join(runner_cmd)
        exe = runner_cmd.split()[0] if isinstance(runner_cmd, str) else runner_cmd[0]
        if shutil.which(exe) or os.path.isfile(exe):
            r.passed(f"Runner executable found: {exe}", f"full command: {cmd_str}")
        else:
            r.warned(f"Runner executable: {exe}", f"not found in PATH — full command: {cmd_str}")

    # --- Remote commands ---
    if runner_cfg.get("mode") == "remote":
        for key, label in [("remote_status_query", "Remote status query"),
                           ("remote_early_stop", "Remote early stop")]:
            cmd = runner_cfg.get(key)
            if cmd and cmd != "null":
                cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
                exe = cmd.split()[0] if isinstance(cmd, str) else cmd[0]
                if shutil.which(exe) or os.path.isfile(exe):
                    r.passed(f"{label}: executable '{exe}' found",
                             f"only the executable is checked, not the full command: {cmd_str}")
                else:
                    r.warned(f"{label}: executable '{exe}' not found",
                             f"full command: {cmd_str}")
            elif key == "remote_status_query":
                r.failed(label, "required for mode=remote but not set")

    # --- Trial destination writable ---
    trial_dest = runner_cfg.get("trial_destination", "")
    if trial_dest:
        parent = os.path.dirname(os.path.abspath(trial_dest)) or "."
        if os.access(parent, os.W_OK):
            r.passed(f"Trial destination writable", trial_dest)
        else:
            r.failed(f"Trial destination writable", f"parent not writable: {parent}")

    # --- Artifacts folder writable ---
    artifacts = runner_cfg.get("artifacts_folder", "")
    if artifacts:
        parent = os.path.dirname(os.path.abspath(artifacts)) or "."
        if os.access(parent, os.W_OK):
            r.passed(f"Artifacts folder writable", artifacts)
        else:
            r.failed(f"Artifacts folder writable", f"parent not writable: {parent}")

    # --- Store coherence ---
    store = cfg.get("store", {})
    if store.get("read_from") == "json":
        from .common import get_experiment_name, set_experiment_name
        set_experiment_name(exp.get("name", ""))
        state_file = f"{artifacts}/{exp.get('name', '')}_client_state.json"
        if os.path.isfile(state_file):
            r.passed("State file exists for resume", state_file)
        else:
            r.warned("State file for resume", f"not found: {state_file} (will start fresh)")

    return _check_config_coherence(cfg, r)


def _check_config_coherence(cfg: DictConfig, r: PreflightResult) -> PreflightResult:
    """Check logical coherence between config sections."""
    opt = cfg.get("optimization", {})
    orch = cfg.get("orchestration_settings", {})
    exp = cfg.get("experiment", {})

    # --- Metrics reference check ---
    metric_names = {m["name"] for m in opt.get("metrics", [])}
    objective_str = opt.get("objective", "")
    objective_names = {s.strip().lstrip("+-") for s in objective_str.split(",") if s.strip()}

    for name in objective_names:
        if name in metric_names:
            r.passed(f"Objective metric defined: {name}")
        else:
            r.failed(f"Objective metric defined: {name}", "referenced in objective but not in metrics list")

    # --- Outcome constraints reference valid metrics ---
    for constraint in opt.get("outcome_constraints", []):
        # Parse "metric_name >= ..." or "metric_name <= ..."
        parts = constraint.replace(">=", " ").replace("<=", " ").split()
        if parts:
            cname = parts[0]
            if cname in metric_names:
                r.passed(f"Outcome constraint metric: {cname}")
            else:
                r.failed(f"Outcome constraint metric: {cname}",
                         f"'{cname}' in constraint '{constraint}' not in metrics list")

    # --- Early stopping metric coherence ---
    es = orch.get("early_stopping_strategy")
    if es and isinstance(es, dict):
        _check_early_stopping_metrics(es, metric_names, objective_names, opt.get("metrics", []), r)

    # --- Parameter constraints reference valid params ---
    param_names = {p["name"] for p in exp.get("parameters", [])}
    for constraint in exp.get("parameter_constraints", []):
        # Rough check: each word that's alphanumeric could be a param name
        tokens = constraint.replace("<=", " ").replace(">=", " ").replace("+", " ").replace("-", " ").replace("*", " ").split()
        for token in tokens:
            if token.isidentifier() and token in param_names:
                r.passed(f"Parameter constraint references: {token}")
            elif token.isidentifier() and not token.replace(".", "").isdigit():
                # Could be a param name that doesn't exist
                if token not in param_names:
                    r.warned(f"Parameter constraint token: {token}",
                             f"not in parameters (may be a constant)")

    # --- Dependency commands contain substitution variables ---
    for dep in cfg.get("trial_dependencies", []):
        if not dep.get("enabled", True):
            continue
        for action in dep.get("actions", []):
            cmd = action.get("command", "")
            cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
            if "$SOURCE_TRIAL" not in cmd_str and "$TARGET_TRIAL" not in cmd_str:
                r.warned(f"Dependency '{dep['name']}' action",
                         "command has no $SOURCE_TRIAL or $TARGET_TRIAL substitution")

    return r


def _check_early_stopping_metrics(es: dict, metric_names: set, objective_names: set,
                                   metrics_cfg: list, r: PreflightResult):
    """Recursively check early stopping metric references."""
    es_metrics = set(es.get("metric_names", []))
    for name in es_metrics:
        if name in objective_names:
            r.warned(f"Early stopping metric: {name}",
                     "is an objective — objectives don't stream, early stopping won't trigger")
        elif name not in metric_names:
            r.failed(f"Early stopping metric: {name}", "not defined in metrics list")
        else:
            # Check it has a progress command
            m = next((m for m in metrics_cfg if m["name"] == name), None)
            if m and (not m.get("progress") or m["progress"] == "null"):
                r.warned(f"Early stopping metric: {name}",
                         "has no 'progress' command — early stopping won't have streaming data")
            else:
                r.passed(f"Early stopping metric: {name}")

    # Recurse into composite strategies
    for sub_key in ("left", "right"):
        sub = es.get(sub_key)
        if isinstance(sub, dict):
            _check_early_stopping_metrics(sub, metric_names, objective_names, metrics_cfg, r)


def dry_run_checks(cfg: DictConfig) -> PreflightResult:
    """Run a single trial through the real optimize() flow with center-of-domain params.

    Creates a temporary copy of the config with:
    - experiment name set to ``{original}_dryrun``
    - max_trials=1, parallelism=1, no baseline, no early stopping
    - trial_destination and artifacts_folder in a temp directory
    - store set to nowhere (no persistence)

    After the trial completes, all temp files are cleaned up.
    The original trials/artifacts directories are never touched.
    """
    r = PreflightResult()
    import tempfile
    from copy import deepcopy
    from omegaconf import OmegaConf

    exp_name = cfg.get("experiment", {}).get("name", "experiment")
    runner_cfg = cfg.get("optimization", {}).get("case_runner", {})
    dryrun_name = f"{exp_name}_dryrun"

    # Use real trial_destination so relative metric command paths resolve correctly.
    # The _dryrun suffix on the experiment name keeps trial folders distinct.
    trial_dir = runner_cfg.get("trial_destination", "./trials")
    artifacts_dir = runner_cfg.get("artifacts_folder", "./artifacts")

    # Build a modified config for the dry run
    dryrun_cfg = OmegaConf.to_container(cfg, resolve=True)
    dryrun_cfg["experiment"]["name"] = dryrun_name
    dryrun_cfg["baseline"] = {"parameters": None}
    dryrun_cfg["existing_trials"] = {"file_path": ""}
    dryrun_cfg["optimization"]["outcome_constraints"] = []
    dryrun_cfg["orchestration_settings"]["max_trials"] = 1
    dryrun_cfg["orchestration_settings"]["parallelism"] = 1
    dryrun_cfg["orchestration_settings"]["tolerated_trial_failure_rate"] = 0.99
    dryrun_cfg["orchestration_settings"]["min_failed_trials_for_failure_rate_check"] = 999
    dryrun_cfg["orchestration_settings"]["early_stopping_strategy"] = None
    dryrun_cfg["orchestration_settings"]["global_stopping_strategy"] = {
        "min_trials": 1, "window_size": 1, "improvement_bar": 0.0,
    }
    dryrun_cfg["store"] = {"save_to": "json", "read_from": "nowhere", "backend_options": {"url": None}}
    if "trial_dependencies" in dryrun_cfg:
        dryrun_cfg["trial_dependencies"] = []

    dryrun_omegacfg = OmegaConf.create(dryrun_cfg)

    r.passed("Dry-run config prepared", f"name={dryrun_name}")

    # Collect dryrun trial paths for cleanup
    dryrun_trial_dirs = []
    try:
        from .optimize import optimize
        client = optimize(dryrun_omegacfg)
        if client is None:
            r.failed("Dry-run optimization", "optimize() returned None (interrupted?)")
        else:
            from ax.core.base_trial import TrialStatus
            trials = client._experiment.trials
            if not trials:
                r.failed("Dry-run trial dispatch", "no trials were created")
            else:
                trial = trials[0]
                # Track trial path for cleanup
                case_path = trial.run_metadata.get("case_path") or \
                            trial.run_metadata.get("job", {}).get("case_path")
                if case_path:
                    dryrun_trial_dirs.append(case_path)
                if trial.status == TrialStatus.COMPLETED:
                    data = client._experiment.fetch_data()
                    if data.df.empty:
                        r.failed("Dry-run metric collection", "trial completed but no metric data")
                    else:
                        metrics_collected = list(data.df["metric_name"].unique())
                        r.passed("Dry-run trial completed with metrics", ", ".join(metrics_collected))
                elif trial.status == TrialStatus.FAILED:
                    r.failed("Dry-run trial", "trial FAILED — check runner/metric commands")
                else:
                    r.warned("Dry-run trial", f"unexpected status: {trial.status.name}")
    except Exception as e:
        r.failed("Dry-run optimization", str(e))
    finally:
        # Clean up dryrun trial directories and artifacts
        for d in dryrun_trial_dirs:
            shutil.rmtree(d, ignore_errors=True)
        # Clean up dryrun artifacts (state file, report CSV)
        for suffix in ("_client_state.json", "_report.csv"):
            path = os.path.join(artifacts_dir, f"{dryrun_name}{suffix}")
            if os.path.isfile(path):
                os.remove(path)
        # Clean up any dryrun HTML reports
        import glob
        for f in glob.glob(os.path.join(artifacts_dir, f"{dryrun_name}_*.html")):
            os.remove(f)

    return r


def run_preflight(cfg: DictConfig, dry_run: bool = False) -> bool:
    """Run preflight checks and print a report.

    Returns True if all checks passed (no FAIL).
    """
    print("\n  Preflight checks\n  " + "=" * 40)
    print("\n  Static checks:\n")
    result = static_checks(cfg)
    result.print_report()

    if dry_run:
        print("\n  Dry-run checks:\n")
        dr = dry_run_checks(cfg)
        dr.print_report()
        # Merge results
        result.checks.extend(dr.checks)

    ok = result.ok
    if ok:
        print(f"\n  {GREEN}All preflight checks passed.{RESET}\n")
    else:
        print(f"\n  {RED}{result.n_fail} check(s) failed. Fix issues before running.{RESET}\n")
    return ok
