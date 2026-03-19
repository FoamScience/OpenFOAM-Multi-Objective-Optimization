"""Fluent API for foamBO optimization.

Usage::

    from foambo import FoamBO

    client = (
        FoamBO("SingleObjF1", case="./case", trials="./trials", artifacts="./artifacts")
        .parameter("x", bounds=[-100.0, 200.0])
        .minimize("F1", command="python3 benchmark.py --F F1 --k 1 --m 0 --lb 0.01")
        .substitute("/FxDict", x="x")
        .stop(max_trials=50, improvement_bar=0.1)
        .run(parallelism=3)
    )

    predictions = client.predict([{"x": 10}])
"""

from __future__ import annotations
from typing import Any


class FoamBO:
    """Builder for foamBO optimization runs.

    Args:
        name: Experiment name (used for artifact naming and state files).
        case: Path to the template OpenFOAM case directory.
        trials: Directory for trial case copies (default: ``./trials``).
        artifacts: Directory for reports and state files (default: ``./artifacts``).
        description: Short experiment description.
        mode: Execution mode — ``"local"`` or ``"remote"`` (default: ``"local"``).
    """

    def __init__(
        self,
        name: str,
        case: str | None = None,
        trials: str = "./trials",
        artifacts: str = "./artifacts",
        description: str = "",
        mode: str = "local",
    ):
        self._name = name
        self._description = description or f"{name} optimization"
        self._case = case
        self._trials = trials
        self._artifacts = artifacts
        self._mode = mode

        self._parameters: list[dict] = []
        self._param_constraints: list[str] = []
        self._metrics: list[dict] = []
        self._objectives: list[str] = []
        self._fn_map: dict[str, Any] = {}           # metric_name -> callable
        self._progress_fn_map: dict[str, Any] = {}   # metric_name -> callable
        self._outcome_constraints: list[str] = []
        self._variable_subst: list[dict] = []
        self._file_subst: list[dict] = []
        self._baseline: dict | None = None
        self._trial_gen: dict = {"method": "fast"}
        self._existing_trials: str = ""
        self._dependencies: list[dict] = []

        # Runner options
        self._runner: str | None = None
        self._log_runner: bool = False
        self._remote_status: str | None = None
        self._remote_stop: str | None = None

        # Orchestration
        self._max_trials: int = 50
        self._parallelism: int = 3
        self._poll_interval: int = 10
        self._poll_backoff: float = 1.5
        self._timeout_hours: int = 24
        self._ttl: int | None = None
        self._global_stop: dict | None = None
        self._early_stop: dict | None = None

        # Store
        self._save_to: str = "json"
        self._read_from: str = "nowhere"
        self._backend_url: str | None = None

        # Visualizer
        self._sensitivity_callback: str | None = None
        self._sensitivity_fn = None

    # --- Parameters ---

    def parameter(
        self,
        name: str,
        bounds: list[float] | None = None,
        values: list | None = None,
        parameter_type: str | None = None,
        depends: dict[str, list[str]] | None = None,
        **kwargs,
    ) -> FoamBO:
        """Add an optimization parameter.

        Args:
            name: Parameter name.
            bounds: ``[lower, upper]`` for range parameters.
            values: List of allowed values for choice parameters.
            parameter_type: ``"float"``, ``"int"``, or ``"str"``. Auto-detected if omitted.
            depends: For choice parameters — maps each value to a list of
                parameter names that are only active when that value is selected.
                Example: ``depends={"PCG": ["pcg_tol"], "GAMG": ["gamg_tol"]}``
            **kwargs: Extra fields (``step_size``, ``scaling``, ``is_ordered``, etc.).
        """
        if depends is not None:
            kwargs["dependent_parameters"] = depends
        p: dict[str, Any] = {"name": name, **kwargs}
        if bounds is not None:
            p["bounds"] = bounds
            if parameter_type is None:
                parameter_type = "int" if all(isinstance(b, int) for b in bounds) else "float"
        elif values is not None:
            p["values"] = values
            if parameter_type is None:
                parameter_type = "str" if all(isinstance(v, str) for v in values) else "float"
        if parameter_type:
            p["parameter_type"] = parameter_type
        self._parameters.append(p)
        return self

    def constraint(self, expr: str) -> FoamBO:
        """Add a parameter constraint (e.g. ``"x1 <= x2 + 5"``)."""
        self._param_constraints.append(expr)
        return self

    # --- Metrics & Objectives ---

    def _add_metric(self, name: str, minimize: bool | None,
                    fn=None, progress_fn=None, **kwargs) -> FoamBO:
        m: dict[str, Any] = {"name": name, **kwargs}
        if minimize is not None:
            m["lower_is_better"] = minimize
        # For fn-based metrics, set a placeholder command so Ax config validation passes
        if fn is not None:
            self._fn_map[name] = fn
            m.setdefault("command", f"__fn__{name}")
        if progress_fn is not None:
            self._progress_fn_map[name] = progress_fn
            m.setdefault("progress", f"__progress_fn__{name}")
        self._metrics.append(m)
        return self

    def minimize(self, name: str, command: str | list[str] | None = None,
                 progress: str | list[str] | None = None,
                 fn=None, progress_fn=None, **kwargs) -> FoamBO:
        """Add a metric and minimize it as an objective.

        Either ``command`` (shell) or ``fn`` (Python callable) must be provided.
        ``fn`` signature: ``fn(parameters: dict) -> float`` or
        ``fn(parameters: dict, case_path: str) -> float``.
        """
        self._objectives.append(f"-{name}")
        return self._add_metric(name, minimize=True, command=command, progress=progress,
                                fn=fn, progress_fn=progress_fn, **kwargs)

    def maximize(self, name: str, command: str | list[str] | None = None,
                 progress: str | list[str] | None = None,
                 fn=None, progress_fn=None, **kwargs) -> FoamBO:
        """Add a metric and maximize it as an objective.

        Either ``command`` (shell) or ``fn`` (Python callable) must be provided.
        """
        self._objectives.append(name)
        return self._add_metric(name, minimize=False, command=command, progress=progress,
                                fn=fn, progress_fn=progress_fn, **kwargs)

    def track(self, name: str, command: str | list[str] | None = None,
              progress: str | list[str] | None = None,
              lower_is_better: bool | None = None,
              fn=None, progress_fn=None, **kwargs) -> FoamBO:
        """Add a tracking metric (not an objective, but available for early stopping).

        Either ``command`` (shell) or ``fn`` (Python callable) must be provided.
        """
        return self._add_metric(name, minimize=lower_is_better, command=command, progress=progress,
                                fn=fn, progress_fn=progress_fn, **kwargs)

    def outcome_constraint(self, expr: str) -> FoamBO:
        """Add an outcome constraint (e.g. ``"metric1 >= 0.5*baseline"``)."""
        self._outcome_constraints.append(expr)
        return self

    # --- Case Setup ---

    def substitute(self, file: str, **parameter_scopes: str) -> FoamBO:
        """Map parameters to OpenFOAM file fields via foamlib.

        Args:
            file: Relative path to the OpenFOAM file (e.g. ``"/0orig/U"``).
            **parameter_scopes: Mapping of parameter name to dotted path
                (e.g. ``x="someDict.x"``).
        """
        self._variable_subst.append({"file": file, "parameter_scopes": parameter_scopes})
        return self

    def file_substitute(self, parameter: str, file_path: str) -> FoamBO:
        """Replace a case file with a variant based on a choice parameter value."""
        self._file_subst.append({"parameter": parameter, "file_path": file_path})
        return self

    def runner(self, command: str, log: bool = False) -> FoamBO:
        """Set the shell command to execute each trial (e.g. ``"./Allrun"``)."""
        self._runner = command
        self._log_runner = log
        return self

    def remote(self, runner: str, status_query: str, early_stop: str | None = None) -> FoamBO:
        """Configure remote (SLURM) execution."""
        self._mode = "remote"
        self._runner = runner
        self._remote_status = status_query
        self._remote_stop = early_stop
        return self

    # --- Stopping & Orchestration ---

    def stop(
        self,
        max_trials: int = 50,
        improvement_bar: float = 0.1,
        min_trials: int = 10,
        window_size: int = 5,
    ) -> FoamBO:
        """Configure global stopping strategy."""
        self._max_trials = max_trials
        self._global_stop = {
            "min_trials": min_trials,
            "window_size": window_size,
            "improvement_bar": improvement_bar,
        }
        return self

    def early_stop(self, **kwargs) -> FoamBO:
        """Configure early stopping strategy (passed directly to Ax).

        Example::

            .early_stop(type="percentile", metric_names=["m1"],
                        percentile_threshold=25, min_progression=5)
        """
        self._early_stop = kwargs
        return self

    # --- Trial Dependencies ---

    def depend(self, name: str, source: str = "best", command: str | list[str] = "",
               fallback: str = "skip", enabled: bool = True) -> FoamBO:
        """Add a trial dependency.

        Args:
            name: Label for this dependency.
            source: Selection strategy (``"best"``, ``"nearest"``, ``"latest"``, ``"baseline"``).
            command: Shell command with ``$SOURCE_TRIAL`` / ``$TARGET_TRIAL`` substitution.
            fallback: ``"skip"`` or ``"error"`` when no source trial found.
        """
        self._dependencies.append({
            "name": name,
            "enabled": enabled,
            "source": {"strategy": source, "fallback": fallback},
            "actions": [{"type": "run_command", "command": command}] if command else [],
        })
        return self

    # --- Baseline & Resumption ---

    def baseline(self, **params) -> FoamBO:
        """Set baseline parameter values for comparison."""
        self._baseline = params
        return self

    def resume(self, backend: str = "json") -> FoamBO:
        """Resume from a previously saved experiment state."""
        self._read_from = backend
        return self

    def generation(self, method: str = "fast", **kwargs) -> FoamBO:
        """Configure trial generation strategy."""
        self._trial_gen = {"method": method, **kwargs}
        return self

    # --- Visualizer ---

    def visualizer(self, sensitivity_callback: str | None = None,
                   sensitivity_fn=None) -> FoamBO:
        """Configure the web-based visualizer.

        Args:
            sensitivity_callback: Dotted import path to a Python function
                (e.g. ``"mymodule.plot_function"``). Used in both CLI and library mode.
            sensitivity_fn: A Python callable ``fn(parameters: dict) -> str``
                that returns a base64-encoded PNG. Library mode only.
                Takes priority over ``sensitivity_callback`` if both are set.
        """
        self._sensitivity_callback = sensitivity_callback
        self._sensitivity_fn = sensitivity_fn
        return self

    @staticmethod
    def cross_validate(client) -> list[dict]:
        """Run leave-one-out cross-validation on the fitted surrogate model.

        Args:
            client: The Ax ``Client`` returned by ``.run()``.

        Returns:
            A list of dicts, one per observation:
            ``{"trial_index", "arm_name", "metric_name", "observed_mean",
              "predicted_mean", "predicted_sem"}``.

        Raises:
            RuntimeError: If no BO model has been fitted yet.

        Example::

            client = FoamBO("Exp").minimize("F1", fn=f).run()
            cv = FoamBO.cross_validate(client)
            for row in cv:
                print(f"Trial {row['trial_index']} {row['metric_name']}: "
                      f"observed={row['observed_mean']:.3f} "
                      f"predicted={row['predicted_mean']:.3f}")
        """
        from ax.adapter.cross_validation import cross_validate as ax_cv

        gs = client._generation_strategy
        if gs is None or gs.adapter is None:
            raise RuntimeError(
                "No fitted model available for cross-validation. "
                "Ensure enough trials completed to reach the BO phase.")

        import math
        cv_results = ax_cv(model=gs.adapter)
        rows = []
        for r in cv_results:
            obs_means = r.observed.data.means_dict
            pred_means = r.predicted.means_dict
            pred_cov = r.predicted.covariance_matrix
            for metric_name, obs_mean in obs_means.items():
                rows.append({
                    "trial_index": r.observed.features.trial_index,
                    "arm_name": r.observed.arm_name,
                    "metric_name": metric_name,
                    "observed_mean": obs_mean,
                    "predicted_mean": pred_means.get(metric_name, float("nan")),
                    "predicted_sem": math.sqrt(abs(
                        pred_cov.get(metric_name, {}).get(metric_name, 0.0))),
                })
        return rows

    def show(self, host: str = "127.0.0.1", port: int = 8099, open_browser: bool = True):
        """Launch the visualization UI for the current experiment.

        Requires a completed or in-progress experiment with saved state.
        """
        from .visualize import visualizer_ui
        # Register callable so the visualizer can use it directly
        if hasattr(self, '_sensitivity_fn') and self._sensitivity_fn is not None:
            from . import visualize as _viz
            _viz._sensitivity_fn_override = self._sensitivity_fn
        cfg = self.build().to_dictconfig()
        visualizer_ui(cfg, host=host, port=port, open_browser=open_browser)

    # --- Build, Check & Run ---

    def build(self) -> "FoamBOConfig":
        """Build the FoamBOConfig without running the optimization."""
        from .orchestrate import FoamBOConfig
        return FoamBOConfig.model_validate(self._to_dict())

    def preflight(self, dry_run: bool = False) -> bool:
        """Run preflight checks and print a report.

        Args:
            dry_run: If True, also clone the template case, substitute
                center-of-domain parameters, and run each metric command once.

        Returns:
            True if all checks passed.
        """
        from omegaconf import DictConfig
        from .preflight import run_preflight
        cfg = DictConfig(self._to_dict())
        return run_preflight(cfg, dry_run=dry_run)

    def run(self, parallelism: int = 3, poll_interval: float | None = None,
            poll_backoff: float = 1.5, timeout_hours: int = 24,
            ttl: int | None = None):
        """Run the optimization and return the Ax Client.

        Args:
            parallelism: Max concurrent trials.
            poll_interval: Initial seconds between status polls. If ``None``
                (default), auto-detected: for Python callable metrics, the
                first metric is profiled and the interval is set to
                ``max(2 * eval_time, 0.1)``; for shell commands, defaults to 10s.
            poll_backoff: Backoff factor for poll interval.
            timeout_hours: Max experiment runtime in hours.
            ttl: Time-to-live in seconds per trial (None = no limit).

        Returns:
            The Ax ``Client`` with the completed experiment, or ``None`` on interrupt.
        """
        self._parallelism = parallelism
        self._poll_backoff = poll_backoff
        self._timeout_hours = timeout_hours
        self._ttl = ttl

        # Register Python callables in the metrics module before optimize() runs
        if self._fn_map or self._progress_fn_map:
            from .metrics import _fn_registry, _progress_fn_registry
            _fn_registry.update(self._fn_map)
            _progress_fn_registry.update(self._progress_fn_map)

        # Auto-detect poll interval from metric evaluation time
        if poll_interval is not None:
            self._poll_interval = poll_interval
        elif self._fn_map:
            self._poll_interval = self._profile_poll_interval()
        else:
            self._poll_interval = 10

        # Use NoCasePreprocessor when no case directory is needed
        if self._caseless:
            from .common import NoCasePreprocessor
            from . import metrics as _m
            _m.case_preprocessor = NoCasePreprocessor()

        from .optimize import optimize
        return optimize(self.build())

    @property
    def _caseless(self) -> bool:
        """True when all metrics use Python callables and no case dir is needed."""
        return (self._case is None
                and all(m["name"] in self._fn_map for m in self._metrics)
                and not self._variable_subst
                and not self._file_subst)

    def _profile_poll_interval(self) -> float:
        """Profile the first fn-based metric and return an appropriate poll interval."""
        import time
        # Build a center-of-domain test point
        test_params = {}
        for p in self._parameters:
            if "bounds" in p:
                lo, hi = p["bounds"]
                test_params[p["name"]] = (lo + hi) / 2.0
            elif "values" in p:
                test_params[p["name"]] = p["values"][0]

        fn = next(iter(self._fn_map.values()))
        # Warm up + measure
        try:
            fn(test_params)
            t0 = time.perf_counter()
            for _ in range(3):
                fn(test_params)
            elapsed = (time.perf_counter() - t0) / 3
        except Exception:
            return 1.0  # fallback if profiling fails

        # Poll interval = 2x eval time, clamped to [0.1, 10]
        interval = round(max(0.1, min(10.0, 2 * elapsed)), 2)
        import logging
        logging.getLogger(__name__).info(
            f"Metric eval profiled at {elapsed:.4f}s — poll interval set to {interval}s")
        return interval

    def _to_dict(self) -> dict:
        import os, tempfile
        objective_str = ", ".join(self._objectives) if self._objectives else ""

        # For caseless mode, create a minimal temp dir as the "template case"
        # so FoamJobRunnerOptions.validate_paths() doesn't fail
        case_path = self._case
        if self._caseless:
            case_path = tempfile.mkdtemp(prefix="foambo_caseless_")

        orch: dict[str, Any] = {
            "max_trials": self._max_trials,
            "parallelism": self._parallelism,
            "initial_seconds_between_polls": self._poll_interval,
            "seconds_between_polls_backoff_factor": self._poll_backoff,
            "timeout_hours": self._timeout_hours,
            "global_stopping_strategy": self._global_stop or {
                "min_trials": 10, "window_size": 5, "improvement_bar": 0.1,
            },
            "early_stopping_strategy": self._early_stop,
        }
        if self._ttl is not None:
            orch["ttl_seconds_for_trials"] = self._ttl

        return {
            "experiment": {
                "name": self._name,
                "description": self._description,
                "parameters": self._parameters,
                "parameter_constraints": self._param_constraints,
            },
            "trial_generation": self._trial_gen,
            "existing_trials": {"file_path": self._existing_trials},
            "baseline": {"parameters": self._baseline},
            "optimization": {
                "metrics": self._metrics,
                "objective": objective_str,
                "outcome_constraints": self._outcome_constraints,
                "case_runner": {
                    "template_case": case_path,
                    "mode": self._mode,
                    "runner": self._runner,
                    "log_runner": self._log_runner,
                    "remote_status_query": self._remote_status,
                    "remote_early_stop": self._remote_stop,
                    "trial_destination": self._trials,
                    "artifacts_folder": self._artifacts,
                    "file_substitution": self._file_subst,
                    "variable_substitution": self._variable_subst,
                },
            },
            "orchestration_settings": orch,
            "store": {
                "save_to": self._save_to,
                "read_from": self._read_from,
                "backend_options": {"url": self._backend_url},
            },
            "trial_dependencies": self._dependencies,
            "visualizer": {
                "sensitivity_callback": self._sensitivity_callback,
            },
        }
