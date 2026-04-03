#!/usr/bin/env python3

""" Perform multi-objective optimization on OpenFOAM cases

We use:
- Meta's Adaptive Experimentation Platform for optimization,
- foamlib for parameter substitution,

Artifacts: CSV data for experiment trials
"""

import sys, argparse, pprint, json

def _apply_objective_thresholds(client, expressions: list[str], log=None,
                                baseline_data: dict[str, float] | None = None):
    """Parse and apply objective threshold expressions to the optimization config.

    Expressions like ``"efficiency >= 0.3"`` or ``"torque <= 0.5*baseline"``
    are supported.  Expressions containing ``baseline`` are silently skipped
    when *baseline_data* is ``None`` (they will be resolved after the
    baseline trial completes).
    """
    from ax.core.outcome_constraint import ObjectiveThreshold
    from ax.core.types import ComparisonOp

    opt_config = client._experiment.optimization_config
    if not hasattr(opt_config, 'objective_thresholds'):
        return

    existing = {ot.metric.name: ot for ot in (opt_config._objective_thresholds or [])}

    for expr in expressions:
        expr = expr.strip()
        if "baseline" in expr and baseline_data is None:
            continue  # defer until baseline is available

        if ">=" in expr:
            name, rhs = expr.split(">=", 1)
            op = ComparisonOp.GEQ
        elif "<=" in expr:
            name, rhs = expr.split("<=", 1)
            op = ComparisonOp.LEQ
        else:
            continue

        name = name.strip()
        rhs = rhs.strip()

        if "baseline" in rhs and baseline_data is not None:
            bl_val = baseline_data.get(name)
            if bl_val is None:
                if log:
                    log.warning(f"Objective threshold: no baseline value for '{name}', skipping")
                continue
            bound = float(eval(rhs.replace("baseline", str(bl_val))))
        else:
            bound = float(rhs)

        metric = opt_config.metrics[name]
        existing[name] = ObjectiveThreshold(metric=metric, bound=bound, op=op, relative=False)

    opt_config._objective_thresholds = list(existing.values())
    if log:
        summary = [f"{ot.metric.name} {'>=' if ot.op == ComparisonOp.GEQ else '<='} {ot.bound}"
                   for ot in existing.values()]
        log.info(f"Objective thresholds set: {summary}")


def optimize(cfg):
    """Main optimization loop.

    Colored logging is set up automatically on the first call.

    Args:
        cfg: Either a ``FoamBOConfig`` instance (recommended for programmatic use)
             or an OmegaConf ``DictConfig`` (backward-compatible with YAML loading).

    Returns:
        The Ax ``Client`` with the completed experiment, or ``None`` on interrupt.
    """
    from omegaconf import OmegaConf, DictConfig
    from .common import VERSION, set_experiment_name, get_experiment_name
    from .metrics import streaming_metric
    from .analysis import compute_analysis_cards, plot_pareto_frontier
    from .orchestrate import (
        ExistingTrialsOptions, ExperimentOptions, OptimizationOptions,
        ConfigOrchestratorOptions, StoreOptions, TrialGenerationOptions, BaselineOptions,
        FoamBOConfig, TrialDependency,
    )
    from ax.api.client import MultiObjective, Orchestrator, db_settings_from_storage_config
    import pandas as pd
    from logging import Logger
    from ax.utils.common.logger import get_logger
    # Must run after Ax imports — Ax adds its own handlers that override ours
    setup_colored_logging()
    log = get_logger(__name__)

    # Accept FoamBOConfig or DictConfig
    if isinstance(cfg, FoamBOConfig):
        foambo_cfg = cfg
        # Keep a DictConfig for backward-compatible dict access in callbacks
        raw_cfg = cfg.to_dictconfig()
        exp_cfg = cfg.experiment
        gs_cfg = cfg.trial_generation
        opt_cfg = cfg.optimization
        orch_cfg = cfg.orchestration_settings
        store_cfg = cfg.store
        baseline_cfg = cfg.baseline
        trial_deps = cfg.trial_dependencies
    elif isinstance(cfg, DictConfig):
        raw_cfg = cfg
        foambo_cfg = None
        exp_cfg = ExperimentOptions.model_validate(dict(cfg['experiment']))
        gs_cfg = TrialGenerationOptions.model_validate(dict(cfg['trial_generation']))
        opt_cfg = OptimizationOptions.model_validate(dict(cfg['optimization']))
        orch_cfg = ConfigOrchestratorOptions.model_validate(dict(cfg['orchestration_settings']))
        store_cfg = StoreOptions.model_validate(dict(cfg['store']))
        baseline_cfg = BaselineOptions.model_validate(dict(cfg['baseline']))
        trial_deps = []
        if 'trial_dependencies' in cfg:
            deps_raw = cfg['trial_dependencies']
            if deps_raw is not None:
                trial_deps = [TrialDependency.model_validate(dict(d)) for d in deps_raw]
    else:
        raise TypeError(f"cfg must be a FoamBOConfig or DictConfig, got {type(cfg).__name__}")

    log.info("============= Running Configuration =============")
    #from pygments import highlight
    #from pygments.lexers import YamlLexer
    #from pygments.formatters import Terminal256Formatter
    #log.info("\n" + highlight(OmegaConf.to_yaml(raw_cfg), YamlLexer(), Terminal256Formatter(style="native")).rstrip())
    log.info("=================================================")

    # Apply user-configurable subprocess timeouts
    from . import metrics as _m
    _m.METRIC_EVAL_TIMEOUT = orch_cfg.metric_eval_timeout
    _m.REMOTE_QUERY_TIMEOUT = orch_cfg.remote_query_timeout
    _m.PROGRESSION_CMD_TIMEOUT = orch_cfg.progression_cmd_timeout
    _m.DEPENDENCY_ACTION_TIMEOUT = orch_cfg.dependency_action_timeout
    _m.PROCESS_REAP_TIMEOUT = orch_cfg.process_reap_timeout
    # Embed the full config in the JSON state for FoamBO.load()
    store_cfg._foambo_config = OmegaConf.to_container(raw_cfg, resolve=True)
    client = store_cfg.load()

    # 1.0 Experiment setup
    has_experiment = False
    try:
        _ = client._experiment
        has_experiment = True
    except:
        pass

    if not has_experiment:
        client.configure_experiment(**exp_cfg.to_dict())

    ## 2.0 Trial generation
    gs_cfg.set_generation_strategy(client)
    log.info(f"Configured trial generation strategy: {client._generation_strategy.name}")
    log.info("=================================================")

    ## 3.0 Optimization setup
    if not has_experiment and not client._experiment.immutable_search_space_and_opt_config:
        client.configure_optimization(**opt_cfg.to_optimization_dict())
        client.configure_metrics(**opt_cfg.to_objective_metrics_dict())
        client.configure_metrics(**opt_cfg.to_tracking_metrics_dict())
        # Apply explicit objective thresholds for multi-objective (prevents Ax inference crash)
        if opt_cfg.objective_thresholds:
            _apply_objective_thresholds(client, opt_cfg.objective_thresholds, log=log)
    log.info("=================================================")
    log.info(f"Objectives:\n%s", pprint.pformat(client._experiment.optimization_config._objective))
    if client._experiment._tracking_metrics:
        log.info(f"Tracking metrics:\n%s", pprint.pformat(client._experiment._tracking_metrics))
        log.info("=================================================")
    client.configure_runner(**opt_cfg.to_runner_dict())
    # Wire trial dependencies and metric names onto the runner
    runner = client._experiment.runner
    if trial_deps:
        runner.trial_dependencies = trial_deps
    runner._parameter_groups = exp_cfg.get_parameter_groups()
    runner._metric_names = [m.name for m in opt_cfg.metrics]
    # Enable streaming data attachment during poll_trial (for early stopping)
    runner._streaming_client = client
    runner._streaming_cfg = raw_cfg["optimization"]
    # Inject custom kernel surrogate spec if set via .kernel() API
    if hasattr(cfg, '_kernel_surrogate_spec') and cfg._kernel_surrogate_spec is not None:
        gs = client._generation_strategy
        from ax.adapter.registry import Generators
        for node in gs._nodes:
            for spec in node.generator_specs:
                if spec.generator_enum == Generators.BOTORCH_MODULAR:
                    spec.generator_kwargs["surrogate_spec"] = cfg._kernel_surrogate_spec
                    log.info(f"Custom kernel set on generation node '{node.name}'")

    client.set_early_stopping_strategy(orch_cfg.early_stopping_strategy)

    # Risk 7: Warn if early stopping references objective metrics (they don't stream)
    if orch_cfg.early_stopping_strategy is not None:
        objective_names = {s.strip().lstrip('+-') for s in opt_cfg.objective.split(',') if s.strip()}
        streaming_names = {m.name for m in opt_cfg.metrics
                           if m.name not in objective_names and m.progress and m.progress != "none"}
        def _collect_es_metric_names(strategy):
            names = set()
            if strategy is None:
                return names
            if hasattr(strategy, 'metric_names') and strategy.metric_names:
                names.update(strategy.metric_names)
            if hasattr(strategy, 'metric_threshold') and isinstance(strategy.metric_threshold, dict):
                names.update(strategy.metric_threshold.keys())
            if hasattr(strategy, 'left'):
                names.update(_collect_es_metric_names(strategy.left))
            if hasattr(strategy, 'right'):
                names.update(_collect_es_metric_names(strategy.right))
            return names
        es_metric_names = _collect_es_metric_names(orch_cfg.early_stopping_strategy)
        obj_in_es = es_metric_names & objective_names
        if obj_in_es:
            log.warning(f"Early stopping strategy references objective metric(s) {obj_in_es}, "
                        f"but only non-objective metrics with a 'progress' command stream intermediate data. "
                        f"Early stopping will NEVER trigger for these metrics.")
        no_stream = es_metric_names - objective_names - streaming_names
        if no_stream:
            log.warning(f"Early stopping strategy references metric(s) {no_stream} that have no "
                        f"'progress' command configured. Early stopping will not trigger for these.")

    data_attacher = ExistingTrialsOptions.model_validate(dict(raw_cfg["existing_trials"]))
    data_attacher.load_data(client)

    _dim_reduction_done = False
    _dim_reduction_attempts = 0

    def _maybe_reduce_dimensions():
        nonlocal _dim_reduction_done, _dim_reduction_attempts
        if _dim_reduction_done:
            return
        dr = orch_cfg.dimensionality_reduction
        if not dr.enabled:
            return
        exp = client._experiment
        from ax.core.base_trial import TrialStatus as AxTrialStatus
        n_completed = sum(1 for t in exp.trials.values() if t.status == AxTrialStatus.COMPLETED)
        if n_completed < dr.after_trials:
            return
        gs = client._generation_strategy
        if gs is None or gs.adapter is None:
            return

        try:
            import torch
            import numpy as _np
            from ax.utils.sensitivity.sobol_measures import SobolSensitivityGPMean
            from ax.adapter.cross_validation import cross_validate as ax_cv

            # Check model fit quality via cross-validation before trusting Sobol
            try:
                cv_results = ax_cv(adapter=gs.adapter)
                cv_errors = []
                for r in cv_results:
                    for mname, obs_mean in r.observed.data.means_dict.items():
                        pred_mean = r.predicted.means_dict.get(mname, obs_mean)
                        cv_errors.append(abs(obs_mean - pred_mean))
                if cv_errors:
                    obs_range = max(abs(e) for e in cv_errors)
                    mean_cv_error = _np.mean(cv_errors)
                    # If mean CV error > 50% of error range, model fit is poor
                    if obs_range > 0 and mean_cv_error / obs_range > 0.5:
                        log.debug(f"Dimensionality reduction deferred — model fit too poor "
                                  f"(CV error ratio: {mean_cv_error / obs_range:.2f})")
                        return
            except Exception:
                pass  # CV failed — proceed with Sobol anyway

            surr = gs.adapter.generator.surrogate
            model = surr.model
            # Build bounds tensor from search space (normalized [0,1] by Ax)
            param_names = [p for p in exp.search_space.parameters
                           if hasattr(exp.search_space.parameters[p], 'lower')]
            n_params = len(param_names)
            bounds = torch.zeros(2, n_params, dtype=torch.float64)
            bounds[1] = 1.0  # model operates in normalized space
            sobol = SobolSensitivityGPMean(model=model, bounds=bounds, num_mc_samples=1000)
            total_order = sobol.total_order_indices()  # includes interactions with other params
            # Map to param names
            sensitivities = {"_aggregated": {
                pname: float(total_order[i].abs()) for i, pname in enumerate(param_names)
            }}
        except Exception as e:
            _dim_reduction_attempts += 1
            if _dim_reduction_attempts >= 3:
                log.warning(f"Dimensionality reduction disabled after 3 failed attempts: {e}")
                _dim_reduction_done = True
            else:
                log.debug(f"Dimensionality reduction attempt {_dim_reduction_attempts} failed: {e}")
            return

        # Aggregate importance across metrics (max importance across all objectives)
        param_names = list(exp.search_space.parameters.keys())
        importance = {}
        for pname in param_names:
            max_imp = 0.0
            for metric_sens in sensitivities.values():
                max_imp = max(max_imp, abs(metric_sens.get(pname, 0.0)))
            importance[pname] = max_imp

        # Sort by importance, find parameters to fix
        sorted_params = sorted(importance.items(), key=lambda x: x[1])
        max_fixable = min(int(len(param_names) * dr.max_fix_fraction), len(param_names) - 1)
        to_fix = [(name, imp) for name, imp in sorted_params if imp < dr.min_importance]
        to_fix = to_fix[:max_fixable]

        if not to_fix:
            log.info("Dimensionality reduction: all parameters are above the importance threshold")
            _dim_reduction_done = True
            return

        # Determine fix values
        if dr.fix_at == "best":
            try:
                best_params, _, _, _ = client.get_best_parameterization(use_model_predictions=False)
            except Exception:
                best_params = {p: exp.search_space.parameters[p].lower +
                               (exp.search_space.parameters[p].upper - exp.search_space.parameters[p].lower) / 2
                               for p in param_names if hasattr(exp.search_space.parameters[p], 'lower')}

        from ax.core.parameter import FixedParameter
        for pname, imp in to_fix:
            param = exp.search_space.parameters[pname]
            if dr.fix_at == "best":
                fix_val = best_params.get(pname)
            else:  # center
                if hasattr(param, 'lower') and hasattr(param, 'upper'):
                    fix_val = (param.lower + param.upper) / 2
                elif hasattr(param, 'values'):
                    fix_val = param.values[0]
                else:
                    continue
            if fix_val is None:
                continue
            fixed = FixedParameter(name=pname, parameter_type=param.parameter_type, value=fix_val)
            exp.search_space.update_parameter(fixed)
            log.info(f"Dimensionality reduction: fixed '{pname}' = {fix_val} (importance={imp:.4f})")

        log.info(f"Dimensionality reduction: {len(to_fix)} parameter(s) fixed, "
                 f"{len(param_names) - len(to_fix)} remaining active")
        _dim_reduction_done = True

    _reported_failures: set[int] = set()

    # API server update function — no-op until server starts
    _update_api = lambda client: None
    if orch_cfg.api_port != 0:
        from .api_server import start_api_server, update_api_state
        _api_thread = start_api_server(
            client=client,
            raw_cfg=raw_cfg,
            orch_cfg=orch_cfg,
            host=orch_cfg.api_host,
            port=orch_cfg.api_port,
        )
        _update_api = update_api_state

    def _log_trial_failure(trial_index: int, case_path: str | None, log: Logger):
        """Log the tail of a failed trial's runner log."""
        if not case_path:
            log.warning(f"Trial {trial_index} FAILED (no case path available)")
            return
        import os
        log_path = os.path.join(case_path, "log.runner")
        tail = ""
        if os.path.isfile(log_path):
            try:
                with open(log_path) as f:
                    lines = f.readlines()
                tail = "".join(lines[-20:]).rstrip()
            except Exception:
                pass
        if tail:
            log.warning(f"Trial {trial_index} FAILED — last lines of {log_path}:\n{tail}")
        else:
            log.warning(f"Trial {trial_index} FAILED — case: {case_path} (no log.runner found)")

    def callback(sched: Orchestrator):
        store_cfg.save(client)
        _maybe_reduce_dimensions()
        streaming_metric(client, raw_cfg["optimization"])
        # Update runner's trial registry for dependency resolution
        runner = client._experiment.runner
        from ax.core.base_trial import TrialStatus as _TS
        for tidx, trial in client._experiment.trials.items():
            case_path = (trial.run_metadata.get("case_path")
                         or trial.run_metadata.get("job", {}).get("case_path"))
            runner.trial_registry[tidx] = {
                "case_path": case_path,
                "status": trial.status.name,
                "parameters": trial.arm.parameters if trial.arm else {},
            }
            if trial.status == _TS.FAILED and tidx not in _reported_failures:
                _reported_failures.add(tidx)
                _log_trial_failure(tidx, case_path, log)
        # Refresh API server state
        if orch_cfg.api_port != 0:
            _update_api(client)
        from .analysis import plot_streaming_metrics
        plot_streaming_metrics(raw_cfg, client)

        data = client._experiment.fetch_data()
        if data.df.empty:
            return
        df = data.df.copy()
        metadf = {
            trial_index: {
                "job_id": trial.run_metadata.get("job_id"),
                "generation_node": trial.generator_run.generator_run_type or "",
                "status": trial.status.__repr__().strip("<enum 'TrialStatus'>."),
                **client._experiment.trials[trial_index].arm.parameters,
                "case_path": trial.run_metadata.get("job", {}).get("case_path"),
            }
            for trial_index, trial in client._experiment.trials.items()
        }
        if metadf:
            for key in metadf[0].keys():
                df[key] = df["trial_index"].map(lambda x: metadf.get(x, {}).get(key))
        df.to_csv(
            f"{raw_cfg["optimization"]["case_runner"]["artifacts_folder"]}/{raw_cfg["experiment"]["name"]}_report.csv",
            index=False
        )

    if not has_experiment and baseline_cfg.parameters:
        log.info("=============== Running Baseline ================")
        baseline_index = 0 if len(client._experiment._trials) == 0 else max(client._experiment._trials.keys()) + 1
        from .orchestrate import ManualGenerationNode
        from ax.generation_strategy.generation_strategy import GenerationStrategy
        from ax.core.trial import Trial
        from pyre_extensions import assert_is_instance
        scheduler = Orchestrator(
            experiment=client._experiment,
            generation_strategy=GenerationStrategy(
                name="baseline",
                nodes=[ManualGenerationNode(node_name="baseline", parameters=baseline_cfg.parameters)]),
            options=orch_cfg.to_scheduler_options(),
            db_settings=db_settings_from_storage_config(client._storage_config)
            if client._storage_config is not None
            else None,
        )
        scheduler.run_n_trials(max_trials=1, timeout_hours=orch_cfg.timeout_hours, idle_callback=callback)
        client._experiment.status_quo = assert_is_instance(
            client._experiment.trials[baseline_index], Trial
        ).arm
        # Resolve baseline-relative objective thresholds now that baseline data exists
        if opt_cfg.objective_thresholds and any("baseline" in e for e in opt_cfg.objective_thresholds):
            data = client._experiment.fetch_data().df
            sq_name = client._experiment.status_quo.name
            bl_data = {row["metric_name"]: row["mean"]
                       for _, row in data[data["arm_name"] == sq_name].iterrows()}
            _apply_objective_thresholds(client, opt_cfg.objective_thresholds,
                                       log=log, baseline_data=bl_data)

    scheduler = Orchestrator(
        experiment=client._experiment,
        generation_strategy=client._generation_strategy_or_choose(),
        options=orch_cfg.to_scheduler_options(),
        db_settings=db_settings_from_storage_config(client._storage_config)
        if client._storage_config is not None
        else None,
    )
    def _stop_running_trials(sched: Orchestrator):
        """Stop all running trials and save state on interrupt."""
        from ax.core.base_trial import TrialStatus as AxTrialStatus
        running = [t for t in client._experiment.trials.values()
                   if t.status == AxTrialStatus.RUNNING]
        if running:
            log.warning(f"Stopping {len(running)} running trial(s)...")
            for trial in running:
                try:
                    trial.stop(new_status=AxTrialStatus.ABANDONED, reason="KeyboardInterrupt")
                except Exception as e:
                    log.error(f"Failed to stop trial {trial.index}: {e}")
        store_cfg.save(client)

    log.info("============ Running optimization ===============")
    try:
        scheduler.run_all_trials(timeout_hours=orch_cfg.timeout_hours, idle_callback=callback)
    except KeyboardInterrupt:
        log.warning("Interrupted — cleaning up running trials...")
        _stop_running_trials(scheduler)
        if orch_cfg.api_port != 0:
            from .api_server import stop_api_server
            stop_api_server()
        log.info("State saved. Exiting.")
        return
    store_cfg.save(client)

    log.info("=========== Best set of parameters ==============")
    best_parameters, prediction = None, None
    if isinstance(client._experiment.optimization_config._objective, MultiObjective):
        try:
            front = client.get_pareto_frontier(use_model_predictions=True)
        except Exception:
            front = client.get_pareto_frontier(use_model_predictions=False)
        for best_parameters, prediction, _, _ in front:
            log.info("Pareto-frontier configuration:\n%s", json.dumps(best_parameters, indent=2))
            log.info("Predictions for Pareto-frontier configuration (mean, variance):\n%s", json.dumps(prediction, indent=2))
        try:
            _ = plot_pareto_frontier(raw_cfg, client, front, open_html=False)
        except Exception as e:
            log.warning(f"Pareto frontier plot skipped: {e}")
    else:
        best_parameters, prediction, _, _ = client.get_best_parameterization(use_model_predictions=True)
        log.info("Best parameter set:\n%s", json.dumps(best_parameters, indent=2))
        log.info("Predictions for best parameter set (mean, variance):\n%s", json.dumps(prediction, indent=2))

    log.info("=================================================")
    cards = compute_analysis_cards(raw_cfg, client, open_html=False)

    store_cfg.save(client, cards)

    if orch_cfg.api_port != 0:
        from .api_server import stop_api_server
        stop_api_server()
    log.info("==================== End ========================")
    return client

def setup_colored_logging():
    import logging
    import colorlog
    from rich.traceback import install as install_rich_traceback
    fmt = colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)-5s %(asctime)s]%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    )
    import sys
    handler = colorlog.StreamHandler(stream=sys.stdout)
    handler.setFormatter(fmt)
    # Clear all existing handlers to avoid duplicates, set colored handler on root only
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    # Strip handlers from child loggers so they propagate to root
    for name in list(logging.Logger.manager.loggerDict):
        lgr = logging.getLogger(name)
        lgr.handlers.clear()
        lgr.propagate = True
    # Rich tracebacks: syntax-highlighted, dimmed frames for libraries
    from rich.console import Console
    install_rich_traceback(
        console=Console(file=sys.stdout, force_terminal=True),
        show_locals=False,
        width=120,
        max_frames=6,
        suppress=[
            "ax",
            "botorch",
            "torch",
            "gpytorch",
            "pydantic",
        ],
    )
    # Suppress foamlib's Rich progress bars (they clutter the log output)
    try:
        from foamlib._cases._run import FoamCaseRunBase
        from contextlib import contextmanager

        class _NoOpProgress:
            """Drop-in replacement for Rich Progress that does nothing."""
            def add_task(self, *a, **kw): return 0
            def update(self, *a, **kw): pass

        @contextmanager
        def _noop_ctx():
            yield _NoOpProgress()

        class _NoOpSingleton:
            """Replaces SingletonContextManager(Progress) with a no-op."""
            def __enter__(self): self._p = _NoOpProgress(); return self._p
            def __exit__(self, *a): pass

        FoamCaseRunBase._FoamCaseRunBase__progress = _NoOpSingleton()
    except Exception:
        pass


def main():
    setup_colored_logging()
    from ._version import VERSION, DEFAULT_CONFIG

    parser = argparse.ArgumentParser(
        description="Multi-objective optimization with Bayesian Algorithms for OpenFOAM cases.",
        epilog="""Examples:
    uvx foamBO --generate-config                         # Generates a default config file: foamBO.yaml
    uvx foamBO --generate-config --config My.yaml        # Generates a default config file: My.yaml
    uvx foamBO --config My.yaml                          # Runs optimization from My.yaml
    uvx foamBO --config My.yaml ++experiment.name=Test2  # Runs optimization from My.yaml with different experiment name
    uvx foamBO --analysis ++store.read_from=json         # Generates reports for optimization from foamBO.yaml configuration
    uvx foamBO --analysis --json                         # Generates reports and exports Plotly figures as JSON
    uvx foamBO --docs                                    # Browse the documentation""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--analysis', action='store_true', help='Generate optimization reports')
    group.add_argument('--generate-config', action='store_true', help='Generate a default config file and exit')
    group.add_argument('--docs', action='store_true', help='Open Configuration Docs explorer')
    group.add_argument('--no-opt', action='store_true', help='Load experiment and start web dashboard without running optimization')
    group.add_argument('--upgrade-config', action='store_true', help='Check config YAML against current schema and show suggested changes')
    group.add_argument('--pack', action='store_true', help='Pack experiment into a .foambo archive')
    group.add_argument('--unpack', type=str, metavar='ARCHIVE', help='Unpack a .foambo archive')
    group.add_argument('--preflight-checks', action='store_true', help='Validate configuration before running')
    group.add_argument('-V', '--version', action='version', version='%(prog)s ' + VERSION)
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help=f'Path to config YAML file (optional, default={DEFAULT_CONFIG})')
    parser.add_argument('--json', action='store_true', help='Export Plotly figures as JSON (only valid with --analysis)')
    parser.add_argument('--dry-run', action='store_true', help='Include dry-run checks (only valid with --preflight-checks)')
    parser.add_argument('--no-ui', action='store_true', help='Disable the web UI / REST API server')
    parser.add_argument('--include-trials', type=str, default=None,
        help='Trials to include in --pack: best, pareto, all, or comma-separated indices')
    parser.add_argument('--skip-patterns', type=str, default=None,
        help='Comma-separated glob patterns to exclude from --pack (e.g. "processor*/,postProcessing/")')
    parser.add_argument('overrides', nargs=argparse.REMAINDER, help='Config overrides in ++key=value format')
    args = parser.parse_args()

    if args.json and not args.analysis:
        parser.error('--json can only be used with --analysis')
    if args.dry_run and not args.preflight_checks:
        parser.error('--dry-run can only be used with --preflight-checks')
    if getattr(args, 'no_opt', False) and args.no_ui:
        parser.error('--no-opt and --no-ui cannot be used together')
    if args.include_trials and not getattr(args, 'pack', False):
        parser.error('--include-trials can only be used with --pack')
    if args.skip_patterns and not getattr(args, 'pack', False):
        parser.error('--skip-patterns can only be used with --pack')

    if args.upgrade_config:
        from .config_upgrade import run_upgrade_check
        ok = run_upgrade_check(args.config)
        sys.exit(0 if ok else 1)

    if args.pack:
        from .archive import pack
        skip = args.skip_patterns.split(",") if args.skip_patterns else None
        pack(args.config, include_trials=args.include_trials, skip_patterns=skip)
        return

    if args.unpack:
        from .archive import unpack
        unpack(args.unpack)
        return

    if args.preflight_checks:
        from .config import load_config, override_config
        from .preflight import run_preflight
        cfg = load_config(args.config)
        if args.overrides:
            overrides = [o for o in args.overrides if o.startswith('++') or '=' in o]
            if overrides:
                cfg = override_config(cfg, overrides)
        ok = run_preflight(cfg, dry_run=args.dry_run)
        sys.exit(0 if ok else 1)

    if args.docs:
        import pathlib
        from .docs import run_docs_tui
        cache_dir = pathlib.Path.home() / ".cache" / "foambo"
        cache_file = cache_dir / f"docs_v{VERSION}.json"
        docs = None
        if cache_file.exists():
            try:
                docs = json.loads(cache_file.read_text())
                print(f"Loading cached docs (v{VERSION})")
            except Exception:
                pass
        if docs is None:
            print(f"Generating docs (v{VERSION})...")
            from .default_config import get_config_docs
            docs = get_config_docs()
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(docs))
            except Exception:
                pass
        run_docs_tui(docs)
        sys.exit(0)

    if args.generate_config:
        from .config import save_default_config
        save_default_config(args.config)
        sys.exit(0)

    if args.analysis:
        import logging
        log = logging.getLogger(__name__)
        from .config import load_config, override_config
        from .analysis import compute_analysis_cards, plot_pareto_frontier
        from .orchestrate import StoreOptions
        cfg = load_config(args.config)
        if args.overrides:
            overrides = [o for o in args.overrides if o.startswith('++') or '=' in o]
            if overrides:
                cfg = override_config(cfg, overrides)
        if cfg['store']['read_from'] == "nowhere":
            log.error(f"Cannot perform analysis without reading in client state\n"
                      f"Please set store.read_from option to either `json` or `sql`")
            exit(1)
        compute_analysis_cards(cfg, None, export_json=args.json)
        _ = plot_pareto_frontier(cfg, client=StoreOptions.model_validate(dict(cfg['store'])).load(), open_html=False, export_json=args.json)
        return

    if args.no_opt:
        import logging
        log = logging.getLogger(__name__)
        from .config import load_config, override_config
        from .common import set_experiment_name
        from .orchestrate import StoreOptions, ConfigOrchestratorOptions
        cfg = load_config(args.config)
        if args.overrides:
            overrides = [o for o in args.overrides if o.startswith('++') or '=' in o]
            if overrides:
                cfg = override_config(cfg, overrides)
        if cfg['store']['read_from'] == "nowhere":
            log.error("Cannot load experiment without client state\n"
                      "Please set store.read_from option to either `json` or `sql`")
            exit(1)
        set_experiment_name(cfg["experiment"]["name"])
        client = StoreOptions.model_validate(dict(cfg['store'])).load()
        orch_cfg = ConfigOrchestratorOptions.model_validate(dict(cfg['orchestration_settings']))
        from .api_server import start_api_server, stop_api_server
        thread = start_api_server(
            client=client, raw_cfg=cfg, orch_cfg=orch_cfg,
            host=orch_cfg.api_host, port=orch_cfg.api_port,
        )
        if thread is None:
            log.error("API server failed to start")
            exit(1)
        log.info("Dashboard running (no optimization). Press Ctrl+C to stop.")
        try:
            thread.join()
        except KeyboardInterrupt:
            stop_api_server()
            log.info("Stopped.")
        return

    from .config import load_config, override_config
    cfg = load_config(args.config)
    if args.overrides:
        overrides = [o for o in args.overrides if o.startswith('++') or '=' in o]
        if overrides:
            cfg = override_config(cfg, overrides)
    if args.no_ui:
        from omegaconf import OmegaConf
        OmegaConf.update(cfg, "orchestration_settings.api_port", 0)
    optimize(cfg)
