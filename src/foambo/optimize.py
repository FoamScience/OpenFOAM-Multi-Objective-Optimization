#!/usr/bin/env python3

""" Perform multi-objective optimization on OpenFOAM cases

We use:
- Meta's Adaptive Experimentation Platform for optimization,
- foamlib for parameter substitution,

Artifacts: CSV data for experiment trials
"""

import sys, argparse, pprint, json

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
            if isinstance(deps_raw, list):
                trial_deps = [TrialDependency.model_validate(dict(d)) for d in deps_raw]
            elif hasattr(deps_raw, 'get') and 'dependencies' in deps_raw:
                trial_deps = [TrialDependency.model_validate(dict(d)) for d in deps_raw['dependencies']]
    else:
        raise TypeError(f"cfg must be a FoamBOConfig or DictConfig, got {type(cfg).__name__}")

    log.info("============= Running Configuration =============")
    from pygments import highlight
    from pygments.lexers import YamlLexer
    from pygments.formatters import Terminal256Formatter
    log.info("\n" + highlight(OmegaConf.to_yaml(raw_cfg), YamlLexer(), Terminal256Formatter(style="native")).rstrip())
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
    runner._metric_names = [m.name for m in opt_cfg.metrics]
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
            if hasattr(strategy, 'metric_threshold') and strategy.metric_threshold:
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
            first_order = sobol.first_order_indices()  # shape: (n_params,) aggregated across outputs
            # Map to param names
            sensitivities = {"_aggregated": {
                pname: float(first_order[i].abs()) for i, pname in enumerate(param_names)
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

    def callback(sched: Orchestrator):
        store_cfg.save(client)
        _maybe_reduce_dimensions()
        streaming_metric(client, raw_cfg["optimization"])
        # Update runner's trial registry for dependency resolution
        runner = client._experiment.runner
        for tidx, trial in client._experiment.trials.items():
            runner.trial_registry[tidx] = {
                "case_path": trial.run_metadata.get("case_path")
                             or trial.run_metadata.get("job", {}).get("case_path"),
                "status": trial.status.name,
                "parameters": trial.arm.parameters if trial.arm else {},
            }
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
    handler = colorlog.StreamHandler()
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
    # Rich tracebacks: syntax-highlighted, with locals, dimmed frames for libraries
    install_rich_traceback(
        show_locals=True,
        width=120,
        suppress=[
            "ax",
            "botorch",
            "torch",
            "gpytorch",
            "pydantic",
        ],
    )


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
    uvx foamBO --docs                                    # Browse the documentation
    uvx foamBO --visualize                               # Run visualization UI for current experiment state""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--analysis', action='store_true', help='Generate optimization reports')
    group.add_argument('--generate-config', action='store_true', help='Generate a default config file and exit')
    group.add_argument('--docs', action='store_true', help='Open Configuration Docs explorer')
    group.add_argument('--visualize', action='store_true', help='Run visualization UI for current experiment state')
    group.add_argument('--preflight-checks', action='store_true', help='Validate configuration before running')
    group.add_argument('-V', '--version', action='version', version='%(prog)s ' + VERSION)
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help=f'Path to config YAML file (optional, default={DEFAULT_CONFIG})')
    parser.add_argument('--json', action='store_true', help='Export Plotly figures as JSON (only valid with --analysis)')
    parser.add_argument('--dry-run', action='store_true', help='Include dry-run checks (only valid with --preflight-checks)')
    parser.add_argument('overrides', nargs=argparse.REMAINDER, help='Config overrides in ++key=value format')
    args = parser.parse_args()

    if args.json and not args.analysis:
        parser.error('--json can only be used with --analysis')
    if args.dry_run and not args.preflight_checks:
        parser.error('--dry-run can only be used with --preflight-checks')

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
            except Exception:
                pass
        if docs is None:
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

    if args.visualize:
        import logging
        log = logging.getLogger(__name__)
        from .config import load_config, override_config
        from .visualize import visualizer_ui
        cfg = load_config(args.config)
        if args.overrides:
            overrides = [o for o in args.overrides if o.startswith('++') or '=' in o]
            if overrides:
                cfg = override_config(cfg, overrides)
        if cfg['store']['read_from'] == "nowhere":
            log.error(f"Cannot perform analysis without reading in client state\n"
                      f"Please set store.read_from option to either `json` or `sql`")
            exit(1)
        visualizer_ui(cfg)
        return

    from .config import load_config, override_config
    cfg = load_config(args.config)
    if args.overrides:
        overrides = [o for o in args.overrides if o.startswith('++') or '=' in o]
        if overrides:
            cfg = override_config(cfg, overrides)
    optimize(cfg)
