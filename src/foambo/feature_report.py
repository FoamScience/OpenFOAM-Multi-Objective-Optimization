#!/usr/bin/env python3

"""Incremental feature usage report for foamBO optimization runs.

Provides a ``FeatureReporter`` that is instantiated once before the
optimization loop and called from the idle callback after every trial.
Each call rewrites a detailed Markdown report to disk.
"""

from __future__ import annotations

import os
from datetime import datetime
from ax.api.client import Client, MultiObjective
from ax.core.base_trial import TrialStatus as AxTrialStatus
from ax.core.parameter import FixedParameter, RangeParameter
from ax.utils.common.logger import get_logger
from omegaconf import DictConfig

log = get_logger(__name__)


class FeatureReporter:
    """Accumulates feature-usage events and writes a Markdown report each update.

    Typical usage inside ``optimize()``::

        reporter = FeatureReporter(raw_cfg, artifacts_folder, experiment_name)

        def callback(sched):
            ...
            reporter.update(client)
    """

    def __init__(self, cfg: DictConfig | dict, artifacts_folder: str, experiment_name: str):
        if isinstance(cfg, dict):
            cfg = DictConfig(cfg)
        self._cfg = cfg
        self._artifacts = artifacts_folder
        self._name = experiment_name
        self._path = os.path.join(artifacts_folder, f"{experiment_name}_feature_report.md")

        # Snapshot of initial search-space parameter names (before dim reduction)
        self._original_param_names: list[str] | None = None

        # Incremental tracking state
        self._seen_early_stopped: set[int] = set()
        self._seen_completed: set[int] = set()
        self._trial_completions: dict[int, float] = {}  # trial_index -> completion timestamp
        self._dim_reduction_events: list[str] = []
        self._best_trace: list[dict] = []  # [{trial: int, metrics: {name: val}}]
        self._events: list[str] = []  # timestamped event log
        self._dep_resolutions: list[dict] = []  # per-trial dependency outcomes

        # Parse config once to know what is configured
        self._es_configured = _is_set(cfg, "orchestration_settings", "early_stopping_strategy")
        self._gs_configured = _is_set(cfg, "orchestration_settings", "global_stopping_strategy")
        dr = _get(cfg, "orchestration_settings", "dimensionality_reduction")
        self._dr_configured = dr is not None and _field(dr, "enabled", False)
        self._dr_after = _field(dr, "after_trials", 0) if dr else 0
        self._dr_min_importance = _field(dr, "min_importance", 0.05) if dr else 0.05
        self._dr_fix_at = _field(dr, "fix_at", "best") if dr else "best"
        self._dr_max_frac = _field(dr, "max_fix_fraction", 0.5) if dr else 0.5
        deps = _get(cfg, "trial_dependencies")
        self._deps = list(deps) if deps is not None else []
        self._deps_configured = len(self._deps) > 0
        baseline = _get(cfg, "baseline")
        bp = _field(baseline, "parameters", None) if baseline else None
        self._baseline_configured = bp is not None and bool(bp)
        self._parallelism = _get(cfg, "orchestration_settings", "parallelism") or 1
        self._max_trials = _get(cfg, "orchestration_settings", "max_trials") or 0

        os.makedirs(artifacts_folder, exist_ok=True)
        self._event("Reporter initialised")

    def update(self, client: Client) -> None:
        """Re-scan experiment state and rewrite the report."""
        try:
            exp = client._experiment
            if self._original_param_names is None:
                self._original_param_names = list(exp.search_space.parameters.keys())
            self._detect_completions(exp)
            self._detect_early_stopping(exp)
            self._detect_dim_reduction(exp)
            self._detect_dep_resolutions(exp)
            self._track_best(exp, client)
            self._write(client)
        except Exception:
            log.debug("Feature report update skipped", exc_info=True)

    @property
    def path(self) -> str:
        return self._path

    def to_dict(self) -> dict:
        """Return serializable state for embedding in the client JSON."""
        return {
            "seen_early_stopped": sorted(self._seen_early_stopped),
            "seen_completed": sorted(self._seen_completed),
            "trial_completions": {str(k): v for k, v in self._trial_completions.items()},
            "dim_reduction_events": self._dim_reduction_events,
            "best_trace": self._best_trace,
            "events": self._events,
            "original_param_names": self._original_param_names,
            "dep_resolutions": self._dep_resolutions,
        }

    def restore(self, state: dict) -> None:
        """Restore incremental state from a previously saved dict."""
        if not state:
            return
        self._seen_early_stopped = set(state.get("seen_early_stopped", []))
        self._seen_completed = set(state.get("seen_completed", []))
        self._trial_completions = {int(k): v for k, v in state.get("trial_completions", {}).items()}
        self._dim_reduction_events = state.get("dim_reduction_events", [])
        self._best_trace = state.get("best_trace", [])
        self._events = state.get("events", [])
        self._original_param_names = state.get("original_param_names")
        self._dep_resolutions = state.get("dep_resolutions", [])
        self._event("Reporter restored from saved state")

    def _detect_completions(self, exp) -> None:
        """Track when trials complete to measure durations and idle time."""
        import time
        now = time.time()
        for idx, trial in exp.trials.items():
            if idx in self._seen_completed:
                continue
            if trial.status in (AxTrialStatus.COMPLETED, AxTrialStatus.EARLY_STOPPED, AxTrialStatus.FAILED):
                self._seen_completed.add(idx)
                self._trial_completions[idx] = now

    def _detect_early_stopping(self, exp) -> None:
        for idx, trial in exp.trials.items():
            if trial.status == AxTrialStatus.EARLY_STOPPED and idx not in self._seen_early_stopped:
                self._seen_early_stopped.add(idx)
                step_info = ""
                try:
                    data = trial.lookup_data().df
                    if not data.empty and "step" in data.columns:
                        max_step = int(data["step"].max())
                        step_info = f" at progression step {max_step}"
                except Exception:
                    pass
                self._event(f"Trial {idx} early-stopped{step_info}")

    def _detect_dim_reduction(self, exp) -> None:
        if not self._dr_configured or self._original_param_names is None:
            return
        current_fixed = {
            p.name: p.value
            for p in exp.search_space.parameters.values()
            if isinstance(p, FixedParameter) and p.name in self._original_param_names
        }
        already_reported = {pname for pname in current_fixed if any(f"'{pname}'" in e for e in self._dim_reduction_events)}
        for pname, val in current_fixed.items():
            if pname not in already_reported:
                msg = f"Parameter '{pname}' fixed to {val}"
                self._dim_reduction_events.append(msg)
                self._event(f"Dim reduction: {msg}")

    def _detect_dep_resolutions(self, exp) -> None:
        """Extract per-trial dependency outcomes from run_metadata."""
        if not self._deps_configured:
            return
        seen_indices = {r["trial"] for r in self._dep_resolutions}
        for idx, trial in exp.trials.items():
            if idx in seen_indices:
                continue
            meta = trial.run_metadata or {}
            deps_meta = meta.get("dependencies")
            if deps_meta is None:
                continue
            for dep_name, info in deps_meta.items():
                if dep_name.startswith("_"):
                    continue  # skip internal keys like _hook_env
                source_index = info.get("source_trial_index", "?")
                source_path = info.get("source_case_path", "?")
                actions = info.get("actions_applied", [])
                phased = info.get("phased_actions", [])
                self._dep_resolutions.append({
                    "trial": idx,
                    "dependency": dep_name,
                    "source_index": source_index,
                    "source_path": source_path,
                    "actions_applied": actions,
                    "phased_actions": phased,
                })
                self._event(f"Trial {idx}: dependency '{dep_name}' resolved from trial {source_index}")

    def _track_best(self, exp, client: Client) -> None:
        """Record the current best objective value(s)."""
        try:
            opt_config = exp.optimization_config
            if opt_config is None:
                return
            data = exp.lookup_data().df
            if data.empty:
                return
            is_mo = isinstance(opt_config.objective, MultiObjective)
            objectives = (
                opt_config.objective.objectives if is_mo
                else [opt_config.objective]
            )
            n_completed = sum(1 for t in exp.trials.values() if t.status == AxTrialStatus.COMPLETED)
            metrics: dict[str, float] = {}
            for obj in objectives:
                mname = obj.metric.name
                vals = data.loc[data["metric_name"] == mname, "mean"]
                if vals.empty:
                    continue
                metrics[mname] = float(vals.min() if obj.minimize else vals.max())
            if metrics:
                entry = {"trial": n_completed, "metrics": metrics}
                if not self._best_trace or self._best_trace[-1]["metrics"] != metrics:
                    self._best_trace.append(entry)
        except Exception:
            pass

    def _write(self, client: Client) -> None:
        exp = client._experiment
        n_total = len(exp.trials)
        n_completed = sum(1 for t in exp.trials.values() if t.status == AxTrialStatus.COMPLETED)
        n_failed = sum(1 for t in exp.trials.values() if t.status == AxTrialStatus.FAILED)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        L: list[str] = []
        budget = f" / {self._max_trials}" if self._max_trials else ""

        L.append(f"# Feature Usage Report -- {self._name}")
        L.append("")
        L.append(f"> Updated after trial **{n_completed}{budget}** | "
                 f"Completed: {n_completed} | Failed: {n_failed} | "
                 f"Total: {n_total} | {now}")
        L.append("")

        L.extend(self._section_early_stopping(exp))
        L.extend(self._section_global_stopping(exp))
        L.extend(self._section_dim_reduction(exp))
        L.extend(self._section_trial_dependencies(exp))
        L.extend(self._section_baseline(exp, client))
        L.extend(self._section_parallelism())
        L.extend(self._section_custom_kernel(client))
        L.extend(self._section_existing_trials(exp))
        L.extend(self._section_efficiency(exp))
        L.extend(self._section_best_progress())
        L.extend(self._section_event_log())

        with open(self._path, "w") as f:
            f.write("\n".join(L) + "\n")

    # -- individual sections -------------------------------------------

    def _section_early_stopping(self, exp) -> list[str]:
        tag = _badge(self._es_configured, len(self._seen_early_stopped) > 0)
        L = [f"## Early Stopping {tag}", ""]
        if not self._es_configured:
            L.append("Not configured.")
            L.append("")
            return L

        es_cfg = _get(self._cfg, "orchestration_settings", "early_stopping_strategy")
        strategy_type = _field(es_cfg, "type", "unknown") if es_cfg else "unknown"
        n_total = len(exp.trials)
        n_stopped = len(self._seen_early_stopped)
        pct = n_stopped / n_total * 100 if n_total else 0

        metric_sigs = _collect_es_metrics(es_cfg)
        metrics_str = ", ".join(f"`{m}`" for m in metric_sigs) if metric_sigs else "all objectives"
        L.append(f"**Strategy:** `{strategy_type}`")
        L.append(f"**Metrics:** {metrics_str}")
        # For composite strategies, show per-child details
        if strategy_type in ("or", "and"):
            for child_key in ("left", "right"):
                child = _field(es_cfg, child_key, None)
                if child is None:
                    continue
                child_type = _field(child, "type", "?")
                child_metrics = _field(child, "metric_signatures", None) or _field(child, "metric_names", [])
                child_thresh = _field(child, "metric_threshold", None)
                child_pct = _field(child, "percentile_threshold", None)
                child_min = _field(child, "min_progression", None)
                parts = [f"`{child_type}`"]
                if child_metrics:
                    parts.append(f"metrics: {', '.join(f'`{m}`' for m in child_metrics)}")
                if child_thresh is not None:
                    parts.append(f"threshold: {child_thresh}")
                if child_pct is not None:
                    parts.append(f"percentile: {child_pct}")
                if child_min is not None:
                    parts.append(f"min_progression: {child_min}")
                L.append(f"- {child_key}: {' | '.join(parts)}")
        else:
            threshold = _field(es_cfg, "percentile_threshold", None)
            min_prog = _field(es_cfg, "min_progression", None)
            if threshold is not None:
                L.append(f"**Percentile threshold:** {threshold}")
            if min_prog is not None:
                L.append(f"**Min progression:** {min_prog} steps before stopping")
        L.append(f"**Trials stopped:** {n_stopped} / {n_total} ({pct:.1f}%)")
        L.append("")

        if self._seen_early_stopped:
            L.append("| Trial | Parameter | Value |")
            L.append("|------:|:----------|:------|")
            for idx in sorted(self._seen_early_stopped):
                trial = exp.trials.get(idx)
                if trial and trial.arm:
                    params = trial.arm.parameters
                    first = True
                    for pname, pval in params.items():
                        trial_col = str(idx) if first else ""
                        val_str = f"{pval:.4g}" if isinstance(pval, float) else str(pval)
                        L.append(f"| {trial_col} | `{pname}` | {val_str} |")
                        first = False
                else:
                    L.append(f"| {idx} | | |")
            L.append("")
            L.append(f"**Gain:** ~{pct:.0f}% compute savings on unpromising evaluations.")
        else:
            L.append("No trials stopped yet.")

        L.append("")
        return L

    def _section_global_stopping(self, exp) -> list[str]:
        n_completed = sum(1 for t in exp.trials.values() if t.status == AxTrialStatus.COMPLETED)
        triggered = self._gs_configured and self._max_trials > 0 and 0 < n_completed < self._max_trials
        tag = _badge(self._gs_configured, triggered)
        L = [f"## Global Stopping {tag}", ""]
        if not self._gs_configured:
            L.append("Not configured.")
            L.append("")
            return L

        gs_cfg = _get(self._cfg, "orchestration_settings", "global_stopping_strategy")
        imp_bar = _field(gs_cfg, "improvement_bar", "?") if gs_cfg else "?"
        window = _field(gs_cfg, "window_size", "?") if gs_cfg else "?"
        L.append(f"**Strategy:** `improvement_bar={imp_bar}, window_size={window}`")

        if self._max_trials:
            pct = n_completed / self._max_trials * 100
            L.append(f"**Budget consumed:** {n_completed} / {self._max_trials} ({pct:.0f}%)")
        else:
            L.append(f"**Completed trials:** {n_completed}")

        if triggered:
            saved = self._max_trials - n_completed
            L.append(f"")
            L.append(f"**Gain:** Converged early, saving {saved} trial(s).")
        else:
            L.append("")
            L.append("Not yet triggered.")

        L.append("")
        return L

    def _section_dim_reduction(self, exp) -> list[str]:
        all_params = list(exp.search_space.parameters.values())
        fixed = [p for p in all_params if isinstance(p, FixedParameter)]
        active = [p for p in all_params if not isinstance(p, FixedParameter)]
        n_total = len(self._original_param_names or all_params)
        triggered = self._dr_configured and len(fixed) > 0

        tag = _badge(self._dr_configured, triggered)
        L = [f"## Dimensionality Reduction {tag}", ""]
        if not self._dr_configured:
            L.append("Not configured.")
            L.append("")
            return L

        L.append(f"**Config:** `after_trials={self._dr_after}  "
                 f"min_importance={self._dr_min_importance}  "
                 f"fix_at={self._dr_fix_at}  "
                 f"max_fix_fraction={self._dr_max_frac}`")
        L.append("")

        n_completed = sum(1 for t in exp.trials.values() if t.status == AxTrialStatus.COMPLETED)
        if not triggered:
            if n_completed < self._dr_after:
                L.append(f"Waiting: {n_completed} / {self._dr_after} trials before analysis.")
            else:
                L.append("All parameters above importance threshold.")
        else:
            pct = len(fixed) / n_total * 100 if n_total else 0
            L.append(f"Fixed **{len(fixed)} / {n_total}** parameters ({pct:.0f}% search space reduction):")
            L.append("")
            L.append("| Parameter | Fixed Value |")
            L.append("|:----------|:------------|")
            for p in fixed:
                L.append(f"| `{p.name}` | {p.value} |")
            L.append("")
            L.append(f"**Active:** {', '.join(f'`{p.name}`' for p in active)}")
            L.append("")
            L.append(f"**Gain:** Search space reduced by {pct:.0f}%, "
                     f"focusing exploration on {len(active)} parameter(s).")

        L.append("")
        return L

    def _section_trial_dependencies(self, exp) -> list[str]:
        n_trials = len(exp.trials)
        has_resolutions = len(self._dep_resolutions) > 0
        triggered = self._deps_configured and (n_trials > 1 or has_resolutions)
        tag = _badge(self._deps_configured, triggered)
        L = [f"## Trial Dependencies {tag}", ""]
        if not self._deps_configured:
            L.append("Not configured.")
            L.append("")
            return L

        L.append("### Configured Rules")
        L.append("")
        for d in self._deps:
            name = _field(d, "name", "unnamed")
            source_cfg = _field(d, "source", None)
            strategy = _field(source_cfg, "strategy", "?") if source_cfg else "?"
            fallback = _field(source_cfg, "fallback", "skip") if source_cfg else "skip"
            actions = _field(d, "actions", [])
            action_list = list(actions) if actions else []
            for a in action_list:
                cmd = _field(a, "command", "")
                phase = _field(a, "phase", "immediate")
                if cmd:
                    env_var = f"`$FOAMBO_{phase.upper()}`" if phase != "immediate" else "*(inline)*"
                    L.append(f"- **`{name}`** | source: `{strategy}` | "
                             f"phase: `{phase}` {env_var} | fallback: `{fallback}`")
                    L.append(f"  ```")
                    L.append(f"  {cmd}")
                    L.append(f"  ```")
        L.append("")

        if has_resolutions:
            L.append("### Resolution Log")
            L.append("")
            L.append("| Trial | Dependency | Source Trial | Phase(s) |")
            L.append("|------:|:-----------|:-------------|:---------|")
            for r in self._dep_resolutions[-20:]:
                src_idx = r.get("source_index", "?")
                phases = ", ".join(r.get("phased_actions", [])) or "immediate"
                L.append(f"| {r['trial']} | `{r['dependency']}` | {src_idx} | `{phases}` |")
            if len(self._dep_resolutions) > 20:
                L.append(f"| ... | *{len(self._dep_resolutions) - 20} earlier* | | |")
            L.append("")

            n_resolved = len(self._dep_resolutions)
            n_unique_sources = len({r["source_path"] for r in self._dep_resolutions})
            L.append(f"**Total resolutions:** {n_resolved} across {n_unique_sources} unique source trial(s).")
        else:
            applied_to = max(0, n_trials - 1)
            L.append(f"**Applied to:** {applied_to} trial(s) (no per-trial metadata available yet).")

        if triggered:
            L.append("")
            L.append("**Gain:** Warm-starting trials from prior solutions.")

        L.append("")
        return L

    def _section_baseline(self, exp, client: Client) -> list[str]:
        has_sq = exp.status_quo is not None
        tag = _badge(self._baseline_configured or has_sq, has_sq)
        L = [f"## Baseline Comparison {tag}", ""]
        if not self._baseline_configured and not has_sq:
            L.append("Not configured.")
            L.append("")
            return L

        if not has_sq:
            L.append("Configured but baseline trial not yet completed.")
            L.append("")
            return L

        L.append(f"**Baseline arm:** `{exp.status_quo.name}`")
        L.append("")

        try:
            opt_config = exp.optimization_config
            is_mo = isinstance(opt_config.objective, MultiObjective)
            objectives = (
                opt_config.objective.objectives if is_mo
                else [opt_config.objective]
            )
            data = exp.lookup_data().df
            if data.empty:
                L.append("No metric data yet.")
                L.append("")
                return L

            L.append("| Metric | Baseline | Best | Improvement |")
            L.append("|:-------|:---------|:-----|:------------|")
            for obj in objectives:
                mname = obj.metric.name
                minimize = obj.minimize

                bl_rows = data[(data["metric_name"] == mname) & (data["arm_name"] == exp.status_quo.name)]
                if bl_rows.empty:
                    continue
                bl_val = float(bl_rows["mean"].iloc[0])
                all_vals = data[data["metric_name"] == mname]["mean"]
                best = float(all_vals.min() if minimize else all_vals.max())

                if bl_val != 0:
                    pct = abs(best - bl_val) / abs(bl_val) * 100
                else:
                    pct = float("inf") if best != bl_val else 0.0
                direction = "reduction" if minimize else "increase"
                L.append(f"| `{mname}` | {bl_val:.4g} | {best:.4g} | {pct:.1f}% {direction} |")
        except Exception as e:
            L.append(f"*(could not compute improvements: {e})*")

        L.append("")
        return L

    def _section_parallelism(self) -> list[str]:
        tag = _badge(self._parallelism > 1, self._parallelism > 1)
        L = [f"## Parallel Evaluation {tag}", ""]
        L.append(f"**Parallelism:** {self._parallelism}")
        if self._parallelism > 1:
            L.append(f"**Gain:** Up to {self._parallelism}x concurrent trial evaluation.")
        else:
            L.append("Sequential execution.")
        L.append("")
        return L

    def _section_custom_kernel(self, client: Client) -> list[str]:
        configured = False
        kernel_info = "default GP"
        try:
            gs = client._generation_strategy
            if gs is not None:
                from ax.adapter.registry import Generators
                for node in gs._nodes:
                    for spec in node.generator_specs:
                        if spec.generator_enum == Generators.BOTORCH_MODULAR:
                            surr = spec.generator_kwargs.get("surrogate_spec")
                            if surr is not None:
                                configured = True
                                kernel_info = type(surr).__name__
        except Exception:
            pass

        tag = _badge(configured, configured)
        L = [f"## Custom Kernel {tag}", ""]
        L.append(f"**Kernel:** `{kernel_info}`")
        L.append("")
        return L

    def _section_existing_trials(self, exp) -> list[str]:
        et_cfg = _get(self._cfg, "existing_trials")
        file_path = (_field(et_cfg, "file_path", None) or _field(et_cfg, "file", None)) if et_cfg else None
        configured = file_path is not None and file_path != ""

        n_manual = 0
        if configured:
            for t in exp.trials.values():
                gr = t.generator_run
                if gr and gr.generator_run_type and "manual" in gr.generator_run_type.lower():
                    n_manual += 1

        tag = _badge(configured, n_manual > 0)
        L = [f"## Existing Trials {tag}", ""]
        if not configured:
            L.append("Not configured.")
        elif n_manual > 0:
            L.append(f"**Source:** `{file_path}`")
            L.append(f"**Loaded:** {n_manual} trial(s)")
            L.append(f"**Gain:** Warm-started surrogate model with prior data.")
        else:
            L.append(f"**Source:** `{file_path}`")
            L.append("No trials loaded.")
        L.append("")
        return L

    def _section_efficiency(self, exp) -> list[str]:
        L = ["---", "", "## Compute Efficiency", ""]

        # Gather trial durations: start_time from run_metadata, end from our tracked completions
        durations: list[float] = []
        start_times: list[float] = []
        for idx, trial in exp.trials.items():
            meta = trial.run_metadata or {}
            start = meta.get("start_time") or (meta.get("job") or {}).get("start_time")
            if start is None:
                continue
            start_times.append(start)
            end = self._trial_completions.get(idx)
            if end is not None and end > start:
                durations.append(end - start)

        if len(durations) < 2:
            L.append("Not enough completed trials to measure efficiency.")
            L.append("")
            return L

        import time
        avg_duration = sum(durations) / len(durations)
        wall_time = time.time() - min(start_times)
        ideal_time = (len(durations) * avg_duration) / self._parallelism
        efficiency = ideal_time / wall_time if wall_time > 0 else 0.0
        idle_fraction = 1.0 - min(efficiency, 1.0)

        def _fmt(seconds: float) -> str:
            if seconds < 60:
                return f"{seconds:.0f}s"
            if seconds < 3600:
                return f"{seconds / 60:.1f}m"
            return f"{seconds / 3600:.1f}h"

        L.append(f"| Metric | Value |")
        L.append(f"|:-------|:------|")
        L.append(f"| Avg trial duration | {_fmt(avg_duration)} |")
        L.append(f"| Wall time so far | {_fmt(wall_time)} |")
        L.append(f"| Ideal wall time | {_fmt(ideal_time)} (= {len(durations)} trials x {_fmt(avg_duration)} / {self._parallelism} slots) |")
        L.append(f"| Compute efficiency | **{efficiency:.0%}** |")
        L.append(f"| Idle fraction | {idle_fraction:.0%} |")
        L.append("")

        if idle_fraction > 0.5:
            L.append(f"> High idle time. Consider increasing parallelism or "
                     f"reducing `initial_seconds_between_polls`.")
        elif idle_fraction < 0.1:
            L.append(f"> Excellent resource utilisation.")
        L.append("")
        return L

    def _section_best_progress(self) -> list[str]:
        L = ["---", "", "## Best Objective Progress", ""]
        if not self._best_trace:
            L.append("No completed trials yet.")
            L.append("")
            return L

        # Table of recent improvements
        recent = self._best_trace[-10:]
        metric_names = list(recent[0]["metrics"].keys())
        header = "| After Trial | " + " | ".join(f"`{m}`" for m in metric_names) + " |"
        sep = "|:-----------:|" + "|".join(":----:" for _ in metric_names) + "|"
        L.append(header)
        L.append(sep)
        for entry in recent:
            vals = " | ".join(
                f"{entry['metrics'][m]:,.4g}" if m in entry["metrics"] else "-"
                for m in metric_names)
            L.append(f"| {entry['trial']} | {vals} |")
        L.append("")

        # Overall improvement
        if len(self._best_trace) >= 2:
            first = self._best_trace[0]["metrics"]
            last = self._best_trace[-1]["metrics"]
            L.append("**Overall improvement:**")
            for mname in last:
                if mname in first and first[mname] != 0:
                    pct = abs(last[mname] - first[mname]) / abs(first[mname]) * 100
                    arrow = "v" if last[mname] < first[mname] else "^"
                    L.append(f"- `{mname}`: {first[mname]:.4g} -> {last[mname]:.4g} ({arrow} {pct:.1f}%)")

        L.append("")
        return L

    def _section_event_log(self) -> list[str]:
        if not self._events:
            return []
        L = ["---", "", "## Event Log", ""]
        for ev in self._events[-25:]:
            L.append(f"- {ev}")
        if len(self._events) > 25:
            L.append(f"- *... {len(self._events) - 25} earlier events omitted*")
        L.append("")
        return L

    def _event(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._events.append(f"`{ts}` {msg}")



def _badge(configured: bool, triggered: bool) -> str:
    if configured and triggered:
        return "**`[ACTIVE]`**"
    elif configured:
        return "`[WATCHING]`"
    return "`[OFF]`"


def _collect_es_metrics(es_cfg) -> list[str]:
    """Recursively collect metric names from (possibly composite) ES config."""
    if es_cfg is None:
        return []
    sigs = _field(es_cfg, "metric_signatures", None) or _field(es_cfg, "metric_names", None)
    if sigs:
        return list(sigs)
    result = []
    for child in ("left", "right"):
        sub = _field(es_cfg, child, None)
        if sub is not None:
            result.extend(_collect_es_metrics(sub))
    return result


def _is_set(cfg, *keys) -> bool:
    val = _get(cfg, *keys)
    return val is not None and val != "none"


def _get(cfg, *keys):
    """Safely traverse nested DictConfig / dict / Pydantic objects."""
    obj = cfg
    for k in keys:
        if obj is None:
            return None
        if isinstance(obj, DictConfig):
            obj = obj.get(k)
        elif isinstance(obj, dict):
            obj = obj.get(k)
        elif hasattr(obj, k):
            obj = getattr(obj, k)
        else:
            return None
    return obj


def _field(obj, field, default=None):
    """Get a field from a dict, DictConfig, or Pydantic model."""
    if obj is None:
        return default
    if isinstance(obj, (dict, DictConfig)):
        return obj.get(field, default)
    return getattr(obj, field, default)
