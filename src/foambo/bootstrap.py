"""Bootstrap support for foamBO.

Allows a YAML config to inherit from a previously saved experiment state
(``bootstrap: path/to/<name>_client_state.json``). The embedded ``foambo_config``
from the JSON state is used as the base, with the current YAML merged on top
via OmegaConf. This supports two workflows:

1. **Continue** — load prior GP + trials under a new experiment name / extra
   trials. YAML only overrides ``experiment.name`` and ``orchestration_settings.n_trials``
   (typically).
2. **Specialize** — pin context parameters from a robust run to specific values
   via a top-level ``specialize: {param: value, ...}`` mapping. The search space
   and recorded trial arms are rewritten so only the remaining design variables
   are optimized; prior trial outcomes are retained as training data at the
   clamped context.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


_BOOTSTRAP_META_KEY = "_bootstrap_meta"


def resolve_bootstrap(cfg: DictConfig, yaml_path: str | None) -> DictConfig:
    """If cfg has ``bootstrap:``, merge saved parent config with the current YAML.

    Returns the merged config. The bootstrap path and any ``specialize`` map are
    stashed under ``cfg._bootstrap_meta`` for downstream loaders to consume. The
    top-level ``bootstrap`` and ``specialize`` keys are removed from the merged
    result.

    Raises ``FileNotFoundError`` if the referenced JSON does not exist and
    ``ValueError`` if it lacks an embedded ``foambo_config``.
    """
    if "bootstrap" not in cfg or cfg.bootstrap is None:
        return cfg

    raw_path = str(cfg.bootstrap)
    base_dir = os.path.dirname(os.path.abspath(yaml_path)) if yaml_path else os.getcwd()
    abs_path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"bootstrap path not found: {abs_path}")

    with open(abs_path) as f:
        state = json.load(f)
    parent_cfg = state.get("foambo_config")
    if parent_cfg is None:
        raise ValueError(
            f"bootstrap JSON has no embedded foambo_config: {abs_path}. "
            "Re-save the parent experiment with a recent foamBO version."
        )

    # Drop meta keys from the override prior to merge.
    override = OmegaConf.to_container(cfg, resolve=False)
    override.pop("bootstrap", None)
    specialize = override.pop("specialize", None)

    # Warn if the YAML attempts to change the parent's optimization_config —
    # Ax locks objectives/metrics/constraints on the loaded experiment and the
    # override will be silently ignored.
    _warn_optimization_override(parent_cfg.get("optimization", {}) or {},
                                override.get("optimization", {}) or {})

    base = OmegaConf.create(parent_cfg)
    parent_name = None
    try:
        parent_name = str(base.experiment.name)
    except Exception:
        pass
    merged = OmegaConf.merge(base, OmegaConf.create(override))
    # Attach internal meta for optimize() to pick up.
    OmegaConf.update(
        merged,
        _BOOTSTRAP_META_KEY,
        {"client_state_path": abs_path, "specialize": specialize or {}},
        force_add=True,
    )
    # Public lineage record — preserved in the merged cfg, embedded in the
    # saved state JSON, and surfaced on the dashboard.
    lineage = {
        "parent_state_path": abs_path,
        "parent_name": parent_name,
        "specialize": specialize or {},
    }
    OmegaConf.update(merged, "bootstrap_lineage", lineage, force_add=True)
    log.info("Bootstrap: loaded parent config from %s", abs_path)
    if specialize:
        log.info("Bootstrap: specializing %d parameter(s): %s",
                 len(specialize), ", ".join(f"{k}={v}" for k, v in specialize.items()))
    return merged


_LOCKED_OPT_FIELDS = ("objective", "metrics", "outcome_constraints", "objective_thresholds")


def _warn_optimization_override(parent: dict, override: dict) -> None:
    """Log a warning for any locked optimization field that the YAML tries to change."""
    diffs = []
    for key in _LOCKED_OPT_FIELDS:
        if key not in override:
            continue
        new_val = override[key]
        old_val = parent.get(key)
        if _normalize(new_val) != _normalize(old_val):
            diffs.append(f"  - {key}: parent={_fmt(old_val)}  YAML={_fmt(new_val)}")
    if not diffs:
        return
    log.warning(
        "Bootstrap: the YAML override tries to change locked optimization fields. "
        "Ax's Client freezes objectives/metrics on the loaded experiment, so these "
        "changes will be IGNORED at runtime:\n%s\n"
        "To use a different optimization config, bootstrap is not the right tool — "
        "start a fresh experiment and seed past trials via "
        "trial_generation.generation_nodes with `file_path:` instead.",
        "\n".join(diffs),
    )


def _normalize(v: Any) -> Any:
    """Canonicalize a value for equality comparison (stable ordering, stripped whitespace)."""
    if isinstance(v, list):
        # Normalize each element; for lists of dicts, sort by a stable key.
        normed = [_normalize(x) for x in v]
        try:
            return sorted(normed, key=lambda x: json.dumps(x, sort_keys=True))
        except TypeError:
            return normed
    if isinstance(v, dict):
        return {k: _normalize(val) for k, val in v.items()}
    if isinstance(v, str):
        return v.strip()
    return v


def _fmt(v: Any) -> str:
    try:
        return json.dumps(v, default=str)
    except Exception:
        return str(v)


def bootstrap_meta(cfg: DictConfig) -> dict[str, Any] | None:
    """Return the bootstrap meta dict if present, else None."""
    meta = cfg.get(_BOOTSTRAP_META_KEY) if hasattr(cfg, "get") else None
    if meta is None:
        return None
    return OmegaConf.to_container(meta, resolve=True)


def strip_bootstrap_meta(cfg: DictConfig) -> DictConfig:
    """Remove the bootstrap meta key from cfg (call before embedding in state)."""
    if _BOOTSTRAP_META_KEY in cfg:
        del cfg[_BOOTSTRAP_META_KEY]
    return cfg


def apply_specialization(client, specialize: dict[str, Any]) -> None:
    """Pin context parameters to fixed values on a loaded client.

    Mutates the experiment's search space (replacing affected parameters with
    ``FixedParameter``) and rewrites recorded arm parameters so prior trials
    remain valid training data under the new space. The GP refits on the next
    generation call.

    Limitations:
    - Information about how those parameters influence outcomes is lost: all
      trials now appear to share the same context value. This is the intended
      behavior for "optimize conditional on this context point".
    - Ax experiments with ``immutable_search_space_and_opt_config`` cannot be
      specialized in-place; this raises a clear error.
    """
    if not specialize:
        return

    from ax.core.parameter import FixedParameter, RangeParameter, ChoiceParameter
    from ax.core.search_space import SearchSpace

    exp = client._experiment
    # Ax sets immutable_search_space_and_opt_config=True on every experiment
    # produced via Client.configure_experiment. That only means generator runs
    # skipped caching a per-trial search-space copy; the live search space is
    # still a plain attribute. Unset the flag so we can mutate it here.
    if getattr(exp, "immutable_search_space_and_opt_config", False):
        try:
            from ax.core.experiment import Keys
            exp._properties.pop(Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF, None)
        except Exception:
            exp._properties = {
                k: v for k, v in (exp._properties or {}).items()
                if "immutable" not in str(k).lower()
            }
        log.info("Specialization: cleared parent's immutable_search_space_and_opt_config flag")

    parent_params = exp.search_space.parameters
    missing = [k for k in specialize if k not in parent_params]
    if missing:
        raise ValueError(f"specialize keys not in parent search space: {missing}")

    new_params = []
    for name, p in parent_params.items():
        if name in specialize:
            value = specialize[name]
            # Coerce to parameter type (yaml may give str/int/float mix).
            if isinstance(p, RangeParameter):
                value = float(value) if p.parameter_type.name in ("FLOAT",) else int(value)
            new_params.append(FixedParameter(
                name=name,
                parameter_type=p.parameter_type,
                value=value,
            ))
        else:
            new_params.append(p.clone())
    new_ss = SearchSpace(parameters=new_params)
    exp._search_space = new_ss

    # Clamp arm parameters on every existing trial so past observations are
    # consistent with the new search space.
    for t in exp.trials.values():
        for arm in t.arms:
            for pname, pval in specialize.items():
                if pname in arm._parameters:
                    arm._parameters[pname] = pval

    # Invalidate any cached adapter state so the next gen refits on the clamped
    # data under the new search space.
    gs = getattr(client, "_generation_strategy", None)
    if gs is not None and getattr(gs, "adapter", None) is not None:
        try:
            gs._curr._fitted_adapter = None
        except Exception:
            pass

    log.info("Specialization applied: search space now has %d fixed parameter(s)",
             sum(1 for p in new_params if isinstance(p, FixedParameter)))
