"""Config upgrade checker — compares user YAML against current Pydantic schema.

Fully passive: prints a diff report, never modifies files.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)

# ANSI colors
_R = "\033[91m"  # red (remove)
_G = "\033[92m"  # green (add)
_Y = "\033[93m"  # yellow (change)
_C = "\033[96m"  # cyan (info)
_D = "\033[2m"   # dim
_0 = "\033[0m"   # reset


def _fmt_val(v) -> str:
    """Format a config value for display, handling special types."""
    try:
        from .default_config import RangeTag
        if isinstance(v, RangeTag):
            if v.step == 1:
                return f"!range [{v.start}, {v.stop}]"
            return f"!range [{v.start}, {v.stop}, {v.step}]"
    except ImportError:
        pass
    return str(v)


def _walk_schema(model_cls: type[BaseModel], prefix: str = "") -> dict[str, dict]:
    """Recursively collect {dotted_path: {type, default, description}} from Pydantic models."""
    import typing
    import types
    from pydantic.fields import PydanticUndefined

    def _is_model(tp):
        try:
            return isinstance(tp, type) and issubclass(tp, BaseModel)
        except TypeError:
            return False

    def _unwrap(ann):
        origin = getattr(ann, "__origin__", None)
        if origin is types.UnionType or origin is typing.Union:
            args = [a for a in typing.get_args(ann) if a is not type(None)]
            return args[0] if len(args) == 1 else ann
        return ann

    fields = {}
    for name, fi in model_cls.model_fields.items():
        path = f"{prefix}.{name}" if prefix else name
        ann = _unwrap(fi.annotation)
        if _is_model(ann):
            fields.update(_walk_schema(ann, path))
        else:
            default = fi.default if fi.default is not PydanticUndefined else None
            fields[path] = {
                "type": getattr(ann, "__name__", str(ann)),
                "default": default,
                "description": fi.description or "",
            }
    return fields


def _flatten_config(cfg: dict | list, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict/list into dotted paths."""
    out = {}
    if isinstance(cfg, list):
        for i, item in enumerate(cfg):
            path = f"{prefix}.{i}" if prefix else str(i)
            if isinstance(item, (dict, list)):
                out.update(_flatten_config(item, path))
            else:
                out[path] = item
        return out
    for k, v in cfg.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_config(v, path))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out.update(_flatten_config(v, path))
        else:
            out[path] = v
    return out


# Known renames across versions: {old_path: new_path}
_KNOWN_RENAMES = {
    "orchestration_settings.early_stopping": "orchestration_settings.early_stopping_strategy",
    "optimization.metric_names": "optimization.metric_signatures",
    "visualizer": None,  # removed
    "visualizer.sensitivity_callback": None,  # removed
}


def run_upgrade_check(config_path: str) -> bool:
    """Compare user config YAML against current schema and print a diff report.

    Returns True if the config is up to date, False if changes are suggested.
    """
    from omegaconf import OmegaConf
    from .config import load_config
    from .orchestrate import FoamBOConfig

    # Load user config (handles custom YAML tags like !range)
    raw = load_config(config_path)
    user_flat = _flatten_config(OmegaConf.to_container(raw, resolve=True))

    # Build schema from Pydantic models
    schema = _walk_schema(FoamBOConfig)

    user_keys = set(user_flat.keys())
    schema_keys = set(schema.keys())

    deprecated = []
    added = []
    type_issues = []
    renames = []

    # Prefixes for dynamic/user-defined sections (not in Pydantic schema but valid)
    # These are List[Any], dict-typed, or user-defined content that can't be schema-walked
    _DYNAMIC_PREFIXES = (
        "baseline.parameters.",
        "orchestration_settings.early_stopping_strategy.",
        "orchestration_settings.global_stopping_strategy.",
        "store.backend_options.",
        "experiment.parameters.",
        "optimization.metrics.",
        "optimization.case_runner.variable_substitution.",
        "optimization.case_runner.runner_scripts.",
        "trial_generation.generation_nodes.",
        "trial_dependencies.",
    )

    # Check for deprecated/unknown keys
    for key in sorted(user_keys - schema_keys):
        # Skip top-level non-schema keys that are allowed (version, etc.)
        if key == "version":
            continue
        # Skip dynamic/user-defined sections
        if any(key.startswith(p) for p in _DYNAMIC_PREFIXES):
            continue
        # Check if it's a known rename
        if key in _KNOWN_RENAMES:
            new = _KNOWN_RENAMES[key]
            if new is None:
                deprecated.append((key, "removed in current version"))
            else:
                renames.append((key, new))
        else:
            # Check if parent section exists (could be an extra key in a section with extra="allow")
            parent = key.rsplit(".", 1)[0] if "." in key else ""
            if parent in schema_keys or not parent:
                deprecated.append((key, "not in current schema"))
            else:
                deprecated.append((key, "unknown section"))

    # Check for new keys with defaults (from Pydantic schema)
    for key in sorted(schema_keys - user_keys):
        info = schema[key]
        # Only suggest if there's a meaningful default
        if info["default"] is not None:
            added.append((key, info["default"], info["description"]))

    # Second pass: compare leaf field names in dynamic list sections
    # For each _DYNAMIC_PREFIXES section, extract the set of leaf field names
    # from the default config and from the user config, then report differences.
    # e.g. default generation_nodes have "exclude_transforms" but user's don't
    from .default_config import get_default_config
    default_flat = _flatten_config(get_default_config())

    def _leaf_names(flat: dict, prefix: str) -> set[str]:
        """Extract unique leaf field names under a prefix, stripping numeric indices."""
        import re
        names = set()
        for k in flat:
            if k.startswith(prefix):
                # Strip prefix and numeric indices: "0.generator_specs.0.model_kwargs.seed" -> "generator_specs.model_kwargs.seed"
                tail = k[len(prefix):]
                cleaned = re.sub(r'\d+\.', '', tail).strip(".")
                if cleaned:
                    names.add(cleaned)
        return names

    missing_options = []
    for prefix in _DYNAMIC_PREFIXES:
        # Skip baseline.parameters unless user's section is empty
        if prefix == "baseline.parameters.":
            has_user_params = any(k.startswith(prefix) for k in user_keys)
            if has_user_params:
                continue
        default_leaves = _leaf_names(default_flat, prefix)
        user_leaves = _leaf_names(user_flat, prefix)
        # Only show if user has this section at all
        if not user_leaves:
            continue
        new_leaves = sorted(default_leaves - user_leaves)
        if new_leaves:
            section = prefix.rstrip(".")
            for leaf in new_leaves:
                missing_options.append((section, leaf))

    # Check type mismatches
    for key in sorted(user_keys & schema_keys):
        info = schema[key]
        user_val = user_flat[key]
        expected_type = info["type"]

        # Basic type checks
        if expected_type == "int" and not isinstance(user_val, int):
            type_issues.append((key, expected_type, type(user_val).__name__, user_val))
        elif expected_type == "float" and not isinstance(user_val, (int, float)):
            type_issues.append((key, expected_type, type(user_val).__name__, user_val))
        elif expected_type == "bool" and not isinstance(user_val, bool):
            type_issues.append((key, expected_type, type(user_val).__name__, user_val))
        elif expected_type == "str" and not isinstance(user_val, str):
            type_issues.append((key, expected_type, type(user_val).__name__, user_val))

    # Print report
    has_issues = bool(deprecated or added or missing_options or type_issues or renames)

    print(f"\n{_C}foamBO config upgrade check{_0}: {config_path}\n")

    if not has_issues:
        print(f"  {_G}✓ Config is up to date — no changes needed.{_0}\n")
        return True

    if renames:
        print(f"  {_Y}Renamed keys{_0} (update path):\n")
        for old, new in renames:
            print(f"    {_Y}~{_0} {old}  →  {new}")
        print()

    if deprecated:
        print(f"  {_R}Deprecated/unknown keys{_0} (consider removing):\n")
        for key, reason in deprecated:
            val = user_flat.get(key, "")
            val_str = f"  {_D}(value: {val}){_0}" if val != "" else ""
            print(f"    {_R}-{_0} {key}  {_D}({reason}){_0}{val_str}")
        print()

    if added:
        print(f"  {_G}New keys available{_0} (consider adding):\n")
        current_section = ""
        for key, default, desc in added:
            section = key.rsplit(".", 1)[0] if "." in key else key
            if section != current_section:
                print(f"    {_D}{section}:{_0}")
                current_section = section
            leaf = key.rsplit(".", 1)[-1]
            desc_short = desc[:80].replace("\n", " ").strip()
            print(f"      {_G}+{_0} {leaf}: {_C}{_fmt_val(default)}{_0}")
            if desc_short:
                print(f"        {_D}{desc_short}{_0}")
        print()

    if missing_options:
        print(f"  {_C}Additional options available{_0} (in sections you already use):\n")
        current_section = ""
        for section, leaf in missing_options:
            if section != current_section:
                print(f"    {_D}{section}:{_0}")
                current_section = section
            # Try to find the default value
            matching = [k for k in default_flat if k.startswith(section) and k.endswith(leaf.split(".")[-1])]
            val = _fmt_val(default_flat[matching[0]]) if matching else "..."
            print(f"      {_G}+{_0} {leaf}: {_C}{val}{_0}")
        print()

    if type_issues:
        print(f"  {_Y}Type mismatches{_0}:\n")
        for key, expected, actual, val in type_issues:
            print(f"    {_Y}!{_0} {key}: expected {_C}{expected}{_0}, got {_R}{actual}{_0} ({_D}{val}{_0})")
        print()

    total = len(deprecated) + len(added) + len(missing_options) + len(type_issues) + len(renames)
    print(f"  {_C}Summary{_0}: {len(renames)} renames, {len(deprecated)} deprecated, "
          f"{len(added)} new, {len(missing_options)} deep options, "
          f"{len(type_issues)} type issues ({total} total)\n")

    return False
