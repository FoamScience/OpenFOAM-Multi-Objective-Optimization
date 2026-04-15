"""Pack and unpack foamBO experiments as compressed HDF5 archives.

Usage:
    foamBO --pack --config MyOpt.yaml
    foamBO --pack --config MyOpt.yaml --include-trials best,pareto,0,15
    foamBO --unpack experiment.foambo
"""
from __future__ import annotations

import json
import logging
import os
import stat
import fnmatch
from datetime import datetime, timezone
from pathlib import Path

import h5py
import hdf5plugin

from .common import VERSION

log = logging.getLogger(__name__)

DEFAULT_SKIP = [
    "*.foam",
    "__pycache__",
    ".git",
    "*.pyc",
    ".foambo_screenshot.png",
]


def _compression():
    """Return Blosc2(zstd) compression kwargs for h5py datasets."""
    return hdf5plugin.Blosc2(cname="zstd", clevel=9, shuffle=hdf5plugin.Blosc2.SHUFFLE)


def _should_skip(path: str, patterns: list[str]) -> bool:
    """Check if a path component matches any skip pattern."""
    parts = Path(path).parts
    for pattern in patterns:
        clean = pattern.rstrip("/")
        for part in parts:
            if fnmatch.fnmatch(part, clean):
                return True
    return False


def _store_directory(group: h5py.Group, local_path: str, skip: list[str]):
    """Recursively store a directory tree into an HDF5 group."""
    local_path = os.path.abspath(local_path)
    n_files = 0
    total_bytes = 0
    for root, dirs, files in os.walk(local_path, followlinks=False):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if not _should_skip(d, skip)]
        rel_root = os.path.relpath(root, local_path)
        for fname in files:
            rel_path = os.path.join(rel_root, fname) if rel_root != "." else fname
            if _should_skip(rel_path, skip):
                continue
            full_path = os.path.join(root, fname)
            # Handle symlinks
            if os.path.islink(full_path):
                target = os.readlink(full_path)
                ds = group.create_dataset(rel_path, data=b"", **_compression())
                ds.attrs["is_symlink"] = True
                ds.attrs["symlink_target"] = target
            else:
                try:
                    with open(full_path, "rb") as f:
                        data = f.read()
                    ds = group.create_dataset(rel_path, data=data, **_compression())
                    ds.attrs["is_symlink"] = False
                    total_bytes += len(data)
                except (PermissionError, OSError) as e:
                    log.warning(f"Skipping {rel_path}: {e}")
                    continue
            ds.attrs["mode"] = os.stat(full_path).st_mode & 0o777
            n_files += 1
    return n_files, total_bytes


def _extract_directory(group: h5py.Group, target_path: str):
    """Extract an HDF5 group into a directory tree."""
    n_files = 0

    def _visit(name, obj):
        nonlocal n_files
        if not isinstance(obj, h5py.Dataset):
            return
        out_path = os.path.join(target_path, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if obj.attrs.get("is_symlink", False):
            link_target = obj.attrs["symlink_target"]
            if os.path.exists(out_path) or os.path.islink(out_path):
                os.unlink(out_path)
            os.symlink(link_target, out_path)
        else:
            with open(out_path, "wb") as f:
                f.write(bytes(obj[()]))
            mode = obj.attrs.get("mode", 0o644)
            os.chmod(out_path, mode)
        n_files += 1

    group.visititems(_visit)
    return n_files


def _collect_script_paths(cfg) -> list[str]:
    """Collect metric and runner script paths from the config."""
    paths = set()
    # Runner script
    runner = cfg.get("optimization", {}).get("case_runner", {}).get("runner")
    if runner and isinstance(runner, str) and os.path.isfile(runner):
        paths.add(os.path.abspath(runner))
    # Metric commands — extract file paths from command lists/strings
    for m in cfg.get("optimization", {}).get("metrics", []):
        for key in ("command", "progress"):
            cmd = m.get(key)
            if not cmd:
                continue
            if isinstance(cmd, list):
                for part in cmd:
                    if isinstance(part, str) and os.path.isfile(part):
                        paths.add(os.path.abspath(part))
            elif isinstance(cmd, str):
                for part in cmd.split():
                    if os.path.isfile(part):
                        paths.add(os.path.abspath(part))
    return sorted(paths)


def _resolve_trial_indices(include_trials: str | None, cfg, client) -> list[int]:
    """Resolve --include-trials argument to a list of trial indices."""
    if not include_trials:
        return []

    exp = client._experiment
    completed = {idx for idx, t in exp.trials.items() if t.status.name == "COMPLETED"}

    if include_trials == "all":
        trial_dest = cfg.get("optimization", {}).get("case_runner", {}).get("trial_destination", "trials")
        if os.path.isdir(trial_dest):
            entries = os.listdir(trial_dest)
            has_dirs = any(os.path.isdir(os.path.join(trial_dest, d)) for d in entries)
            has_files = any(f.endswith(".json") and os.path.isfile(os.path.join(trial_dest, f)) for f in entries)
            if has_dirs or has_files:
                # Mix: prefer Ax trial index mapping to avoid fragile name parsing
                return sorted(exp.trials.keys())
        return sorted(exp.trials.keys())

    if include_trials == "best":
        opt_config = exp.optimization_config
        if hasattr(opt_config.objective, "objectives"):
            # MOO — return Pareto-optimal
            return _resolve_trial_indices("pareto", cfg, client)
        # SOO — return best
        try:
            _, _, idx, _ = client.get_best_parameterization(use_model_predictions=False)
            return [idx] if idx is not None else []
        except Exception:
            return []

    if include_trials == "pareto":
        try:
            from ax.service.utils.best_point_mixin import BestPointMixin
            frontier = client.get_pareto_optimal_parameters()
            return sorted(frontier.keys()) if frontier else []
        except Exception:
            return sorted(completed)[:5]  # fallback: first 5 completed

    # Comma-separated indices
    indices = []
    for part in include_trials.split(","):
        part = part.strip()
        if part.isdigit():
            indices.append(int(part))
    return indices


def pack(config_path: str, include_trials: str | None = None,
         skip_patterns: list[str] | None = None, output: str | None = None) -> str:
    """Pack a foamBO experiment into a .foambo HDF5 archive.

    Args:
        config_path: Path to the YAML config file.
        include_trials: Trial selection: 'best', 'pareto', 'all', or comma-separated indices.
        skip_patterns: Additional glob patterns to exclude.
        output: Output file path (default: {experiment_name}.foambo).

    Returns:
        Path to the created archive.
    """
    from omegaconf import OmegaConf
    from .config import load_config
    from .common import set_experiment_name
    from .orchestrate import StoreOptions

    cfg = load_config(config_path)
    exp_name = cfg["experiment"]["name"]
    set_experiment_name(exp_name)

    # Resolve paths
    case_runner = cfg.get("optimization", {}).get("case_runner", {})
    base_case_path = case_runner.get("template_case", "./case")
    artifacts_folder = case_runner.get("artifacts_folder", "./artifacts")
    trial_destination = case_runner.get("trial_destination", "trials")

    # Load client state
    store_cfg = StoreOptions.model_validate(dict(cfg["store"]))
    client_state_path = os.path.join(artifacts_folder, f"{exp_name}_client_state.json")
    if not os.path.exists(client_state_path):
        log.error(f"Client state not found: {client_state_path}")
        raise FileNotFoundError(client_state_path)

    # Load client for trial selection
    client = store_cfg.load()
    exp = client._experiment
    n_completed = sum(1 for t in exp.trials.values() if t.status.name == "COMPLETED")

    # Resolve trials
    trial_indices = _resolve_trial_indices(include_trials, cfg, client)

    # Build skip patterns
    all_skip = list(DEFAULT_SKIP)
    if skip_patterns:
        all_skip.extend(skip_patterns)

    # Output path
    archive_path = output or f"{exp_name}.foambo"

    # Collect script paths
    script_paths = _collect_script_paths(OmegaConf.to_container(cfg, resolve=True))

    # Build manifest
    objectives = []
    opt_config = exp.optimization_config
    if hasattr(opt_config.objective, "objectives"):
        for obj in opt_config.objective.objectives:
            for mn in obj.metric_names:
                objectives.append(mn)
    else:
        for mn in opt_config.objective.metric_names:
            objectives.append(mn)

    manifest = {
        "foambo_version": VERSION,
        "packed_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": exp_name,
        "n_trials": len(exp.trials),
        "n_completed": n_completed,
        "objectives": objectives,
        "included_trials": trial_indices,
        "skip_patterns": all_skip,
        "original_paths": {
            "config": os.path.abspath(config_path),
            "base_case": os.path.abspath(base_case_path),
            "artifacts": os.path.abspath(artifacts_folder),
            "trial_destination": os.path.abspath(trial_destination),
        },
    }

    print(f"Packing experiment '{exp_name}' → {archive_path}")
    print(f"  Trials: {len(exp.trials)} total, {n_completed} completed")
    if trial_indices:
        print(f"  Including trial folders: {trial_indices}")
    print(f"  Skip patterns: {all_skip}")

    with h5py.File(archive_path, "w") as f:
        # Attributes
        f.attrs["foambo_version"] = VERSION
        f.attrs["experiment_name"] = exp_name
        f.attrs["packed_at"] = manifest["packed_at"]

        # Manifest
        f.create_dataset("manifest", data=json.dumps(manifest).encode(), **_compression())

        # Config YAML
        with open(config_path, "rb") as cf:
            f.create_dataset("config", data=cf.read(), **_compression())
        print(f"  + config ({os.path.getsize(config_path)} bytes)")

        # Client state JSON
        with open(client_state_path, "rb") as cs:
            f.create_dataset("client_state", data=cs.read(), **_compression())
        print(f"  + client_state ({os.path.getsize(client_state_path)} bytes)")

        # Base case
        if os.path.isdir(base_case_path):
            grp = f.create_group("base_case")
            n, sz = _store_directory(grp, base_case_path, all_skip)
            print(f"  + base_case/ ({n} files, {sz / 1024 / 1024:.1f} MB)")
        else:
            log.warning(f"Base case not found: {base_case_path}")

        # Scripts
        if script_paths:
            grp = f.create_group("scripts")
            for sp in script_paths:
                rel = os.path.basename(sp)
                with open(sp, "rb") as sf:
                    data = sf.read()
                ds = grp.create_dataset(rel, data=data, **_compression())
                ds.attrs["mode"] = os.stat(sp).st_mode & 0o777
                ds.attrs["is_symlink"] = False
                ds.attrs["original_path"] = sp
            print(f"  + scripts/ ({len(script_paths)} files)")

        # Artifacts
        if os.path.isdir(artifacts_folder):
            grp = f.create_group("artifacts")
            n, sz = _store_directory(grp, artifacts_folder, all_skip)
            print(f"  + artifacts/ ({n} files, {sz / 1024 / 1024:.1f} MB)")

        # Trials
        if trial_indices:
            trials_grp = f.create_group("trials")
            for tidx in trial_indices:
                trial = exp.trials.get(tidx)
                if trial is None:
                    log.warning(f"Trial {tidx} not found in experiment")
                    continue
                # Find trial folder OR caseless json file
                trial_dir = None
                trial_file = None
                run_meta = trial.run_metadata or {}
                case_path = run_meta.get("case_path") or run_meta.get("job", {}).get("case_path")
                if case_path and os.path.isdir(case_path):
                    trial_dir = case_path
                elif case_path and os.path.isfile(case_path):
                    trial_file = case_path
                else:
                    # Try common naming patterns (directory, then flat-file caseless)
                    import glob
                    for pattern in [f"{exp_name}_trial_*_{tidx}", f"trial_{tidx}", f"*_trial_{tidx:04d}"]:
                        matches = glob.glob(os.path.join(trial_destination, pattern))
                        if matches:
                            trial_dir = matches[0]
                            break
                    if not trial_dir:
                        for pattern in [f"{exp_name}_trial_*.json", f"trial_{tidx}.json"]:
                            matches = glob.glob(os.path.join(trial_destination, pattern))
                            if matches:
                                trial_file = matches[0]
                                break
                if trial_dir and os.path.isdir(trial_dir):
                    tgrp = trials_grp.create_group(str(tidx))
                    tgrp.attrs["status"] = trial.status.name
                    tgrp.attrs["parameters"] = json.dumps(trial.arm.parameters if trial.arm else {})
                    n, sz = _store_directory(tgrp, trial_dir, all_skip)
                    print(f"  + trials/{tidx}/ ({n} files, {sz / 1024 / 1024:.1f} MB)")
                elif trial_file and os.path.isfile(trial_file):
                    tgrp = trials_grp.create_group(str(tidx))
                    tgrp.attrs["status"] = trial.status.name
                    tgrp.attrs["parameters"] = json.dumps(trial.arm.parameters if trial.arm else {})
                    tgrp.attrs["caseless"] = True
                    tgrp.attrs["filename"] = os.path.basename(trial_file)
                    with open(trial_file, "rb") as tf:
                        data = tf.read()
                    ds = tgrp.create_dataset("__file__", data=data, **_compression())
                    ds.attrs["mode"] = os.stat(trial_file).st_mode & 0o777
                    print(f"  + trials/{tidx} (caseless file, {len(data)} bytes)")
                else:
                    log.warning(f"Trial {tidx} folder/file not found")

    archive_size = os.path.getsize(archive_path)
    print(f"\nArchive created: {archive_path} ({archive_size / 1024 / 1024:.1f} MB)")
    return archive_path


def unpack(archive_path: str, output_dir: str | None = None) -> str:
    """Unpack a .foambo HDF5 archive.

    Args:
        archive_path: Path to the .foambo archive.
        output_dir: Target directory (default: ./{experiment_name}/).

    Returns:
        Path to the unpacked directory.
    """
    from omegaconf import OmegaConf

    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with h5py.File(archive_path, "r") as f:
        # Read manifest
        manifest = json.loads(bytes(f["manifest"][()]).decode())
        exp_name = manifest["experiment_name"]
        pack_version = manifest.get("foambo_version", "unknown")

        if pack_version != VERSION:
            log.warning(f"Archive packed with foamBO v{pack_version}, current is v{VERSION}")

        target = output_dir or f"./{exp_name}"
        os.makedirs(target, exist_ok=True)

        print(f"Unpacking '{exp_name}' → {target}/")
        print(f"  Packed: {manifest.get('packed_at', '?')}")
        print(f"  Trials: {manifest.get('n_trials', '?')} total, {manifest.get('n_completed', '?')} completed")
        if manifest.get("included_trials"):
            print(f"  Included trial folders: {manifest['included_trials']}")

        # Config
        config_bytes = bytes(f["config"][()])
        config_path = os.path.join(target, "config.yaml")

        # Client state
        if "client_state" in f:
            cs_dir = os.path.join(target, "artifacts")
            os.makedirs(cs_dir, exist_ok=True)
            cs_path = os.path.join(cs_dir, f"{exp_name}_client_state.json")
            with open(cs_path, "wb") as out:
                out.write(bytes(f["client_state"][()]))
            print(f"  + artifacts/ (client state)")

        # Base case
        if "base_case" in f:
            bc_dir = os.path.join(target, "base_case")
            n = _extract_directory(f["base_case"], bc_dir)
            print(f"  + base_case/ ({n} files)")

        # Scripts
        if "scripts" in f:
            scripts_dir = os.path.join(target, "scripts")
            n = _extract_directory(f["scripts"], scripts_dir)
            print(f"  + scripts/ ({n} files)")

        # Artifacts (non-client-state files)
        if "artifacts" in f:
            art_dir = os.path.join(target, "artifacts")
            n = _extract_directory(f["artifacts"], art_dir)
            print(f"  + artifacts/ ({n} files)")

        # Trials
        if "trials" in f:
            trials_dir = os.path.join(target, "trials")
            os.makedirs(trials_dir, exist_ok=True)
            for tidx_str in f["trials"]:
                tgrp = f["trials"][tidx_str]
                if tgrp.attrs.get("caseless", False):
                    fname = tgrp.attrs.get("filename", f"{exp_name}_trial_{tidx_str}.json")
                    if isinstance(fname, bytes):
                        fname = fname.decode()
                    out_path = os.path.join(trials_dir, fname)
                    with open(out_path, "wb") as out:
                        out.write(bytes(tgrp["__file__"][()]))
                    mode = tgrp["__file__"].attrs.get("mode", 0o644)
                    os.chmod(out_path, mode)
                    print(f"  + trials/{fname} (caseless file)")
                else:
                    tdir = os.path.join(trials_dir, f"{exp_name}_trial_{tidx_str}")
                    n = _extract_directory(tgrp, tdir)
                    print(f"  + trials/{tidx_str}/ ({n} files)")

        # Rewrite config paths
        cfg = OmegaConf.load_from_string(config_bytes.decode())
        OmegaConf.update(cfg, "optimization.case_runner.template_case", "./base_case", force_add=True)
        OmegaConf.update(cfg, "optimization.case_runner.artifacts_folder", "./artifacts", force_add=True)
        OmegaConf.update(cfg, "optimization.case_runner.trial_destination", "./trials", force_add=True)
        OmegaConf.update(cfg, "store.read_from", "json", force_add=True)

        # Rewrite metric script paths to ./scripts/
        if "scripts" in f:
            metrics = OmegaConf.to_container(cfg.get("optimization", {}).get("metrics", []))
            for m in metrics:
                for key in ("command", "progress"):
                    cmd = m.get(key)
                    if not cmd:
                        continue
                    if isinstance(cmd, list):
                        m[key] = [f"./scripts/{os.path.basename(c)}" if os.path.basename(c) in os.listdir(os.path.join(target, "scripts")) else c for c in cmd]
                    elif isinstance(cmd, str):
                        parts = cmd.split()
                        m[key] = " ".join(f"./scripts/{os.path.basename(p)}" if os.path.basename(p) in os.listdir(os.path.join(target, "scripts")) else p for p in parts)
            OmegaConf.update(cfg, "optimization.metrics", metrics, force_add=True)

        with open(config_path, "w") as out:
            out.write(OmegaConf.to_yaml(cfg))
        print(f"  + config.yaml (paths rewritten)")

    print(f"\nUnpacked to: {os.path.abspath(target)}/")
    print(f"  To view: foamBO --no-opt --config {config_path} ++store.read_from=json")
    return target
