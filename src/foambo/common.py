"""
Common constants and shared definitions for the foambo package.
"""

import hashlib, json, os, shutil, time
from contextlib import contextmanager
from typing import List, Literal, Dict, Optional
from omegaconf import DictConfig, OmegaConf
from foamlib import FoamCase, FoamFile
from pydantic import BaseModel, ConfigDict
import numpy as np
import ast

from logging import Logger
from ax.utils.common.logger import get_logger
log : Logger = get_logger(__name__)


class FoamBOBaseModel(BaseModel):
    """Base model for all FoamBO configuration models."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

from ._version import VERSION, DEFAULT_CONFIG

# Default hash length
DEFAULT_HASH_LENGTH = 8

# Global experience name, set from config; not best practice but simplifies things
EXPERIMENT_NAME="DEFAULT_EXPERIMENT"
def set_experiment_name(new_name:str):
    global EXPERIMENT_NAME
    EXPERIMENT_NAME = new_name
def get_experiment_name():
    global EXPERIMENT_NAME
    return EXPERIMENT_NAME

# Status mapping for SLURM jobs
SLURM_STATUS_MAP = {
    "RUNNING": "RUNNING",
    "CONFIGURING": "RUNNING",
    "COMPLETING": "RUNNING",
    "PENDING": "RUNNING",
    "PREEMPTED": "FAILED",
    "FAILED": "FAILED",
    "SUSPENDED": "ABANDONED",
    "TIMEOUT": "ABANDONED",
    "STOPPED": "EARLY_STOPPED",
    "CANCELED": "EARLY_STOPPED",
    "CANCELLED+": "EARLY_STOPPED",
    "COMPLETED": "COMPLETED",
}

# HTML report common header
HTML_CARD_HEADER="""<link rel="stylesheet" href="https://unpkg.com/@picocss/pico@latest/css/pico.min.css">
<style>
  .group-header, .group-subtitle, p, h1, h2, h3, h4, h5, h6 { margin: 1.1rem; margin: 1.1rem; }
  .card { margin: 1.1rem; }
</style>"""
HTML_CARD_SCRIPT="""<script>
document.querySelectorAll('.card').forEach(card => {
    const summary = card.querySelector('.card-header details summary');
    if (summary && summary.textContent.trim().startsWith("Summary")) {
        card.style.gridColumn = "1 / -1";
    }
});
</script>
"""

def parse_outcome_for_metric(
        outcome: bytes | str | float,
        metric: str
    ) -> tuple[np.int64, np.float64 | tuple[np.float64, np.float64]]:
    """
    Parses metric observation using the following format to translate to Ax metrics:
    <scalar>   -> (0, <mean>)
    (<mean, sem>) -> (0, (<mean>, <sem>))
    first int (0) being the progression value
    Supports NaN values (string 'nan', 'NaN', float('nan'), np.nan)
    """

    zero = np.int64(0)
    if isinstance(outcome, bytes):
        outcome = outcome.decode("utf-8")
    if isinstance(outcome, str):
        outcome = outcome.rstrip()
    if isinstance(outcome, (float, int)):
        return (zero, np.float64(outcome))
    if isinstance(outcome, str) and outcome.strip().lower() in {"nan", "none"}:
        return (zero, np.nan)
    try:
        parsed = ast.literal_eval(outcome)
        if isinstance(parsed, (float, int)):
            return (zero, np.float64(parsed))
        elif isinstance(parsed, tuple) and len(parsed) == 2:
            if all(isinstance(x, (float, float)) for x in parsed):
                mean, sem = parsed
                return (zero, (np.float64(mean), np.float64(sem)))
    except:
        log.warning(f"Metric '{metric}' output could not be parsed,\n"
            f"expected format <scalar> or (<mean>, <sem>); got: `{outcome}`\n"
            f"a NAN value will otherwise be assigned. The metric command output was: {outcome}")
    return (zero, np.nan)


def process_input_command(command: str | List[str] | None, case: FoamCase):
    """
        Process commands from config files to provide some flexibility
    """
    if not command:
        return None
    if isinstance(command, str):
        command = list(command.split())
    case_path_str = str(case.path)
    case_name_str = str(case.name)
    return [c
            .replace("$FOAMBO_CASE_PATH", case_path_str).replace("$CASE_PATH", case_path_str)
            .replace("$FOAMBO_CASE_NAME", case_name_str).replace("$CASE_NAME", case_name_str)
            for c in command]

def assign_foam_path(foamfile, dotted_path, new_value):
    """
        Given a FoamFile, assign a value to a dotted path 
    """
    keys = dotted_path.split(".")
    for k in keys[:-1]:
        try:
            k = int(k)
        except ValueError:
            pass
        foamfile = foamfile[k]
    last = keys[-1]
    try:
        last = int(last)
    except ValueError:
        pass
    foamfile[last] = new_value
    return foamfile[last]


def _detect_format(path: str, explicit: Optional[str] = None) -> str:
    """Pick file-format handler. Explicit wins. Else extension. Else openfoam."""
    if explicit:
        return explicit
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return "json"
    if ext in (".yaml", ".yml"):
        return "yaml"
    return "openfoam"


@contextmanager
def _open_param_file(path: str, fmt: Optional[str] = None):
    """Open a parameter file for dotted-path mutation across formats.

    Yields a mapping-like object that supports __getitem__/__setitem__.
    Writes back on exit for json/yaml (foamlib handles persistence itself).
    """
    resolved = _detect_format(path, fmt)
    if resolved == "openfoam":
        with FoamFile(path) as ff:
            yield ff
        return
    if resolved == "json":
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = {}
        yield data
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return
    if resolved == "yaml":
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True
        if os.path.exists(path):
            with open(path, "r") as f:
                data = yaml.load(f) or {}
        else:
            data = {}
        yield data
        with open(path, "w") as f:
            yaml.dump(data, f)
        return
    raise ValueError(f"Unknown parameter file format: {resolved}")


class CasePreprocessor:
    """Protocol for case setup before trial execution.

    Subclass this to support non-OpenFOAM cases. The default implementation
    (FoamCasePreprocessor) clones an OpenFOAM template case and substitutes
    parameter values using foamlib.

    A preprocessor receives the trial parameterization and runner config,
    and must return a dict with at least:
        {"case": <case_object_with_.path>, "casename": "<trial_path>"}
    """
    def setup(self, parameters: dict, cfg) -> dict:
        raise NotImplementedError("Subclass must implement setup()")


class FoamCasePreprocessor(CasePreprocessor):
    """Default OpenFOAM case preprocessor using foamlib."""
    def setup(self, parameters: dict, cfg) -> dict:
        return preprocess_case(parameters, cfg)


class _FakeCasePath:
    """Minimal stand-in for FoamCase.path when no case directory exists."""
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
    def __str__(self):
        return self.path


class NoCasePreprocessor(CasePreprocessor):
    """Caseless preprocessor: writes a single JSON file per trial.

    Output: `{trial_destination}/{EXPERIMENT_NAME}_trial_{hash}.json`
    containing the full parameter dict. No case directory is created.
    Any `variable_substitution` entries with empty/`.` file target this JSON.
    """
    def setup(self, parameters: dict, cfg) -> dict:
        hash = hashlib.md5()
        encoded = repr(OmegaConf.to_yaml(parameters)).encode()
        hash.update(encoded + f"{time.time()}".encode())
        hash = hash.hexdigest()[:DEFAULT_HASH_LENGTH]

        trial_dest = cfg['template_case'].get('trial_destination', 'trials') \
            if isinstance(cfg, (dict, DictConfig)) and 'template_case' in cfg \
            else 'trials'
        os.makedirs(trial_dest, exist_ok=True)
        trial_name = f"{EXPERIMENT_NAME}_trial_{hash}"
        trial_path = os.path.join(trial_dest, f"{trial_name}.json")

        payload = dict(parameters)
        with open(trial_path, "w") as f:
            json.dump(payload, f, indent=2)

        # Apply variable_substitution entries targeting the trial JSON
        if isinstance(cfg, (dict, DictConfig)) and 'templating' in cfg:
            for elmt in cfg['templating'].get('variables', []) or []:
                tgt = (elmt.get('file') or '').strip()
                if tgt in ('', '.', '/'):
                    # Entry targets the trial JSON itself — already written as flat dict.
                    # Dotted paths in parameter_scopes allow nested rewrites.
                    elmv = elmt['parameter_scopes']
                    with _open_param_file(trial_path, 'json') as pf:
                        for p in elmv:
                            if p in parameters:
                                assign_foam_path(pf, elmv[p], parameters[p])

        fake = _FakeCasePath(trial_path)
        fake.name = trial_name
        return {"case": fake, "casename": trial_path}


# Module-level default preprocessor; can be swapped by users
case_preprocessor: CasePreprocessor = FoamCasePreprocessor()


def preprocess_case(parameters, cfg):
    """
        Copy template, and substitute parameter values
    """
    from .common import DEFAULT_HASH_LENGTH
    data = {}
    hash = hashlib.md5()
    encoded = repr(OmegaConf.to_yaml(parameters)).encode()
    hash.update(encoded+f"{time.time()}".encode())
    hash = hash.hexdigest()[:DEFAULT_HASH_LENGTH]

    # Clone template case
    def prepare_case(
        path: str,
        mode: Literal["local", "slurm"],
        trial_destination: str,
        runner: str | None = None,
        log_runner: bool | None = False,
        remote_status_query: str | None = None,
        remote_early_stop: str | None = None,
    ):
        templateCase = FoamCase(path)
        newcase = os.path.join(trial_destination, f"{EXPERIMENT_NAME}_trial_"+hash)
        case = templateCase.clone(newcase)
        # foamlib clone preserves symlinks (symlinks=True). Relative symlinks
        # break when the trial dir is at a different depth than the base case.
        # Dereference any broken symlinks by replacing them with the real file.
        for root, dirs, files in os.walk(str(case.path)):
            for name in files + dirs:
                fpath = os.path.join(root, name)
                if os.path.islink(fpath) and not os.path.exists(fpath):
                    # Broken symlink — resolve relative to original base case
                    link_target = os.readlink(fpath)
                    resolved = os.path.normpath(os.path.join(os.path.dirname(
                        os.path.join(str(path), os.path.relpath(fpath, str(case.path)))),
                        link_target))
                    if os.path.exists(resolved):
                        os.unlink(fpath)
                        if os.path.isdir(resolved):
                            shutil.copytree(resolved, fpath)
                        else:
                            shutil.copy2(resolved, fpath)
                        log.debug("Dereferenced broken symlink: %s → %s", name, resolved)
                    else:
                        log.warning("Broken symlink in trial (target not found): %s → %s",
                                    fpath, link_target)
        return case

    case = prepare_case(**cfg['template_case'])
    
    # Process parameters which require file copying
    if "files" in cfg['templating'].keys() and cfg['templating']['files']:
        for entry in cfg['templating']['files']:
            param_name = entry['parameter']
            template_path = entry['file_path']
            if param_name in parameters:
                shutil.copyfile(
                    os.path.join(case.path, template_path + "." + parameters[param_name]),
                    os.path.join(case.path, template_path)
                )
    # Process parameters with foamlib
    if "variables" in cfg['templating'].keys():
        for elmt in cfg['templating']['variables']:
            elm = elmt['file']
            elmv = elmt['parameter_scopes']
            param_file_path = os.path.join(case.path, elm.lstrip('/'))
            with _open_param_file(param_file_path, elmt.get('format')) as paramFile:
                for param in elmv:
                    try:
                        if param in parameters.keys():
                            log.debug(f"##### replacing {elmv[param]} with {parameters[param]} in {param_file_path}")
                            new_val = assign_foam_path(paramFile, elmv[param], parameters[param])
                            log.debug(f"##### new value = {new_val}")
                    except:
                        log.warn(
                            f"Something went wrong when substituting:\n"
                            f"{elmv[param]} in {param_file_path} with {parameters[param]}\n"
                            f"If it's not a dependent parameter, this is serious..."
                        )
    data["case"] = case
    data["casename"] = case.path
    return data


def unixlike_filename(filename: str) -> str:
    import re
    name = filename.lower().strip().replace(" ", "_")
    name = re.sub(r"[^a-z0-9._-]", "_", name)
    return name
