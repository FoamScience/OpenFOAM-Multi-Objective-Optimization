"""
Common constants and shared definitions for the foambo package.
"""

import hashlib, os, shutil, time
from typing import List, Literal, Dict
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
    return [c.replace("$CASE_PATH", case_path_str).replace("$CASE_NAME", case_name_str) for c in command]

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
    """No-op preprocessor for caseless (pure-Python) optimization.

    Uses a virtual path as the trial "case path" — no directories are
    created on disk since there are no files to store.
    """
    _counter: int = 0

    def setup(self, parameters: dict, cfg) -> dict:
        NoCasePreprocessor._counter += 1
        virtual_path = f"<caseless_trial_{NoCasePreprocessor._counter}>"
        fake = _FakeCasePath(virtual_path)
        return {"case": fake, "casename": virtual_path}


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
            with FoamFile(param_file_path) as paramFile:
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
