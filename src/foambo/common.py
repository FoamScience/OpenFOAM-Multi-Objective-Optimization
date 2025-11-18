"""
Common constants and shared definitions for the foambo package.
"""

import inspect, hashlib, os, shutil, time
from types import UnionType
from typing import List, Literal, Dict, Union, get_type_hints, get_origin, get_args
from omegaconf import DictConfig, OmegaConf, ListConfig
from foamlib import FoamCase, FoamFile
import numpy as np
import inspect, ast

from logging import Logger
from ax.utils.common.logger import get_logger
log : Logger = get_logger(__name__)

VERSION = "1.1.3"

# Default config filename
DEFAULT_CONFIG = "foamBO.yaml"

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

def resolve_dotted_callable(path: str):
    raise TypeError(f"dotted string paths to factories is not supported, use actual references in __nested_fields__")

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


def validate_args(cls_or_func, data: DictConfig, *, path: str="", allow_none=False) -> bool:
    """
    Check if `data` satisfies all required arguments of `cls_or_func`, recusriveley piercing through class members.
    Requires classes to have __nested_fields__: Dict[str] naming functions used to create members
    """
    sig = inspect.signature(cls_or_func)
    parameters = sig.parameters
    required = [
        name for name, param in parameters.items()
        if name != 'self' and
           param.default is inspect.Parameter.empty and
           param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    ]
    optional = {
        name: param.default for name, param in parameters.items()
        if name != 'self' and
           param.default is not inspect.Parameter.empty and
           param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    expected = required + list(optional.keys())
    missing = [name for name in required if name not in data]
    extra = [key for key in data if key not in expected]
    if extra or missing:
        msg = f"Validation error in '{path or cls_or_func.__qualname__}':"
        if extra:
            msg += f"\n  ❌ Unexpected fields supplied: {extra}"
            msg += f"\n  But we require the following fields: {sorted(expected)}"
        if missing:
            msg += f"\n  ❗Missing required fields: {missing}"
        if optional:
            msg += f"\n    Optional (=defaulted) fields:"
            for name, default in optional.items():
                msg += f"\n      {name} = {default!r}"
        raise TypeError(msg)
    for name in required:
        if data.get(name) is None:
            expected_type = get_type_hints(cls_or_func).get(name, "unknown")
            isNoneAllowed = expected_type is type(None) or (get_origin(expected_type) in (Union, UnionType) and type(None) in get_args(expected_type))
            if not isNoneAllowed:
                raise TypeError(
                    f"Field '{name}' is required and cannot be null "
                    f"(expected type: {expected_type})"
                )
    nested_fields = getattr(cls_or_func, "__nested_fields__", {})
    for key, validators in nested_fields.items():
        if key not in data:
            continue
        if not isinstance(validators, list):
            validators = [validators]
        for validator in validators:
            if isinstance(validator, str):
                validator_func = resolve_dotted_callable(validator)
            else:
                validator_func = getattr(cls_or_func, validator) if isinstance(validator, str) else validator
            if not callable(validator_func):
                raise TypeError(f"{validator} is not callable on {cls_or_func}")
            try:
                validate_args(validator_func, data[key], path=f"{path}.{key}" if path else key)
                break
            except TypeError:
                continue
        else:
            raise TypeError(f"None of the validators passed for '{path}.{key}'")
    return True

def has_unexpected_args(cls, data: DictConfig) -> List:
    """Return `data` keys not accepted by `cls.__init__`."""
    sig = inspect.signature(cls.__init__)
    accepted_args = {
        name for name, param in sig.parameters.items()
        if name != 'self'
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    has_kwargs = any(
        param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
    )
    extra = [name for name in data.keys() if name not in accepted_args]
    return extra

def func_params(func):
    params = inspect.signature(func).parameters
    conf_dict = {}
    for name, param in params.items():
        if name == "self" or name == "name":
            continue
        default = param.default if param.default is not inspect.Parameter.empty else None
        conf_dict[name] = {
            "default": default,
            "type": str(param.annotation),
        }
    return OmegaConf.to_yaml(OmegaConf.create(conf_dict))


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


def preprocess_case(parameters, cfg):
    """
        Copy template, and substitute parameter values
    """
    from .common import DEFAULT_HASH_LENGTH, validate_args
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

    validate_args(prepare_case, cfg['template_case'])
    case = prepare_case(**cfg['template_case'])
    
    # Process parameters which require file copying
    if "files" in cfg['templating'].keys() and cfg['templating']['files']:
        for elm,elmv in cfg['templating']['files'].items():
            shutil.copyfile(
                os.path.join(case.path, elmv.template + "." + parameters[elm]),
                os.path.join(case.path, elmv.template)
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


def preprocess_config(cls, config: Dict) -> Dict:
    if not config:
        return config
    processed = dict(config)
    nested_fields = getattr(cls, "__nested_fields__", {})

    for key, factories in nested_fields.items():
        if key not in config:
            continue
        nested_config = config[key]
        if not isinstance(factories, list):
            factories = [factories]
        
        factory_fns = []
        for factory in factories:
            if isinstance(factory, str):
                factory_fn = resolve_dotted_callable(factory)
            else:
                factory_fn = factory
            if not callable(factory_fn):
                raise TypeError(f"{cls.__name__}.{factory} is not callable")
            factory_fns.append(factory_fn)

        def try_factories(nested_item):
            errors = []
            for factory_fn in factory_fns:
                sig = inspect.signature(factory_fn)
                has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
                if has_var_kwargs:
                    return factory_fn(**nested_item)
                try:
                   return factory_fn(**nested_item)
                except TypeError as e:
                    expected_args = []
                    for p in sig.parameters.values():
                        if p.name == 'self':
                            continue
                        if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                            continue
                        if p.default is inspect.Parameter.empty:
                            expected_args.append(p.name)
                        else:
                            expected_args.append(f"{p.name}={p.default!r}")
                    provided = list(nested_item.keys()) if nested_item else 'empty configuration'
                    errors.append(
                        f"- {factory_fn.__qualname__}:\n"
                        f"    Provided: {provided}\n"
                        f"    Expected: {expected_args}"
                    )
            if errors:
                raise TypeError(
                    f"'{key}' configuration couldn't be processed; valid {key} entries:\n" +
                    "\n".join(errors)
                )
        if nested_config is None:
            expected_type = get_type_hints(cls).get(key, "unknown")
            isNoneAllowed = expected_type is type(None) or (get_origin(expected_type) in (Union, UnionType) and type(None) in get_args(expected_type))
            if not isNoneAllowed:
                raise TypeError(f"'{key}' config is null, expected a dict or list of dicts to configure {key} elements.")
        else:
            if isinstance(nested_config, list | ListConfig):
                processed[key] = [try_factories(item) for item in nested_config]
            elif isinstance(nested_config, Dict | DictConfig):
                processed[key] = try_factories(nested_config)
            else:
                raise TypeError(f"Expected dict or list of dicts for nested config '{key}', got {type(nested_config)}")
    return processed


def instantiate_with_nested_fields(cls, config: Dict):
    processed = preprocess_config(cls, config)
    validate_args(cls.__init__, processed)
    return cls(**processed)

def unixlike_filename(filename: str) -> str:
    import re
    name = filename.lower().strip().replace(" ", "_")
    name = re.sub(r"[^a-z0-9._-]", "_", name)
    return name
