"""
Configuration system for foamBO.
Supports:
- Loading config from YAML
- OmegaConf-based structure
- Generating a default config
- Overriding config values from CLI (e.g., foamBO --run ++problem.name=TEST)
"""

import yaml
from omegaconf import OmegaConf, DictConfig
from typing import Optional

from .common import DEFAULT_CONFIG
from .default_config import get_default_config

from logging import Logger
from ax.utils.common.logger import get_logger
log : Logger = get_logger(__name__)


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """
    Load configuration from a YAML file and return as OmegaConf DictConfig.
    If no path is given, use the default config filename.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return OmegaConf.create(data)

def save_default_config(path: Optional[str] = None):
    """
    Generate and save a default configuration YAML file.
    """
    if path is None:
        path = DEFAULT_CONFIG
    default = get_default_config()
    with open(path, 'w') as f:
        yaml.dump(default, f, default_flow_style=False, sort_keys=False)
    log.info(f"Default config written to {path}")

def override_config(cfg: DictConfig, overrides: list[str]) -> DictConfig:
    """
    Apply CLI-style overrides to the config (e.g., ["problem.name=TEST", "meta.n_trials=5"]).
    """
    for override in overrides:
        # Only support ++key=value or key=value
        if not (override.startswith('++') or '=' in override):
            continue
        keyval = override.lstrip('+').split('=', 1)
        if len(keyval) != 2:
            continue
        key, val = keyval
        OmegaConf.update(cfg, key, yaml.safe_load(val), merge=True)
    return cfg
