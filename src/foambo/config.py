"""
Configuration system for foamBO.
Supports:
- Loading config from YAML
- OmegaConf-based structure
- Generating a default config
- Overriding config values from CLI (e.g., foamBO --run ++problem.name=TEST)
"""

import os
import sys
import yaml
from omegaconf import OmegaConf, DictConfig
from typing import Any, Dict, Optional

from .common import DEFAULT_CONFIG

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
        yaml.dump(OmegaConf.to_container(default, resolve=True), f, default_flow_style=False)
    print(f"Default config written to {path}")

def get_default_config() -> DictConfig:
    """
    Return a DictConfig object with the default configuration.
    """
    default = {
        'problem': {
            'name': 'SingleObjF1',
            'template_case': 'case',
            'type': 'optimization',
            'models': 'auto',
            'parameters': {
                'x': {
                    'type': 'range',
                    'value_type': 'float',
                    'bounds': [-200, 200],
                    'log_scale': False,
                }
            },
            'scopes': {
                '/FxDict': {
                    'x': 'x',
                }
            },
            'objectives': {
                'F1': {
                    'mode': 'shell',
                    'command': 'python3 benchmark.py --F F1 --k 1 --m 0 --lb 0.01',
                    'threshold': 80,
                    'minimize': True,
                    'lower_is_better': True,
                }
            }
        },
        'meta': {
            'clone_destination': './trials/',
            'case_run_mode': 'local',
            'case_run_command': 'echo 0',
            'metric_value_mode': 'local',
            'n_trials': 300,
            'n_parallel_trials': 5,
            'ttl_trial': 3600,
            'init_poll_wait': 0.1,
            'poll_factor': 1.5,
            'timeout': 10,
            'use_saasbo': False,
            'n_pareto_points': 5,
            'stopping_strategy': {
                'improvement_bar': 1e-4,
                'min_trials': 30,
                'window_size': 10,
            }
        }
    }
    return OmegaConf.create(default)

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
        # Support nested keys
        OmegaConf.update(cfg, key, yaml.safe_load(val), merge=True)
    return cfg
