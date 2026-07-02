"""Robust + multi-fidelity glue — thin adapter over ``ax_robust_multifid``.

foamBO's own robust/MF implementation was extracted into the shared package
``ax_robust_multifid`` (pinned to ax-platform 1.3.1 / botorch 0.18.1) so foamBO and UPO
consume one copy instead of divergent forks. This module now:

- keeps foamBO's ``RobustOptimizationConfig`` (its ``risk_measure`` field is vestigial and
  ignored by the library, but preserved for config/YAML/api back-compat);
- re-exports the library's ``SubstituteContextFeatures`` / ``RobustAcquisition`` /
  ``cycle_context`` and the MF ``MFHVKGAcquisition``;
- wraps ``resolve_context_points`` / ``augment_generator_specs`` / ``build_context_tensor``
  so the existing ``(…, exp_cfg, …)`` call sites keep working — the library reads parameter
  ``groups``/``bounds`` off param objects, whereas foamBO stores groups on ``exp_cfg`` via
  ``get_parameter_groups()``, so we build a per-parameter shim list from ``exp_cfg``.

Importing this module fires the library's Ax JSON-registry side effects (so saved robust/MF
experiments (de)serialize).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Literal

from pydantic import Field

from ax_robust_multifid.robustness import (
    SubstituteContextFeatures,
    RobustAcquisition,
    cycle_context,
    context_param_names as _lib_context_param_names,
    context_dim_indices as _lib_context_dim_indices,
    resolve_context_points as _lib_resolve_context_points,
    augment_generator_specs as _lib_augment_generator_specs,
    build_context_tensor as _lib_build_context_tensor,
)
from ax_robust_multifid.multifidelity import MultiFidHVKGAcquisition as MFHVKGAcquisition

log = logging.getLogger("ax.foambo.robustness")

__all__ = [
    "RobustOptimizationConfig",
    "RobustOptimizationConfig_factory",
    "SubstituteContextFeatures",
    "RobustAcquisition",
    "MFHVKGAcquisition",
    "cycle_context",
    "resolve_context_points",
    "augment_generator_specs",
    "build_context_tensor",
]


# ---------------------------------------------------------------------------
# Config (foamBO-owned; duck-typed into the library functions)
# ---------------------------------------------------------------------------

def _foambo_base():
    """Lazy import to avoid circular dependency."""
    from foambo.orchestrate import FoamBOBaseModel
    return FoamBOBaseModel


def RobustOptimizationConfig_factory():
    """Build the config class lazily to avoid import-time circular refs."""
    Base = _foambo_base()

    class _RobustOptimizationConfig(Base):
        """Robust optimization across environmental/context variables."""
        context_groups: list[str] = Field(
            description="Parameter group names treated as context (environmental) variables")
        risk_measure: Literal["auto", "cvar", "mars"] = Field(
            default="auto",
            description="Risk measure. 'auto' picks MARS for MOO, CVaR for SOO")
        robustness: float = Field(
            default=0.5, ge=0.0, le=1.0,
            description=(
                "0=risk-neutral, 1=most conservative.  Maps directly to "
                "CVaR/MARS alpha (clamped >= 0.05).  Higher alpha optimises over "
                "the worst (1-alpha) fraction of context scenarios."))
        context_points: list[dict[str, float]] | None = Field(
            default=None,
            description="Explicit context scenarios. If None, auto-generated via Sobol")
        context_samples: int = Field(
            default=10, ge=2,
            description="Number of Sobol samples when auto-generating context points")
        context_constraints: list[str] = Field(
            default=[],
            description="Inequality filters on context points (e.g. 'flowRate >= 0.01')")

    return _RobustOptimizationConfig


# Module-level class (created once)
RobustOptimizationConfig = RobustOptimizationConfig_factory()


# ---------------------------------------------------------------------------
# exp_cfg -> params adapter
# ---------------------------------------------------------------------------

def _params_from_exp_cfg(exp_cfg):
    """Build the per-parameter shim list the library expects from ``exp_cfg``.

    The library reads ``name``/``bounds``/``parameter_type``/``groups`` off each param;
    foamBO keeps groups on ``exp_cfg`` (``get_parameter_groups()``), so stamp them on.
    Ordering matches ``exp_cfg.parameters`` (== the feature-tensor column order the
    context-dim indices are computed against).
    """
    groups = exp_cfg.get_parameter_groups()
    return [
        SimpleNamespace(
            name=p.name,
            bounds=getattr(p, "bounds", None),
            parameter_type=getattr(p, "parameter_type", "float"),
            groups=groups.get(p.name, []),
        )
        for p in exp_cfg.parameters
    ]


def resolve_context_points(robust_cfg, exp_cfg) -> None:
    """Auto-generate ``context_points`` via Sobol if not explicit (mutates in place)."""
    return _lib_resolve_context_points(robust_cfg, _params_from_exp_cfg(exp_cfg))


def augment_generator_specs(client, exp_cfg, robust_cfg, multifidelity: bool = False):
    """Wire robust (± MF) config into the client's generation strategy."""
    return _lib_augment_generator_specs(
        client, _params_from_exp_cfg(exp_cfg), robust_cfg, multifidelity=multifidelity)


def build_context_tensor(robust_cfg, exp_cfg):
    """Build a normalized ``(n_w, d_ctx)`` tensor from ``context_points``."""
    return _lib_build_context_tensor(robust_cfg, _params_from_exp_cfg(exp_cfg))


def _get_context_param_names(robust_cfg, exp_cfg) -> list[str]:
    """Ordered names of parameters tagged with any of ``context_groups``."""
    return _lib_context_param_names(robust_cfg, _params_from_exp_cfg(exp_cfg))


def _get_context_dim_indices(robust_cfg, exp_cfg) -> list[int]:
    """Integer indices of context parameters in the feature tensor."""
    return _lib_context_dim_indices(robust_cfg, _params_from_exp_cfg(exp_cfg))
