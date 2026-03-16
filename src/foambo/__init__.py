"""foamBO: Multi-objective Bayesian Optimization for OpenFOAM cases."""

from ._version import VERSION
from .api import FoamBO

__all__ = [
    "VERSION",
    "FoamBO",
]
