"""
Common constants and shared definitions for the foambo package.
"""

from ax.modelbridge.registry import Models

# Default config filename
DEFAULT_CONFIG = "foamBO.yaml"

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

# Manual model registry name
MANUAL_MODEL_NAME = "MANUAL"

# List of supported Algorithms
SUPPORTED_MODELS = {
    #"ALEBO": Models.ALEBO,
    #"ALEBO_INITIALIZER": Models.ALEBO_INITIALIZER,
    "BOTORCH": Models.BOTORCH_MODULAR,
    "BOTORCH_MODULAR": Models.BOTORCH_MODULAR,
    "BO_MIXED": Models.BO_MIXED,
    "CONTEXT_SACBO": Models.CONTEXT_SACBO,
    "EMPIRICAL_BAYES_THOMPSON": Models.EMPIRICAL_BAYES_THOMPSON,
    "FACTORIAL": Models.FACTORIAL,
    "FULLYBAYESIAN": Models.FULLYBAYESIAN,
    "FULLYBAYESIANMOO": Models.FULLYBAYESIANMOO,
    "FULLYBAYESIANMOO_MTGP": Models.FULLYBAYESIANMOO_MTGP,
    "FULLYBAYESIAN_MTGP": Models.FULLYBAYESIAN_MTGP,
    "GPEI": Models.GPEI,
    "MOO": Models.MOO,
    "SOBOL": Models.SOBOL,
    "ST_MTGP": Models.ST_MTGP,
    "ST_MTGP_NEHVI": Models.ST_MTGP_NEHVI,
    "THOMPSON": Models.THOMPSON,
    "UNIFORM": Models.UNIFORM,
}
