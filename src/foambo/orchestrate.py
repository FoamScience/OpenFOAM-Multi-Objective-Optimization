from ax.utils.common.logger import get_logger
from ax.api.client import AnalysisCardBase, PercentileEarlyStoppingStrategy, Client
from ax.orchestration.orchestrator_options import OrchestratorOptions
from ax.api.types import TParameterization
from ax.early_stopping.strategies import AndEarlyStoppingStrategy, OrEarlyStoppingStrategy, ThresholdEarlyStoppingStrategy
from ax.generation_strategy.generation_node import MaxGenerationParallelism
from ax.storage.json_store.registry import CenterGenerationNode
from foambo.metrics import FoamJobRunner, LocalJobMetric
from .common import *
from .common import FoamBOBaseModel
from pydantic import Field, field_validator, model_validator
from typing import Any, Iterable, List, Literal, Dict
from functools import reduce
from ax.global_stopping.strategies import BaseGlobalStoppingStrategy, ImprovementGlobalStoppingStrategy
from ax.api.configs import StorageConfig, RangeParameterConfig, ChoiceParameterConfig
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.adapter.registry import Generators
from ax.generation_strategy.transition_criterion import (
    MinTrials, AutoTransitionAfterGen,
    IsSingleObjective, AuxiliaryExperimentCheck,
)
from omegaconf import DictConfig, ListConfig

def create_from_map(cfg, map):
    cfg = dict(cfg)
    transition_type = cfg.pop("type", None)
    if transition_type is None:
        from pprint import pformat
        raise ValueError(f"Missing 'type' key in config:\n{pformat(cfg)}\n"
                         f"Supported types: {pformat(list(map.keys()))}")
    if transition_type == "none":
        return None
    if transition_type not in map:
        raise ValueError(f"Type {transition_type} not supported. Supported ones are:\n{list(map.keys())}")
    # Backward compat: metric_names was renamed to metric_signatures in Ax 1.2+
    if "metric_names" in cfg and "metric_signatures" not in cfg:
        cfg["metric_signatures"] = cfg.pop("metric_names")
    # Recursively parse composite strategies (or/and) whose children are dicts
    for key in ("left", "right"):
        if key in cfg and isinstance(cfg[key], (dict, DictConfig)):
            cfg[key] = create_from_map(cfg[key], map)
    # Filter out kwargs not accepted by the target class
    cls = map[transition_type]
    import inspect
    valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    unknown = set(cfg.keys()) - valid_params
    if unknown:
        from ax.utils.common.logger import get_logger
        get_logger(__name__).debug(f"Ignoring unknown kwargs for {cls.__name__}: {unknown}")
        cfg = {k: v for k, v in cfg.items() if k in valid_params}
    return cls(**cfg)

EARLY_STOPPER_MAP = {
    "none": None,
    "percentile": PercentileEarlyStoppingStrategy,
    "threshold": ThresholdEarlyStoppingStrategy,
    "and": AndEarlyStoppingStrategy,
    "or": OrEarlyStoppingStrategy,
}


class DimensionalityReductionOptions(FoamBOBaseModel):
    """Automatic parameter screening based on Sobol sensitivity analysis.

    For high-dimensional problems (15+ parameters), consider coupling with SAASBO
    in the generation strategy: use SAASBO for the initial exploration phase (its
    sparsity-inducing priors via MCMC identify important parameters automatically),
    then transition to BOTORCH_MODULAR + dimensionality reduction to hard-drop
    confirmed-unimportant parameters for faster iterations.
    """
    enabled: bool = Field(default=False, description="Enable automatic dimensionality reduction")
    after_trials: int = Field(default=10, description=(
        "Run sensitivity analysis after this many completed trials. "
        "Must be enough for the BO model to be fitted (past the SOBOL phase)."
    ))
    min_importance: float = Field(default=0.05, description=(
        "Parameters with first-order Sobol index below this threshold are fixed. "
        "Range [0, 1]. Example: 0.05 means parameters contributing less than 5%% of variance are dropped."
    ))
    fix_at: Literal["best", "center"] = Field(default="best", description=(
        "How to choose the fixed value for dropped parameters. "
        "'best' uses the value from the best trial so far. "
        "'center' uses the midpoint of the parameter bounds."
    ))
    max_fix_fraction: float = Field(default=0.5, description=(
        "Never fix more than this fraction of total parameters. "
        "At least one parameter always remains active. "
        "Example: 0.5 means at most half the parameters can be fixed."
    ))


class ConfigOrchestratorOptions(FoamBOBaseModel):
    """Controls for timeouts, poll times, early-stopping and convergence criteria."""
    max_trials: int = Field(description="Maximum number of trials to run, baseline included", examples=[20])
    parallelism: int = Field(description="Maximum number of trials running at the same time", examples=[3])
    global_stopping_strategy: Any = Field(description=(
        "When to stop the optimization based on the best improvement so far. "
        "Stops generating new trials once improvement over `window_size` consecutive trials "
        "falls below `improvement_bar * IQR`."
    ))
    early_stopping_strategy: Any = Field(description=(
        "Stops trials mid-way if they are not interesting enough. "
        "Supported types: `percentile`, `threshold`, or composite `and`/`or`. "
        "Requires streaming metrics with a `progress` command."
    ))
    tolerated_trial_failure_rate: float = Field(default=0.5, description=(
        "Maximum fraction of failed trials before the experiment is terminated. "
        "Set to 1.0 to never stop due to failures, or 0.0 to stop at the first failure."
    ))
    min_failed_trials_for_failure_rate_check: int = Field(default=5, description=(
        "Minimum number of failed trials before checking the failure rate. "
        "Prevents early termination due to a few initial failures."
    ))
    initial_seconds_between_polls: float = Field(default=1, description=(
        "Initial wait time in seconds before polling trial status. "
        "Fractional values (e.g. 0.1) are supported for fast Python callables. "
        "The actual wait time increases with the backoff factor."
    ), examples=[60])
    min_seconds_before_poll: float = Field(default=1.0, description=(
        "Minimum wait time in seconds between polls. "
        "Even with backoff, the orchestrator waits at least this long."
    ))
    seconds_between_polls_backoff_factor: float = Field(default=1.5, description=(
        "Multiplier to increase wait time between polls (exponential backoff). "
        "Example: with initial=60s and factor=1.5, polls happen at 60s, 90s, 135s, ..."
    ))
    timeout_hours: int | None = Field(default=None, description=(
        "Maximum runtime in hours for the entire experiment, excluding baseline trial. "
        "Set to None for no timeout."
    ), examples=[48])
    ttl_seconds_for_trials: int | None = Field(default=None, description=(
        "Time-to-live in seconds for candidate (queued, not yet running) trials. "
        "Candidates exceeding this are marked STALE by Ax and may be re-suggested. "
        "Does NOT stop or kill already-running trials."
    ), examples=[2400])
    trial_timeout: int | float | str | None = Field(default=None, description=(
        "Kill running trials that exceed this time limit. "
        "An integer or float is an absolute timeout in seconds (e.g. 3600). "
        "A string like '5*baseline' means 5x the baseline (T0) execution time. "
        "Killed trials are marked FAILED. Set to None to disable."
    ), examples=[3600, "5*baseline"])
    fatal_error_on_metric_trouble: bool = Field(default=False, description=(
        "Whether to terminate the experiment if metric evaluation fails. "
        "True: stop experiment on any metric failure. False: mark trial as failed and continue."
    ))
    immutable_search_space: bool = Field(default=True, description=(
        "Whether the search space is fixed after experiment creation. "
        "Most users should keep True for reproducibility."
    ))
    metric_eval_timeout: int = Field(default=600, description=(
        "Timeout in seconds for metric evaluation commands. "
        "If a metric command takes longer, the subprocess is killed."
    ))
    remote_query_timeout: int = Field(default=60, description=(
        "Timeout in seconds for remote status queries and remote kill commands."
    ))
    progression_cmd_timeout: int = Field(default=30, description=(
        "Timeout in seconds for progression source commands. "
        "These run each poll cycle so should be fast."
    ))
    dependency_action_timeout: int = Field(default=120, description=(
        "Timeout in seconds for trial dependency actions (e.g. file copy, mapFields)."
    ))
    process_reap_timeout: int = Field(default=5, description=(
        "Timeout in seconds to wait for a killed process to exit before giving up."
    ))
    api_port: int = Field(default=8098, description=(
        "Port for the live REST API server. Set to 0 to disable. "
        "The API is served in a background thread for web UI access."
    ))
    api_host: str = Field(default="127.0.0.1", description=(
        "Bind address for the live REST API server."
    ))
    allow_pvpython_upload: bool = Field(default=True, description=(
        "Allow uploading and running pvpython visualization scripts from the web dashboard. "
        "Set to false to disable script upload/execution for security."
    ))
    run_progress_commands_each_poll: bool = Field(default=True, description=(
        "When True, foamBO runs the configured progress commands each poll cycle "
        "to actively pull streaming metric values. When False, streaming metrics "
        "are only received via HTTP push (trials must POST to "
        "``/api/v1/trials/{idx}/push/metrics``). "
        "Set to False when using OpenFOAM function objects or Allrun scripts that "
        "push metrics directly — avoids redundant subprocess spawns. Process "
        "status polling (alive/dead) is always active regardless."
    ))
    dimensionality_reduction: DimensionalityReductionOptions = Field(
        default_factory=lambda: DimensionalityReductionOptions(enabled=False),
        description="Automatic parameter fixing based on Sobol sensitivity analysis after an exploration phase"
    )

    @field_validator("dimensionality_reduction", mode="before")
    @classmethod
    def parse_dim_reduction(cls, v):
        if isinstance(v, dict | DictConfig):
            return DimensionalityReductionOptions.model_validate(dict(v))
        return v

    @field_validator("global_stopping_strategy", mode="before")
    @classmethod
    def parse_global_stopping(cls, v):
        if isinstance(v, dict | DictConfig):
            v = dict(v)
            gss = ImprovementGlobalStoppingStrategy(
                min_trials=v["min_trials"],
                window_size=v["window_size"],
                improvement_bar=v["improvement_bar"],
                inactive_when_pending_trials=True,
            )
            # Wrap to handle Ax crash when inferring objective thresholds
            # with incomplete data (e.g. all trials infeasible).
            # See: ax/service/utils/best_point.py:1378
            _orig = gss._should_stop_optimization
            def _safe_should_stop(experiment, **kwargs):
                try:
                    return _orig(experiment, **kwargs)
                except (ValueError, RuntimeError) as e:
                    import logging
                    get_logger(__name__).warning(
                        f"Global stopping evaluation failed (incomplete data): {e}. "
                        f"Continuing optimization.")
                    return False, f"Skipped due to error: {e}"
            gss._should_stop_optimization = _safe_should_stop
            return gss
        return v

    @field_validator("early_stopping_strategy", mode="before")
    @classmethod
    def parse_early_stopping(cls, v):
        if isinstance(v, dict | DictConfig):
            return create_from_map(dict(v), EARLY_STOPPER_MAP)
        return v

    @model_validator(mode="after")
    def resolve_composite_stoppers(self):
        def _resolve(strategy):
            if strategy is None:
                return strategy
            if hasattr(strategy, 'left'):
                if isinstance(strategy.left, dict | DictConfig):
                    strategy.left = create_from_map(dict(strategy.left), EARLY_STOPPER_MAP)
                _resolve(strategy.left)
            if hasattr(strategy, 'right'):
                if isinstance(strategy.right, dict | DictConfig):
                    strategy.right = create_from_map(dict(strategy.right), EARLY_STOPPER_MAP)
                _resolve(strategy.right)
            return strategy
        _resolve(self.early_stopping_strategy)
        return self

    @model_validator(mode="after")
    def unlock_search_space_for_dim_reduction(self):
        if self.dimensionality_reduction.enabled:
            object.__setattr__(self, 'immutable_search_space', False)
        return self

    def to_scheduler_options(self) -> OrchestratorOptions:
        return OrchestratorOptions(
            total_trials=self.max_trials,
            max_pending_trials=self.parallelism,
            global_stopping_strategy=self.global_stopping_strategy,
            early_stopping_strategy=self.early_stopping_strategy,
            tolerated_trial_failure_rate=self.tolerated_trial_failure_rate,
            min_failed_trials_for_failure_rate_check=self.min_failed_trials_for_failure_rate_check,
            init_seconds_between_polls=self.initial_seconds_between_polls,
            min_seconds_before_poll=self.min_seconds_before_poll,
            ttl_seconds_for_trials=self.ttl_seconds_for_trials,
            seconds_between_polls_backoff_factor=self.seconds_between_polls_backoff_factor,
            validate_metrics=self.fatal_error_on_metric_trouble,
            enforce_immutable_search_space_and_opt_config=self.immutable_search_space,
            logging_level=log.level,
        )


import time as _time
from ax.api.client import Orchestrator

class TimedOrchestrator(Orchestrator):
    """Orchestrator that tracks wall-clock time spent in candidate generation.

    ``_gen_time_s`` accumulates the total seconds spent inside
    ``_prepare_trials`` (model fitting + acquisition function optimization).
    Read it after ``run_all_trials`` to get the generation overhead.
    """
    _gen_time_s: float = 0.0

    def _prepare_trials(self, *args, **kwargs):
        t0 = _time.perf_counter()
        result = super()._prepare_trials(*args, **kwargs)
        self._gen_time_s += _time.perf_counter() - t0
        return result


class ExperimentOptions(FoamBOBaseModel):
    """Settings for the experiment's search space."""
    name: str = Field(description="Experiment name, used to identify artifacts like reports and state files",
                       examples=["Sample"])
    description: str = Field(description="A short experiment description",
                              examples=["Sample experiment description"])
    parameter_constraints: List[str] = Field(description=(
        "Inequalities between optimization parameters. Linear constraints "
        "(e.g. ``'x1 <= x2 + 5'``) are passed to Ax directly. Nonlinear "
        "constraints (e.g. ``'x1 * x2 <= 10'``, ``'x1**2 + x2**2 <= 100'``) "
        "are detected automatically and forwarded to BoTorch's acquisition "
        "function optimizer."
    ), examples=[[]])
    parameters: List[Any] = Field(description=(
        "List of RangeParameterConfig or ChoiceParameterConfig. "
        "Range parameters need `bounds`, choice parameters need `values`."
    ))

    @field_validator("parameters", mode="before")
    @classmethod
    def parse_parameters(cls, v):
        if v and len(v) > 0 and isinstance(v[0], dict | DictConfig):
            return cls.generate_parameter_space(v)
        return v

    @model_validator(mode="after")
    def set_name(self):
        set_experiment_name(self.name)
        return self

    def _classify_constraints(self) -> tuple[list[str], list[str]]:
        """Split parameter_constraints into (linear, nonlinear)."""
        if not self.parameter_constraints:
            return [], []
        from foambo.constraints import classify_constraints
        param_names = [p.name for p in self.parameters]
        return classify_constraints(self.parameter_constraints, param_names)

    def to_dict(self):
        linear, _ = self._classify_constraints()
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "parameter_constraints": linear,
        }

    def get_nonlinear_constraints(self) -> list[str]:
        """Return the nonlinear constraint expressions (not passed to Ax)."""
        _, nonlinear = self._classify_constraints()
        return nonlinear

    def get_parameter_groups(self) -> dict[str, list[str]]:
        """Return {param_name: [group1, group2, ...]} mapping."""
        return getattr(self.__class__, '_parameter_groups', {})

    def get_group_params(self, group: str) -> list[str]:
        """Return parameter names belonging to a given group."""
        return [name for name, groups in self.get_parameter_groups().items()
                if group in groups]

    @classmethod
    def generate_parameter_space(cls, params_cfg):
        """
        Generate parameter configs from input configuration
        """
        if isinstance(params_cfg, dict | DictConfig):
            items = params_cfg.items()
        elif isinstance(params_cfg, list | ListConfig):
            items = [(p["name"], p) for p in params_cfg]
        else:
            raise TypeError(f"Invalid parameter format: {type(params_cfg)}")

        l = []
        for key, item in items:
            e = {
                "name": key,
                **item
            }
            # Extract and store groups (not an Ax parameter field)
            groups = e.pop("groups", None)
            if groups:
                if not hasattr(cls, '_parameter_groups'):
                    cls._parameter_groups = {}
                cls._parameter_groups[e["name"]] = list(groups)
            if "values" in e.keys():
                l.append(ChoiceParameterConfig(**e))
            elif "bounds" in e.keys():
                l.append(RangeParameterConfig(**e))
            else:
                raise ValueError(f"Invalid parameter configuration for {e['name']}; expecting either 'values' or 'bounds' in {e}")
        return l


def _get_transform_registry() -> Dict[str, type]:
    """Lazy registry mapping transform names to classes."""
    from ax.adapter.transforms.cast import Cast
    from ax.adapter.transforms.map_key_to_float import MapKeyToFloat
    from ax.adapter.transforms.remove_fixed import RemoveFixed
    from ax.adapter.transforms.choice_encode import OrderedChoiceToIntegerRange
    from ax.adapter.transforms.one_hot import OneHot
    from ax.adapter.transforms.int_to_float import IntToFloat
    from ax.adapter.transforms.log import Log
    from ax.adapter.transforms.logit import Logit
    from ax.adapter.transforms.winsorize import Winsorize
    from ax.adapter.transforms.derelativize import Derelativize
    from ax.adapter.transforms.bilog_y import BilogY
    from ax.adapter.transforms.standardize_y import StandardizeY
    return {
        "Cast": Cast,
        "MapKeyToFloat": MapKeyToFloat,
        "RemoveFixed": RemoveFixed,
        "OrderedChoiceToIntegerRange": OrderedChoiceToIntegerRange,
        "OneHot": OneHot,
        "IntToFloat": IntToFloat,
        "Log": Log,
        "Logit": Logit,
        "Winsorize": Winsorize,
        "Derelativize": Derelativize,
        "BilogY": BilogY,
        "StandardizeY": StandardizeY,
    }

# The default Ax transform order (all transforms)
DEFAULT_TRANSFORMS = [
    "Cast", "MapKeyToFloat", "RemoveFixed", "OrderedChoiceToIntegerRange",
    "OneHot", "IntToFloat", "Log", "Logit", "Winsorize", "Derelativize",
    "BilogY", "StandardizeY",
]


def resolve_transforms(
    only: List[str] | None = None,
    exclude: List[str] | None = None,
) -> list[type]:
    """Resolve transform names to classes.

    Args:
        only: If set, use only these transforms (in order).
        exclude: If set, remove these from the default list.

    Returns:
        Ordered list of transform classes.
    """
    registry = _get_transform_registry()
    if only is not None:
        names = only
    elif exclude is not None:
        names = [n for n in DEFAULT_TRANSFORMS if n not in exclude]
    else:
        names = DEFAULT_TRANSFORMS
    result = []
    for name in names:
        if name not in registry:
            raise ValueError(f"Unknown transform '{name}'. Available: {list(registry.keys())}")
        result.append(registry[name])
    return result


class ModelSpecConfig(FoamBOBaseModel):
    """Configuration for a trial generator algorithm."""
    generator_enum: Literal[
        "SOBOL",
        "FACTORIAL",
        "SAASBO",
        "SAAS_MTGP",
        "THOMPSON",
        "BOTORCH_MODULAR",
        "EMPIRICAL_BAYES_THOMPSON",
        "EB_ASHR",
        "UNIFORM",
        "ST_MTGP",
        "BO_MIXED",
    ] = Field(description="Generator algorithm name (e.g. SOBOL for random init, BOTORCH_MODULAR for BO)")
    model_kwargs: Dict | None = Field(default=None, description="Extra keyword arguments passed to the generator (e.g. seed for SOBOL)")
    transforms: List[str] | None = Field(default=None, description=(
        "Explicit list of transform names to use, in order. "
        "Overrides the default Ax transform chain. "
        "Available: Cast, MapKeyToFloat, RemoveFixed, OrderedChoiceToIntegerRange, "
        "OneHot, IntToFloat, Log, Logit, Winsorize, Derelativize, BilogY, StandardizeY"
    ))
    exclude_transforms: List[str] | None = Field(default=None, description=(
        "Transform names to remove from the default chain. "
        "Example: ``[BilogY, Winsorize]`` to disable output compression and outlier clipping"
    ))

    def to_generator_spec(self) -> GeneratorSpec:
        if not hasattr(Generators, self.generator_enum):
            from pprint import pformat
            raise ValueError(f"{self.generator_enum} is not a supported generator; supported specs:\n"
                             f"{pformat([name for name, value in vars(Generators).items() if not name.startswith('_')])}")
        model = Generators[self.generator_enum]
        kwargs = dict(self.model_kwargs) if self.model_kwargs else {}
        if self.transforms is not None or self.exclude_transforms is not None:
            kwargs["transforms"] = resolve_transforms(
                only=self.transforms, exclude=self.exclude_transforms)
        return GeneratorSpec(
            generator_enum=model,
            generator_kwargs=kwargs,
        )

TRANSITION_MAP = {
    "max_trials": MinTrials,
    "min_trials": MinTrials,
    "auto_transition_after_gen": AutoTransitionAfterGen,
    "is_single_objective": IsSingleObjective,
    "max_generation_parallelism": MaxGenerationParallelism,
    "auxiliary_experiment_check": AuxiliaryExperimentCheck,
}


class GenerationNodeConfig(FoamBOBaseModel):
    """Configuration for a generation node in a custom generation strategy."""
    node_name: str = Field(description="Unique name for this generation node")
    generator_specs: List[ModelSpecConfig] = Field(description="List of generator algorithms to use in this node")
    transition_criteria: List[Any] = Field(default=[], description="Conditions for transitioning to the next node (e.g. max_trials, min_trials)")

    @field_validator("generator_specs", mode="before")
    @classmethod
    def parse_specs(cls, v):
        if v and isinstance(v[0], dict | DictConfig):
            return [ModelSpecConfig.model_validate(dict(item)) for item in v]
        return v

    @field_validator("transition_criteria", mode="before")
    @classmethod
    def parse_transitions(cls, v):
        if not v:
            return v
        return [create_from_map(dict(item), TRANSITION_MAP) if isinstance(item, dict | DictConfig) else item for item in v]

    def to_generation_node(self) -> GenerationNode:
        spec_list = [spec.to_generator_spec() for spec in self.generator_specs]
        return GenerationNode(
            name=self.node_name,
            generator_specs=spec_list,
            transition_criteria=self.transition_criteria if self.transition_criteria else None,
        )


class CenterGenerationNodeConfig(FoamBOBaseModel):
    """Generates a single trial at the center of the search space, then transitions."""
    next_node: str = Field(description="Name of the generation node to transition to after the center trial")

    def to_generation_node(self):
        return CenterGenerationNode(next_node_name=self.next_node)


class ManualGenerationNode(GenerationNode):
    parameters: TParameterization
    def __init__(self, node_name: str, parameters: TParameterization):
        super().__init__(name=node_name, generator_specs=[])
        self.parameters = parameters

    def gen(
        self,
        *,
        experiment,
        pending_observations,
        skip_fit = False,
        data = None,
        **gs_gen_kwargs,
    ) :
        return GeneratorRun(
            arms=[Arm(name=self.name, parameters=self.parameters)],
            optimization_config=experiment.optimization_config,
            search_space=experiment.search_space,

        )


class SeedDataNodeConfig(FoamBOBaseModel):
    """Config for a SeedDataNode: loads trial data from CSV or foamBO JSON state."""
    file_path: str = Field(description=(
        "Path to a CSV or foamBO JSON state file. "
        "CSV columns are parameters + metrics (scalar or `(mean, sem)`). "
        "JSON is a foamBO-saved Ax Client state (loaded via ``FoamBO.load``)."
    ))
    filter: Dict[str, Any] | None = Field(default=None, description=(
        "Optional per-parameter filter. Short form (scalar) = exact match. "
        "Long form: ``{value, tolerance, direction}`` where direction is "
        "``both`` (default), ``left`` or ``right``."
    ))
    drop_params: List[str] | None = Field(default=None, description=(
        "Parameter names to drop from imported data. The target experiment "
        "must NOT contain these parameters in its search space."
    ))
    next_node: str = Field(description="Name of the next generation node to transition to after seeding")

    def to_generation_node(self) -> GenerationNode:
        return SeedDataNode(
            file_path=self.file_path,
            filter=self.filter,
            drop_params=self.drop_params,
            next_node=self.next_node,
        )


class SeedDataNode(GenerationNode):
    """Generation node that seeds the experiment with pre-existing trial data.

    Data is attached to the Client during strategy setup (via ``load_into``).
    The node's ``gen()`` emits an empty GeneratorRun and immediately transitions
    to ``next_node`` via ``AutoTransitionAfterGen``.
    """
    def __init__(self, file_path: str, filter=None, drop_params=None, next_node=None):
        if not next_node:
            raise ValueError("SeedDataNode requires `next_node`")
        super().__init__(
            name="seed_data",
            generator_specs=[],
            transition_criteria=[AutoTransitionAfterGen(transition_to=next_node)],
        )
        self._file_path = file_path
        self._filter = dict(filter) if filter else {}
        self._drop_params = list(drop_params or [])
        self._next_node = next_node
        self._loaded = False

    def gen(self, *, experiment, pending_observations, skip_fit=False, data=None, **gs_gen_kwargs):
        return GeneratorRun(
            arms=[],
            optimization_config=experiment.optimization_config,
            search_space=experiment.search_space,
        )

    def load_into(self, client: Client):
        """Attach seed trial data to ``client``. Idempotent."""
        if self._loaded:
            return
        from pathlib import Path
        path = Path(self._file_path)
        if not path.exists():
            raise FileNotFoundError(f"SeedDataNode: {self._file_path} not found")
        if path.suffix.lower() == ".json":
            self._load_foambo_json(client, path)
        else:
            self._load_csv(client, path)
        self._loaded = True

    def _load_csv(self, client: Client, path):
        import pandas as pd
        df = pd.read_csv(path, header=0)
        df = self._filter_rows_df(df)
        if self._drop_params:
            df = df.drop(columns=self._drop_params, errors="ignore")
        metrics = list(client._experiment.metrics.keys())
        params = [p for p in client._experiment.parameters.keys() if p not in self._drop_params]
        missing_metrics = [c for c in metrics if c not in df.columns]
        missing_params = [c for c in params if c not in df.columns]
        if missing_params or missing_metrics:
            raise ValueError(
                f"SeedDataNode: missing columns in {path} — "
                f"parameters: {missing_params}, metrics: {missing_metrics}"
            )
        if df.empty:
            raise ValueError(f"SeedDataNode: no rows survive filter in {path}")
        n_trials = client._experiment.fetch_data().df.shape[0]
        new_trial_idx = n_trials - 1
        grouped = df.groupby(params) if params else [(None, df)]
        for _, group_df in grouped:
            new_trial_idx += 1
            progressions = group_df.shape[0]
            idx_in_grp = -1
            for _, row in group_df.iterrows():
                idx_in_grp += 1
                row_params = row[params].to_dict()
                row_metrics = {k: parse_outcome_for_metric(v, k) for k, v in row[metrics].to_dict().items()}
                if idx_in_grp == 0:
                    client.attach_trial(parameters=row_params, arm_name=f"{new_trial_idx}_seed")
                if idx_in_grp == progressions - 1:
                    client.complete_trial(trial_index=new_trial_idx, raw_data=row_metrics, progression=idx_in_grp)
                    continue
                client.attach_data(trial_index=new_trial_idx, raw_data=row_metrics, progression=idx_in_grp)
        log.info(f"SeedDataNode loaded {new_trial_idx - n_trials + 1} trials from CSV {path}")

    def _load_foambo_json(self, client: Client, path):
        from .api import FoamBO
        saved_name = get_experiment_name()
        try:
            result = FoamBO.load(name=path.stem, artifacts=str(path.parent), backend="json")
            src_client = result.client
        finally:
            set_experiment_name(saved_name)
        src_exp = src_client._experiment
        tgt_params = [p for p in client._experiment.parameters.keys() if p not in self._drop_params]
        metrics = list(client._experiment.metrics.keys())
        n_trials = client._experiment.fetch_data().df.shape[0]
        new_trial_idx = n_trials - 1
        attached = 0
        for trial in src_exp.trials.values():
            if not trial.status.is_completed:
                continue
            arm = trial.arm
            if arm is None:
                continue
            full_params = dict(arm.parameters)
            if not self._row_passes_filter(full_params):
                continue
            row_params = {k: v for k, v in full_params.items() if k not in self._drop_params}
            missing = [p for p in tgt_params if p not in row_params]
            if missing:
                raise ValueError(
                    f"SeedDataNode: JSON trial {trial.index} missing target params {missing}"
                )
            try:
                data_df = trial.fetch_data().df
            except Exception as e:
                log.warning(f"SeedDataNode: skipping trial {trial.index}, fetch_data failed: {e}")
                continue
            row_metrics: Dict[str, Any] = {}
            for m in metrics:
                m_rows = data_df[data_df["metric_name"] == m]
                if m_rows.empty:
                    continue
                last = m_rows.iloc[-1]
                sem = float(last["sem"]) if "sem" in last and last["sem"] == last["sem"] else 0.0
                row_metrics[m] = (float(last["mean"]), sem)
            if not row_metrics:
                continue
            new_trial_idx += 1
            client.attach_trial(parameters=row_params, arm_name=f"{new_trial_idx}_seed")
            client.complete_trial(trial_index=new_trial_idx, raw_data=row_metrics)
            attached += 1
        if attached == 0:
            raise ValueError(f"SeedDataNode: no trials survived filter in {path}")
        log.info(f"SeedDataNode loaded {attached} trials from foamBO JSON {path}")

    def _filter_rows_df(self, df):
        if not self._filter:
            return df
        import pandas as pd
        mask = pd.Series([True] * len(df), index=df.index)
        for key, spec in self._filter.items():
            val, tol, direction = self._parse_filter_spec(spec)
            col = df[key].astype(float)
            if direction == "both":
                mask &= (col >= val - tol) & (col <= val + tol)
            elif direction == "left":
                mask &= (col >= val - tol) & (col <= val)
            elif direction == "right":
                mask &= (col >= val) & (col <= val + tol)
            else:
                raise ValueError(f"SeedDataNode: unknown filter direction {direction}")
        return df[mask]

    def _row_passes_filter(self, params: dict) -> bool:
        if not self._filter:
            return True
        for key, spec in self._filter.items():
            if key not in params:
                return False
            val, tol, direction = self._parse_filter_spec(spec)
            v = float(params[key])
            if direction == "both" and not (val - tol <= v <= val + tol):
                return False
            if direction == "left" and not (val - tol <= v <= val):
                return False
            if direction == "right" and not (val <= v <= val + tol):
                return False
        return True

    @staticmethod
    def _parse_filter_spec(spec):
        if isinstance(spec, dict | DictConfig):
            s = dict(spec)
            return float(s["value"]), float(s.get("tolerance", 0.0)), s.get("direction", "both")
        return float(spec), 0.0, "both"


class TrialGenerationOptions(FoamBOBaseModel):
    """Controls over trial generation methods."""
    method: Literal["fast", "random_search", "custom"] = Field(description=(
        "How to generate new parameterizations. "
        "'fast' and 'random_search' use built-in Ax strategies. "
        "'custom' requires `generation_nodes` to be defined."
    ), examples=["custom"])
    generation_nodes: List[Any] | None = Field(default=None, description=(
        "List of generation nodes for 'custom' method. "
        "Each node specifies generator algorithms and transition criteria."
    ))
    initialization_budget: int | None = Field(default=5, description="Number of initial quasi-random trials before switching to BO")
    initialization_random_seed: int | None = Field(default=None, description="Random seed for reproducible initialization trials")
    initialize_with_center: bool = Field(default=True, description="Whether the first trial uses the center of the search space")
    use_existing_trials_for_initialization: bool = Field(default=True, description="Count pre-loaded trials toward the initialization budget")
    min_observed_initialization_trials: int | None = Field(default=None, description="Minimum observed init trials before transitioning to BO")
    allow_exceeding_initialization_budget: bool = Field(default=False, description="Allow more init trials than the budget if needed")

    @field_validator("generation_nodes", mode="before")
    @classmethod
    def parse_generation_nodes(cls, v):
        if not v:
            return v
        nodes = []
        for item in v:
            if isinstance(item, dict | DictConfig):
                item = dict(item)
                if "file_path" in item:
                    nodes.append(SeedDataNodeConfig.model_validate(item).to_generation_node())
                elif "next_node_name" in item:
                    nodes.append(CenterGenerationNode(**item))
                elif "next_node" in item:
                    nodes.append(CenterGenerationNodeConfig.model_validate(item).to_generation_node())
                else:
                    node_cfg = GenerationNodeConfig.model_validate(item)
                    nodes.append(node_cfg.to_generation_node())
            else:
                nodes.append(item)
        return nodes

    @model_validator(mode="after")
    def validate_method_nodes(self):
        if self.method == "custom" and not self.generation_nodes:
            raise ValueError("generation_nodes must be provided when method is 'custom'")
        if self.method != "custom" and self.generation_nodes:
            log.warning(f"generation_nodes is provided but method is {self.method}, not custom. The nodes will be ignored.")
        return self

    def to_dict(self):
        return {
            "method": self.method,
            "initialization_budget": self.initialization_budget,
            "initialization_random_seed": self.initialization_random_seed,
            "initialize_with_center": self.initialize_with_center,
            "use_existing_trials_for_initialization": self.use_existing_trials_for_initialization,
            "min_observed_initialization_trials": self.min_observed_initialization_trials,
            "allow_exceeding_initialization_budget": self.allow_exceeding_initialization_budget,
        }

    def set_generation_strategy(self, client):
        if self.method != "custom":
            client.configure_generation_strategy(**self.to_dict())
            return
        if not self.generation_nodes:
            raise ValueError("generation_nodes must be provided when method is 'custom'")
        gs = GenerationStrategy(name=f"+".join([node.name for node in self.generation_nodes]), nodes=self.generation_nodes)
        client.set_generation_strategy(gs)


class VariableSubstOptions(FoamBOBaseModel):
    """Maps parameters to OpenFOAM file fields for value substitution via foamlib."""
    file: str = Field(description="Relative path to the OpenFOAM file (e.g. `/0orig/field`)",
                       examples=["/0orig/field"])
    parameter_scopes: Dict[str, str] = Field(description="Mapping of parameter name to dotted path in the file (e.g. `x: someDict.x`)",
                                              examples=[{"x": "someDict.x"}])


class FileSubstOptions(FoamBOBaseModel):
    """Replaces a case file with a variant based on a choice parameter value."""
    parameter: str = Field(description="Name of the choice parameter whose value selects the file variant",
                            examples=["y"])
    file_path: str = Field(description="Relative path to the file (e.g. `/system/fvSolution`). Variants must exist as `<file_path>.<value>`",
                            examples=["/constant/y"])


class FoamJobRunnerOptions(FoamBOBaseModel):
    """Configuration for trial case dispatch and monitoring."""
    mode: Literal["local", "remote"] = Field(description="Execution mode: 'local' runs on this machine, 'remote' dispatches to a cluster",
                                              examples=["remote"])
    template_case: str = Field(description="Path to the OpenFOAM case used as a template for trials",
                                examples=["./case"])
    trial_destination: str = Field(description="Directory where trial case copies are created",
                                    examples=["./trials"])
    artifacts_folder: str = Field(description="Directory for reports, CSV exports, and experiment state files",
                                   examples=["./artifacts"])
    variable_substitution: List[VariableSubstOptions] = Field(description="Parameter-to-file-field mappings for value substitution via foamlib")
    file_substitution: List[FileSubstOptions] = Field(description="File replacement rules for choice parameters")
    runner: str | None = Field(default="", description=(
        "Shell command to run the trial (e.g. `./Allrun`). "
        "Supports `$FOAMBO_CASE_NAME`/`$FOAMBO_CASE_PATH` substitution. "
        "These are also available as environment variables, along with "
        "`$FOAMBO_PRE_INIT`, `$FOAMBO_PRE_MESH`, `$FOAMBO_PRE_SOLVE`, `$FOAMBO_POST_SOLVE` "
        "hook scripts from trial dependencies. Empty = no execution"),
                               examples=["./run_on_cluster.sh $FOAMBO_CASE_NAME"])
    log_runner: bool | None = Field(default=False, description="Write runner stdout to `log.runner` in the trial folder instead of discarding it")
    remote_status_query: str | None = Field(default="", description="Command returning SLURM-like job status (required for mode=remote)",
                                             examples=["./state_on_cluster.sh $FOAMBO_CASE_NAME"])
    remote_early_stop: str | None = Field(default="", description="Command to kill a remote job (e.g. `scancel --name $FOAMBO_CASE_NAME`). Required if early stopping is on",
                                           examples=["scancel --name $FOAMBO_CASE_NAME"])

    @field_validator("variable_substitution", mode="before")
    @classmethod
    def parse_variable_subst(cls, v):
        if v and isinstance(v[0], dict | DictConfig):
            return [VariableSubstOptions.model_validate(dict(item)) for item in v]
        return v

    @field_validator("file_substitution", mode="before")
    @classmethod
    def parse_file_subst(cls, v):
        if v and isinstance(v[0], dict | DictConfig):
            return [FileSubstOptions.model_validate(dict(item)) for item in v]
        return v

    @model_validator(mode="after")
    def validate_paths(self):
        if not os.path.isdir(self.template_case):
            raise ValueError(f"template_case directory does not exist: {self.template_case}")
        os.makedirs(self.trial_destination, exist_ok=True)
        os.makedirs(self.artifacts_folder, exist_ok=True)
        return self

    def to_runner(self):
        cfg = DictConfig({
            "template_case": {
                "mode": self.mode,
                "path": self.template_case,
                "trial_destination": self.trial_destination,
                "runner": self.runner,
                "log_runner": self.log_runner or False,
                "remote_status_query": self.remote_status_query,
                "remote_early_stop": self.remote_early_stop,
            },
            "templating": {
                "variables": [subst.model_dump() for subst in self.variable_substitution] if self.variable_substitution else [],
                "files": [subst.model_dump() for subst in self.file_substitution] if self.file_substitution else []
            }
        })
        runner = FoamJobRunner(cfg=cfg)
        return runner


class BaselineOptions(FoamBOBaseModel):
    """A set of parameter values to consider as a baseline for the optimization analysis."""
    parameters: TParameterization | None = Field(description=(
        "Parameter values for the baseline trial. Set to `null` to skip. "
        "If defined, a single trial runs before the generation strategy and counts toward max_trials."
    ), examples=[{"x": 0.8, "y": "zero"}])


class OptimizationOptions(FoamBOBaseModel):
    """Controls for the optimization algorithm, including objectives, constraints, and case dispatch."""
    objective: str = Field(description=(
        "What to optimize. Prefix with `-` to minimize, no prefix to maximize. "
        "Multi-objective: `-metric1, metric2`"
    ), examples=["-metric"])
    outcome_constraints: List[str] = Field(description=(
        "Linear objective constraints, e.g. `metric1 >= 0.5*baseline`, `metric2 <= 10`. "
        "Provide upper bounds for minimized metrics, lower bounds otherwise."
    ), examples=[["metric >= 0.9*baseline"]])
    objective_thresholds: List[str] | None = Field(default=None, description=(
        "Objective thresholds for multi-objective optimization. Defines the reference point "
        "for Pareto hypervolume computation. Format: `metric >= value` (for maximized) or "
        "`metric <= value` (for minimized). If not set, Ax infers them from data."
    ), examples=[["efficiency >= 0.3", "pressureHead >= 0.01", "torque <= 100"]])
    metrics: List[LocalJobMetric] = Field(description="List of metric evaluation commands (objectives and tracking metrics)")
    case_runner: FoamJobRunnerOptions = Field(description="Trial case dispatch and monitoring configuration")

    @field_validator("metrics", mode="before")
    @classmethod
    def parse_metrics(cls, v):
        if v and isinstance(v[0], dict | DictConfig):
            return [LocalJobMetric.model_validate(dict(item)) for item in v]
        return v

    @field_validator("case_runner", mode="before")
    @classmethod
    def parse_case_runner(cls, v):
        if isinstance(v, dict | DictConfig):
            return FoamJobRunnerOptions.model_validate(dict(v))
        return v

    def to_optimization_dict(self):
        return {
            "objective": self.objective,
            "outcome_constraints": self.outcome_constraints,
        }

    def to_objective_metrics_dict(self):
        return {
            "metrics": [m.to_metric() for m in self.metrics if m.name in self.objective],
        }

    def to_tracking_metrics_dict(self):
        return {
            "metrics": [m.to_metric() for m in self.metrics if not m.name in self.objective]
        }

    def to_runner_dict(self):
        return {
            "runner": self.case_runner.to_runner()
        }


class JSONBackendOptions(FoamBOBaseModel):
    """JSON file storage backend (default). No extra configuration needed."""
    pass


class SQLBackendOptions(FoamBOBaseModel):
    """SQL database storage backend."""
    url: str = Field(description="SQL database connection URL")


class StoreOptions(FoamBOBaseModel):
    """Controls where to store experiment state and where to read it from."""
    read_from: Literal["json", "sql", "nowhere"] = Field(description="Storage backend to load experiment state from. Use 'nowhere' for a fresh start",
                                                         examples=["json"])
    save_to: Literal["json", "sql"] = Field(description="Storage backend to save experiment state to",
                                             examples=["json"])
    backend_options: JSONBackendOptions | SQLBackendOptions = Field(description="Backend-specific settings (e.g. SQL database URL)",
                                                                     examples=[{"url": None}])

    @field_validator("backend_options", mode="before")
    @classmethod
    def parse_backend(cls, v):
        if isinstance(v, dict | DictConfig):
            v = dict(v)
            if v.get("url"):
                return SQLBackendOptions.model_validate(v)
            return JSONBackendOptions.model_validate({k: val for k, val in v.items() if k != "url"})
        return v

    def load(self) -> Client:
        if self.read_from == "nowhere":
            return Client()
        client = None
        if self.read_from == "json":
            client = Client.load_from_json_file(filepath=f"artifacts/{get_experiment_name()}_client_state.json")
        if self.read_from == "sql":
            client = Client.load_from_database(experiment_name=get_experiment_name(),
                                  storage_config = StorageConfig(url=self.backend_options.url))
        if not client:
            raise ValueError(f"read_from entry must be either 'nowhere', 'json', or 'sqlite'")
        return client
    # Attached foamBO config to embed in the JSON state file
    _foambo_config: dict | None = None
    def save(self, client: Client, cards: Iterable[AnalysisCardBase] | None = None):
        filepath = f"artifacts/{get_experiment_name()}_client_state.json"
        if self.save_to == "json":
            client.save_to_json_file(filepath=filepath)
            # Embed foamBO extras inside the JSON state
            extras = {}
            if self._foambo_config is not None:
                extras["foambo_config"] = self._foambo_config
            if extras:
                import json
                with open(filepath, "r") as f:
                    state = json.load(f)
                state.update(extras)
                with open(filepath, "w") as f:
                    json.dump(state, f)
        client._maybe_save_experiment_and_generation_strategy(client._experiment, client._generation_strategy)
        # Cache GP model state_dict for fast reload (skips hyperparameter optimization)
        try:
            gs = client._generation_strategy
            if gs and gs.adapter and hasattr(gs.adapter, 'generator'):
                surr = getattr(gs.adapter.generator, 'surrogate', None)
                if surr and surr.model is not None:
                    import torch
                    model_path = f"artifacts/{get_experiment_name()}_model.pt"
                    torch.save(surr.model.state_dict(), model_path)
        except Exception:
            pass  # non-critical — full refit will happen on next load
        if cards:
            for card in cards:
                client._save_analysis_card_to_db_if_possible(experiment=client._experiment, analysis_card=card)

    @staticmethod
    def load_foambo_config(name: str, artifacts: str = "./artifacts") -> dict | None:
        """Read the embedded foamBO config from a saved JSON state file."""
        import json, os
        filepath = os.path.join(artifacts, f"{name}_client_state.json")
        if not os.path.isfile(filepath):
            return None
        try:
            with open(filepath) as f:
                state = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return None
        return state.get("foambo_config")


class TrialSelector(FoamBOBaseModel):
    """How to pick the source trial for a dependency."""
    strategy: Literal["best", "nearest", "latest", "baseline", "by_index", "matching_group", "custom"] = Field(
        description=(
            "Selection strategy for the source trial.\n"
            "- `best`: completed trial with the best primary objective value\n"
            "- `nearest`: completed trial closest in parameter space (L2, normalized)\n"
            "- `latest`: most recently completed trial\n"
            "- `baseline`: the baseline trial (index 0)\n"
            "- `by_index`: a specific trial by index\n"
            "- `matching_group`: latest trial where all parameters in the specified group have identical values\n"
            "- `custom`: run a command that prints a trial index to stdout"
        ), examples=["best"])
    index: int | None = Field(default=None,
        description="Trial index to use (only for `by_index` strategy)")
    command: str | List[str] | None = Field(default=None,
        description="Command that prints a trial index to stdout (only for `custom` strategy)")
    group: str | None = Field(default=None,
        description="Parameter group name for `matching_group` strategy. "
        "Matches against the `groups` list defined on each parameter.")
    similarity_threshold: float | None = Field(default=None,
        description="Maximum normalized L2 distance for `nearest` strategy. "
        "If the nearest trial exceeds this threshold, the dependency is treated as unresolved (applies fallback). "
        "Range: 0.0 (exact match) to 1.0+ (no restriction). Default: no threshold.")
    fallback: Literal["skip", "error"] = Field(default="skip",
        description="What to do when no suitable source trial exists. 'skip' proceeds without the dependency, 'error' fails the trial")


class TrialAction(FoamBOBaseModel):
    """An action to execute using the resolved source trial."""
    type: Literal["run_command"] = Field(
        description="Action type. `run_command` executes a shell command with `$FOAMBO_SOURCE_TRIAL` and `$FOAMBO_TARGET_TRIAL` substitution",
        examples=["run_command"])
    command: str | List[str] = Field(
        description=(
            "Shell command to execute. Supports substitution variables:\n"
            "- `$FOAMBO_SOURCE_TRIAL`: absolute path to the source trial's case directory\n"
            "- `$FOAMBO_TARGET_TRIAL`: absolute path to the new trial's case directory\n\n"
            "Example: `cp -rT $FOAMBO_SOURCE_TRIAL/0.5 $FOAMBO_TARGET_TRIAL/0`"
        ), examples=["cp -rT $FOAMBO_SOURCE_TRIAL/constant/polyMesh $FOAMBO_TARGET_TRIAL/constant/polyMesh"])
    phase: Literal["immediate", "pre_init", "pre_mesh", "pre_solve", "post_solve"] = Field(
        default="immediate",
        description=(
            "When the action executes in the trial lifecycle:\n"
            "- `immediate`: executed during dependency resolution, before the runner starts (default)\n"
            "- `pre_init`, `pre_mesh`, `pre_solve`, `post_solve`: deferred to hook scripts "
            "that the runner can invoke via environment variables.\n\n"
            "Hook scripts are written to `.foambo_<phase>.sh` in the trial case directory "
            "and exposed as environment variables to the runner subprocess:\n"
            "- `$FOAMBO_PRE_INIT`  — before any case initialisation\n"
            "- `$FOAMBO_PRE_MESH`  — before mesh generation\n"
            "- `$FOAMBO_PRE_SOLVE` — after meshing, before the solver\n"
            "- `$FOAMBO_POST_SOLVE` — after the solver finishes\n\n"
            "Hooks default to no-op (`true`) when no actions target that phase, "
            "so runner scripts work safely from the very first trial (before any "
            "source trial exists).\n\n"
            "Additional environment variables available to the runner:\n"
            "- `$FOAMBO_CASE_PATH` / `$FOAMBO_CASE_NAME` — trial case directory\n"
            "- `$FOAMBO_SOURCE_TRIAL` — resolved source trial path (empty on first trial)\n"
            "- `$FOAMBO_TARGET_TRIAL` — current trial path (same as `$FOAMBO_CASE_PATH`)\n\n"
            "Example Allrun (shell):\n"
            "  $FOAMBO_PRE_INIT\n"
            "  blockMesh\n"
            "  $FOAMBO_PRE_SOLVE\n"
            "  simpleFoam\n"
            "  $FOAMBO_POST_SOLVE\n\n"
            "For non-shell runners (Python, binary), execute the hook script directly:\n"
            "  subprocess.run(os.environ['FOAMBO_PRE_SOLVE'])  # Python\n"
            "  ./.foambo_pre_solve.sh                          # from case directory"
        ))


class TrialDependency(FoamBOBaseModel):
    """A dependency relationship between a source trial and the trial being created.

    Each dependency selects a source trial and runs one or more actions.
    Actions with ``phase: immediate`` execute before the runner starts.
    Actions with a named phase (``pre_init``, ``pre_mesh``, ``pre_solve``,
    ``post_solve``) are deferred to hook scripts that the runner invokes
    via ``$FOAMBO_<PHASE>`` environment variables.

    On the first trial (no completed source yet), dependencies with
    ``fallback: skip`` are silently skipped and all hooks are no-ops.
    """
    name: str = Field(
        description="A label for this dependency (e.g. 'warm_start', 'mesh_inherit'). Recorded in trial metadata for traceability",
        examples=["warm_start"])
    source: TrialSelector = Field(
        description="How to select the source trial")
    actions: List[TrialAction] = Field(
        description=(
            "Actions to execute after source trial is resolved. "
            "`immediate` actions run inline before the runner command. "
            "Phased actions are written to hook scripts invoked by the runner "
            "via `$FOAMBO_PRE_INIT`, `$FOAMBO_PRE_MESH`, `$FOAMBO_PRE_SOLVE`, `$FOAMBO_POST_SOLVE`"
        ))
    enabled: bool = Field(default=True,
        description="Toggle this dependency on/off without removing the configuration")

    @field_validator("actions", mode="before")
    @classmethod
    def parse_actions(cls, v):
        if v and isinstance(v[0], dict | DictConfig):
            return [TrialAction.model_validate(dict(item)) for item in v]
        return v

    @field_validator("source", mode="before")
    @classmethod
    def parse_source(cls, v):
        if isinstance(v, dict | DictConfig):
            return TrialSelector.model_validate(dict(v))
        return v


class FoamBOConfig(FoamBOBaseModel):
    """Top-level configuration for a foamBO optimization run.

    Can be constructed programmatically from Python or loaded from a YAML file
    via ``FoamBOConfig.from_yaml(path)``.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # allow extra keys like 'version', 'visualizer'
    )

    experiment: ExperimentOptions = Field(description="Experiment name, parameters, and search space")
    trial_generation: TrialGenerationOptions = Field(description="How to generate new parameterizations")
    baseline: BaselineOptions = Field(description="Baseline parameter set for comparison")
    optimization: OptimizationOptions = Field(description="Objectives, metrics, constraints, and case runner")
    orchestration_settings: ConfigOrchestratorOptions = Field(description="Timeouts, polling, stopping strategies")
    store: StoreOptions = Field(description="Where to save/load experiment state")
    trial_dependencies: List[TrialDependency] = Field(default=[], description="Trial-to-trial dependency definitions")
    robust_optimization: Any = Field(default=None, description="Robust optimization config (parsed lazily from foambo.robustness)")

    @field_validator("robust_optimization", mode="before")
    @classmethod
    def parse_robust_cfg(cls, v):
        if v is None:
            return None
        if isinstance(v, dict | DictConfig):
            from foambo.robustness import RobustOptimizationConfig
            return RobustOptimizationConfig.model_validate(dict(v) if isinstance(v, DictConfig) else v)
        return v

    @field_validator("trial_dependencies", mode="before")
    @classmethod
    def parse_trial_deps(cls, v):
        if v and isinstance(v[0], dict | DictConfig):
            return [TrialDependency.model_validate(dict(d)) for d in v]
        return v

    # Store raw input before validators transform it (for OmegaConf roundtrip)
    _raw_input: dict | None = None

    @model_validator(mode="wrap")
    @classmethod
    def _capture_raw_input(cls, values, handler):
        raw = dict(values) if isinstance(values, dict) else values
        obj = handler(values)
        object.__setattr__(obj, '_raw_input', raw if isinstance(raw, dict) else None)
        return obj

    @classmethod
    def from_yaml(cls, path: str) -> "FoamBOConfig":
        """Load a FoamBOConfig from a YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_dictconfig(self):
        """Convert to OmegaConf DictConfig for backward compatibility.

        Uses the raw input dict (before validators transform Ax objects)
        so OmegaConf can handle all values as primitives.
        """
        if self._raw_input is not None:
            return DictConfig(self._raw_input)
        return DictConfig(self.model_dump(mode="python"))
