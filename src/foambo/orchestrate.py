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
    return map[transition_type](**cfg)

EARLY_STOPPER_MAP = {
    "none": None,
    "percentile": PercentileEarlyStoppingStrategy,
    "threshold": ThresholdEarlyStoppingStrategy,
    "and": AndEarlyStoppingStrategy,
    "or": OrEarlyStoppingStrategy,
}


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
        "Time-to-live in seconds for individual trials. "
        "Trials exceeding this are marked as failed. Useful for preventing runaway simulations."
    ), examples=[2400])
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

    @field_validator("global_stopping_strategy", mode="before")
    @classmethod
    def parse_global_stopping(cls, v):
        if isinstance(v, dict | DictConfig):
            v = dict(v)
            return ImprovementGlobalStoppingStrategy(
                min_trials=v["min_trials"],
                window_size=v["window_size"],
                improvement_bar=v["improvement_bar"],
                inactive_when_pending_trials=True,
            )
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


class ExperimentOptions(FoamBOBaseModel):
    """Settings for the experiment's search space."""
    name: str = Field(description="Experiment name, used to identify artifacts like reports and state files",
                       examples=["Sample"])
    description: str = Field(description="A short experiment description",
                              examples=["Sample experiment description"])
    parameter_constraints: List[str] = Field(description=(
        "Linear inequalities between optimization parameters, e.g. `'param1 <= param2 + 2*param3'`"
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

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "parameter_constraints": self.parameter_constraints,
        }

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
        "LEGACY_BOTORCH",
        "BOTORCH_MODULAR",
        "EMPIRICAL_BAYES_THOMPSON",
        "EB_ASHR",
        "UNIFORM",
        "ST_MTGP",
        "BO_MIXED",
        "CONTEXT_SACBO",
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
                if "next_node_name" in item:
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
    runner: str | None = Field(default="", description="Shell command to run the trial (e.g. `./Allrun`). Supports `$CASE_NAME`/`$CASE_PATH`. Empty = no execution",
                               examples=["./run_on_cluster.sh $CASE_NAME"])
    log_runner: bool | None = Field(default=False, description="Write runner stdout to `log.runner` in the trial folder instead of discarding it")
    remote_status_query: str | None = Field(default="", description="Command returning SLURM-like job status (required for mode=remote)",
                                             examples=["./state_on_cluster.sh $CASE_NAME"])
    remote_early_stop: str | None = Field(default="", description="Command to kill a remote job (e.g. `scancel --name $CASE_NAME`). Required if early stopping is on",
                                           examples=["scancel --name $CASE_NAME"])

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
            # Embed the foamBO config inside the JSON state
            if self._foambo_config is not None:
                import json
                with open(filepath, "r") as f:
                    state = json.load(f)
                state["foambo_config"] = self._foambo_config
                with open(filepath, "w") as f:
                    json.dump(state, f)
        client._maybe_save_experiment_and_generation_strategy(client._experiment, client._generation_strategy)
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
        with open(filepath) as f:
            state = json.load(f)
        return state.get("foambo_config")


class ExistingTrialsOptions(FoamBOBaseModel):
    """Load pre-existing trial data from a CSV file."""
    file_path: str = Field(description=(
        "Path to a CSV with columns for parameters and metrics. "
        "Metrics can be scalars or `(mean, sem)` pairs. "
        "Rows with the same parameter set are treated as progressions of the same trial."
    ), examples=[""])
    def load_data(self, client: Client):
        """
        Expected format of CSV filename (column order is flexible):
        obj1,obj2,param1,param2,param3
        mean,mean,p11,p12,p13
        "(mean,sem)","(mean,sem)",p21,p22,p23
        run_metadata columns, eg case_path can be specified
        """
        import pandas as pd
        if (not self.file_path) or (self.file_path == "none"):
            return
        n_trials = client._experiment.fetch_data().df.shape[0]
        df = pd.read_csv(self.file_path, header=0)
        metrics = client._experiment.metrics.keys()
        params = client._experiment.parameters.keys()
        missing_metrics = [col for col in metrics if col not in df.columns]
        missing_params = [col for col in params if col not in df.columns]

        if missing_params or missing_metrics:
            raise ValueError(
                f"Missing columns while loaded existing trial data from {self.file_path}"
                f" — Parameters: {missing_params}, Metrics: {missing_metrics}\n"
            )

        grouped_by_params = df.groupby(list(params))
        new_trial_idx = n_trials-1
        for _, group_df in grouped_by_params:
            new_trial_idx += 1
            progressions = group_df.shape[0]
            idx_in_grp = -1
            for idx, row in group_df.iterrows():
                idx_in_grp += 1
                row_params = row[params].to_dict()
                row_metrics = {k: parse_outcome_for_metric(v, k) for k, v in row[metrics].to_dict().items()}
                if idx_in_grp == 0:
                    trial_index = client.attach_trial(parameters=row_params, arm_name=f"{new_trial_idx}_supplied")
                if idx_in_grp == progressions-1:
                    client.complete_trial(trial_index=new_trial_idx,
                                          raw_data=row_metrics,
                                          progression=idx_in_grp)
                    continue
                client.attach_data(trial_index=new_trial_idx,
                                      raw_data=row_metrics,
                                      progression=idx_in_grp)
        if not client._experiment.fetch_data().df.empty:
            log.info(f"Loaded existing trial data:\n{client._experiment.fetch_data().df.to_string(index=False)}")


class TrialSelector(FoamBOBaseModel):
    """How to pick the source trial for a dependency."""
    strategy: Literal["best", "nearest", "latest", "baseline", "by_index", "custom"] = Field(
        description=(
            "Selection strategy for the source trial.\n"
            "- `best`: completed trial with the best primary objective value\n"
            "- `nearest`: completed trial closest in parameter space (L2, normalized)\n"
            "- `latest`: most recently completed trial\n"
            "- `baseline`: the baseline trial (index 0)\n"
            "- `by_index`: a specific trial by index\n"
            "- `custom`: run a command that prints a trial index to stdout"
        ), examples=["best"])
    index: int | None = Field(default=None,
        description="Trial index to use (only for `by_index` strategy)")
    command: str | List[str] | None = Field(default=None,
        description="Command that prints a trial index to stdout (only for `custom` strategy)")
    fallback: Literal["skip", "error"] = Field(default="skip",
        description="What to do when no suitable source trial exists. 'skip' proceeds without the dependency, 'error' fails the trial")


class TrialAction(FoamBOBaseModel):
    """An action to execute using the resolved source trial."""
    type: Literal["run_command"] = Field(
        description="Action type. `run_command` executes a shell command with `$SOURCE_TRIAL` and `$TARGET_TRIAL` substitution",
        examples=["run_command"])
    command: str | List[str] = Field(
        description=(
            "Shell command to execute. Supports substitution variables:\n"
            "- `$SOURCE_TRIAL`: absolute path to the source trial's case directory\n"
            "- `$TARGET_TRIAL`: absolute path to the new trial's case directory\n\n"
            "Example: `cp -rT $SOURCE_TRIAL/0.5 $TARGET_TRIAL/0`"
        ), examples=["cp -rT $SOURCE_TRIAL/constant/polyMesh $TARGET_TRIAL/constant/polyMesh"])


class TrialDependency(FoamBOBaseModel):
    """A dependency relationship between a source trial and the trial being created."""
    name: str = Field(
        description="A label for this dependency (e.g. 'warm_start', 'mesh_inherit'). Recorded in trial metadata for traceability",
        examples=["warm_start"])
    source: TrialSelector = Field(
        description="How to select the source trial")
    actions: List[TrialAction] = Field(
        description="Actions to execute after source trial is resolved. Run in order before the trial's runner command")
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
    existing_trials: ExistingTrialsOptions = Field(description="Pre-existing trial data to load from CSV")
    baseline: BaselineOptions = Field(description="Baseline parameter set for comparison")
    optimization: OptimizationOptions = Field(description="Objectives, metrics, constraints, and case runner")
    orchestration_settings: ConfigOrchestratorOptions = Field(description="Timeouts, polling, stopping strategies")
    store: StoreOptions = Field(description="Where to save/load experiment state")
    trial_dependencies: List[TrialDependency] = Field(default=[], description="Trial-to-trial dependency definitions")

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
