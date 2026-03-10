from ax.api.client import AnalysisCardBase, PercentileEarlyStoppingStrategy, Client
from ax.service.utils.orchestrator_options import OrchestratorOptions
from ax.api.types import TParameterization
from ax.early_stopping.strategies import AndEarlyStoppingStrategy, OrEarlyStoppingStrategy, ThresholdEarlyStoppingStrategy
from ax.generation_strategy.generation_node import MaxGenerationParallelism
from ax.storage.json_store.decoder import AuxiliaryExperimentCheck
from ax.storage.json_store.registry import CenterGenerationNode
from foambo.metrics import FoamJobRunner, LocalJobMetric
from .common import *
from .common import FoamBOBaseModel
from pydantic import field_validator, model_validator
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
    MaxTrials, MinTrials, AutoTransitionAfterGen,
    IsSingleObjective, MinimumPreferenceOccurances, MinimumTrialsInStatus
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
    """
    Configuration hub for trial orchestration that handles stopping strategy
    (early/global)
    """
    max_trials: int
    parallelism: int
    global_stopping_strategy: Any  # BaseGlobalStoppingStrategy
    early_stopping_strategy: Any
    tolerated_trial_failure_rate: float = 0.5
    min_failed_trials_for_failure_rate_check: int = 5
    initial_seconds_between_polls: int = 1
    min_seconds_before_poll: float = 1.0
    seconds_between_polls_backoff_factor: float = 1.5
    timeout_hours: int | None = None
    ttl_seconds_for_trials: int | None = None
    fatal_error_on_metric_trouble: bool = False
    immutable_search_space: bool = True

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
    """
    Configuration hub for optimization experiment
    """
    name: str
    description: str
    parameter_constraints: List[str]
    parameters: List[Any]  # List[ChoiceParameterConfig | RangeParameterConfig]

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


class ModelSpecConfig(FoamBOBaseModel):
    """Configuration for GeneratorSpec"""
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
    ]
    model_kwargs: Dict | None = None

    def to_generator_spec(self) -> GeneratorSpec:
        if not hasattr(Generators, self.generator_enum):
            from pprint import pformat
            raise ValueError(f"{self.generator_enum} is not a supported generator; supported specs:\n"
                             f"{pformat([name for name, value in vars(Generators).items() if not name.startswith('_')])}")
        model = Generators[self.generator_enum]
        return GeneratorSpec(
            generator_enum=model,
            model_kwargs=self.model_kwargs if self.model_kwargs else {},
        )

TRANSITION_MAP = {
    "max_trials": MaxTrials,
    "min_trials": MinTrials,
    "auto_transition_after_gen": AutoTransitionAfterGen,
    "is_single_objective": IsSingleObjective,
    "max_generation_parallelism": MaxGenerationParallelism,
    "minimum_preference_occurances": MinimumPreferenceOccurances,
    "auxiliary_experiment_check": AuxiliaryExperimentCheck,
    "minimum_trials_in_status": MinimumTrialsInStatus,
}


class GenerationNodeConfig(FoamBOBaseModel):
    """Configuration for a generation node"""
    node_name: str
    generator_specs: List[ModelSpecConfig]
    transition_criteria: List[Any] = []

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
            node_name=self.node_name,
            generator_specs=spec_list,
            transition_criteria=self.transition_criteria if self.transition_criteria else None,
        )


class CenterGenerationNodeConfig(FoamBOBaseModel):
    """Configuration for center generation node"""
    next_node: str

    def to_generation_node(self):
        return CenterGenerationNode(next_node_name=self.next_node)


class ManualGenerationNode(GenerationNode):
    parameters: TParameterization
    def __init__(self, node_name: str, parameters: TParameterization):
        super().__init__(node_name=node_name, generator_specs=[])
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
            arms=[Arm(name=self.node_name, parameters=self.parameters)],
            optimization_config=experiment.optimization_config,
            search_space=experiment.search_space,

        )


class TrialGenerationOptions(FoamBOBaseModel):
    """
    Configuration hub for generation strategy
    """
    method:  Literal["fast", "random_search", "custom"]
    generation_nodes: List[Any] | None = None
    initialization_budget: int | None = 5
    initialization_random_seed: int | None = None
    initialize_with_center: bool = True
    use_existing_trials_for_initialization: bool = True
    min_observed_initialization_trials: int | None = None
    allow_exceeding_initialization_budget: bool = False

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
        gs = GenerationStrategy(name=f"+".join([node.node_name for node in self.generation_nodes]), nodes=self.generation_nodes)
        client.set_generation_strategy(gs)


class VariableSubstOptions(FoamBOBaseModel):
    file: str
    parameter_scopes: Dict[str, str]


class FileSubstOptions(FoamBOBaseModel):
    parameter: str
    file_path: str


class FoamJobRunnerOptions(FoamBOBaseModel):
    mode: Literal["local", "remote"]
    template_case: str
    trial_destination: str
    artifacts_folder: str
    variable_substitution: List[VariableSubstOptions]
    file_substitution: List[FileSubstOptions]
    runner: str | None = ""
    log_runner: bool | None = False
    remote_status_query: str | None = ""
    remote_early_stop: str | None = ""

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
    parameters: TParameterization | None


class OptimizationOptions(FoamBOBaseModel):
    """
    Configuration hub for optimization
    """
    objective: str
    outcome_constraints: List[str]
    metrics: List[LocalJobMetric]
    case_runner: FoamJobRunnerOptions

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
    pass


class SQLBackendOptions(FoamBOBaseModel):
    url: str


class StoreOptions(FoamBOBaseModel):
    read_from: Literal["json", "sql", "nowhere"]
    save_to: Literal["json", "sql"]
    backend_options: JSONBackendOptions | SQLBackendOptions

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
    def save(self, client: Client, cards: Iterable[AnalysisCardBase] | None = None):
        if self.save_to == "json":
            client.save_to_json_file(filepath=f"artifacts/{get_experiment_name()}_client_state.json")
        client._maybe_save_experiment_and_generation_strategy(client._experiment, client._generation_strategy)
        if cards:
            for card in cards:
                client._save_analysis_card_to_db_if_possible(experiment=client._experiment, analysis_card=card)


class ExistingTrialsOptions(FoamBOBaseModel):
    file_path: str
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
