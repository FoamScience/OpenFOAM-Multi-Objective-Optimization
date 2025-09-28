from .common import VERSION
from typing import Dict, Any, get_args
from dataclasses import fields
import yaml

CURRENT_AX_VERSION="1.1.2"
def get_ax_path(name, file_path, lines=None):
    if lines:
        return f"[{name}](https://github.com/facebook/Ax/blob/{CURRENT_AX_VERSION}/{file_path}#L{lines[0]}-L{lines[1]})"
    return f"[{name}](https://github.com/facebook/Ax/blob/{CURRENT_AX_VERSION}/{file_path})"

class RangeTag:
    yaml_tag = '!range'
    def __init__(self, start, stop, step=1):
        self.start = start
        self.stop = stop
        self.step = step
def range_representer(dumper, data: RangeTag):
    seq = [data.start, data.stop] if data.step == 1 else [data.start, data.stop, data.step]
    return dumper.represent_sequence('!range', seq, flow_style=True)
yaml.add_representer(RangeTag, range_representer)
def yaml_range(loader, node):
    values = loader.construct_sequence(node)
    return list(range(*values)) if len(values) > 1 else []
yaml.add_constructor("!range", yaml_range)
yaml.SafeLoader.add_constructor("!range", yaml_range)

def get_default_config() -> Dict[str, Any]:
    """
    Return a DictConfig object with the default configuration.
    """
    default = {
        "version": str(VERSION),
        "experiment": {
            "name": "Sample",
            "description": "Sample experiment description",
            "parameters": [
                {
                    "name": "x",
                    "bounds": [ 0.0, 1.0 ],
                    "step_size": None,
                    "scaling": None,
                    "parameter_type": "float"
                },
                {
                    "name": "y",
                    "values": [ "zero", "one" ],
                    "is_ordered": None,
                    "dependent_parameters": None,
                    "parameter_type": "str"
                },
            ],
            "parameter_constraints": []
        },
        "trial_generation": {
            "method": "custom",
            "generation_nodes": [
                {
                    "next_node_name": "init",
                },
                {
                    "node_name": "init",
                    "generator_specs": [
                        {
                            "generator_enum": "SOBOL",
                            "model_kwargs": {
                                "seed": 12345
                            }
                        }
                    ],
                    "transition_criteria": [
                        {
                            "type": "max_trials",
                            "threshold": 7,
                            "transition_to": "BOM",
                            "use_all_trials_in_exp": True
                        }
                    ]
                },
                {
                    "node_name": "BOM",
                    "generator_specs": [
                        {
                            "generator_enum": "BOTORCH_MODULAR",
                        }
                    ],
                    "transition_criteria": []
                },
            ]
        },
        "existing_trials": {
            "file_path": ""
        },
        "baseline": {
            "parameters": {
                "x": 0.8,
                "y": "zero"
            }
        },
        "optimization": {
            "metrics": [
                {
                    "name": "metric",
                    "progress": [ "echo", "$STEP" ],
                    "command": [ "echo", "0" ],
                },
            ],
            "objective": "-metric",
            "outcome_constraints": ["metric >= 0.9*baseline"],
            "case_runner": {
                "template_case": "./case",
                "mode": "remote", # or local
                "runner": "./run_on_cluster.sh $CASE_NAME",
                "log_runner": False,
                "remote_status_query": "./state_on_cluster.sh $CASE_NAME",
                "remote_early_stop": "scancel --name $CASE_NAME",
                "trial_destination": "./trials",
                "artifacts_folder": "./artifacts",
                "file_substitution": [
                    {
                        "parameter": "y",
                        "file_path": "/constant/y",
                    },
                ],
                "variable_substitution": [
                    {
                        "file": "/0orig/field",
                        "parameter_scopes": {
                            "x": "someDict.x"
                        }
                    }
                ],
            }
        },
        "orchestration_settings": {
            "max_trials": 20,
            "parallelism": 3,
            "initial_seconds_between_polls": 60,
            "seconds_between_polls_backoff_factor": 1.5,
            "timeout_hours": 48,
            "ttl_seconds_for_trials": 2400,
            "global_stopping_strategy": {
                "min_trials": 10,
                "window_size": 5,
                "improvement_bar": 0.1
            },
            "early_stopping_strategy": {
                "type": "percentile",
                "metric_names": [
                    "metric"
                ],
                "percentile_threshold": 25,
                "min_progression": 5,
                "trial_indices_to_ignore": RangeTag(0, 10, 1)
            }
        },
        "store": {
            "save_to": "json",
            "read_from": "json",
            "backend_options": {
                "url": None
            }
        }
    }
    return default

def get_config_docs() -> Dict[str, Any]:
    from .orchestrate import TrialGenerationOptions, ModelSpecConfig, StoreOptions
    docs = {
        "version": f"The foamBO version to run this configuration with (v{str(VERSION)})",
        "experiment": "[Section] Settings for the experiment's search space",
        "experiment.name": "A descriptive experiment name, used to identify artifacts like analysis reports and experiment state files.",
        "experiment.description": "A short experiment description",
        "experiment.parameters": f"""
            List of either RangeParameter or ChoiceParameter configuration,
            accepting any arguments to these class constructors:
            - {get_ax_path("ChoiceParameter", "ax/core/parameter.py", [574, 590])}
            - {get_ax_path("RangeParameter", "ax/core/parameter.py", [261, 269])}

            Alternatively, set a random property name:
            ```yaml
            experiment:
                parameters:
                - arbitrary: true
            ```
            and foamBO will complain with a list of needed properties (The BANANA trick).
            """,
        "experiment.parameter_constraints": """
            A list of strings describing inequalities between linear combinations
            of optimization parameters, eg. `'param1 <= param2 + 2*param3'`
            """,
        "trial_generation": "[Section] Controls over trial generation methods",
        "trial_generation.method": f"""
            One of `{get_args(TrialGenerationOptions.__annotations__["method"])}`
            to define how to generate new parameterizations.

            For finer control over trial generations, we provide a "custom" strategy, where the user specifies the generation
            nodes, and when to transition between them:
            ```yaml
            trial_generation:
                method: custom
                generation_nodes:
                    - next_node_name: init                # this node will pick the center of the search space
                    -   node_name: init                     # reproduceable random initial points
                        generator_specs:
                            -   generator_enum: SOBOL
                                model_kwargs:
                                    seed: 12345
                        transition_criteria:                # transition to next node once experiment has 5 trials
                            -   type: max_trials
                                threshold: 5
                                transition_to: BOM
                                use_all_trials_in_exp: true
                    -   node_name: BOM                      # Let BOTORCH decide on best next parameterization
                        generator_specs:
                            -   generator_enum: BOTORCH_MODULAR
                                transition_criteria: []
            ```
            Each node is either a:
            - {get_ax_path("GenerationNone", "ax/generation_strategy/generation_node.py", [85, 119])}
              with support for some GeneratorSpecs arguments:
              - `{fields(ModelSpecConfig)[0].name}`: `{fields(ModelSpecConfig)[0].type}` 
              - `{fields(ModelSpecConfig)[1].name}`: arguments of the respective generator from
                {get_ax_path("generators", "ax/generators")}
            - {get_ax_path("CenterGenerationNode", "ax/generation_strategy/center_generation_node.py", [26, 27])}

            As an example the seed property of SOBOL generator and related options can be found at
            {get_ax_path("sobol.py", "ax/generators/random/sobol.py", [35, 40])}

            Transition options are listed in the
            {get_ax_path("transition_criterion.py file", "ax/generation_strategy/transition_criterion.py", [117, 844])}
            and a `type` entry is required to identify them
            """,
        "existing_trials": "[Section] Load pre-existing trial data from a CSV file",
        "existing_trials.file_path": f"""
            Path to the CSV file to load the trial data from.

            Expected format of the CSV file: `param1, param2, metric1, metric2, case_path`
            - `metrics` can be just scalar values or `(mean, SEM)` pairs
            - `case_path` column can be empty, as the trial cases don't actrually have to be on disk
            - Rows that have the same parameter sets will be treated as progressions of the same trial

            **Note: This is an experimental feature**
            """,
        "baseline": "[Section] A set of parameter values to consider as a baseline for the optimization analysis",
        "baseline.parameters": f"""
            The parameter values to assign as optimization baseline.

            Setting this entry to `null` skips defining a baseline:
            ```yaml
            baseline:
                parameters: null
            ```

            If a baseline is to be defined, a single trial is always generated and ran before starting the
            generation strategy, and it counts towards the maximum number of trials:
            ```yaml
            baseline:
                parameters:
                    param1: 1.0
                    param2: "value2"
            ```
            """,

        "optimization": "[Section] Controls for the bayesian optimization algorithm, including objectives, constraints, and case dispatch", 
        "optiomization.metrics": f"""
            A list of metrics that are either used for:
            - Computing optimization objectives
            - Tracking some non-objective quantity, for example, to use in early-stopping strategies

            A metric has the following properties:
            ```yaml
            name: metric_name,
            command: "echo 1"     # A shell command to evaluate the metric at trial completion
            progress: []          # An optional shell command to evaluate the metric while the trial is still running
            lower_is_better: True # Required only if the metric is not an objective and early-stopping is on
            ```
            
            The `command` entry can be a list of strings or a string, specifying a shell command to evaluate the metric:
            - As a blocking operation; it must return a value, even if the returned value is NaN 
            - It has to write either a `<scalar>` or a `<mean>, <sem>` to its standard output (NaN is supported as a "scalar")
            - It always runs locally with the trial's folder as its CWD
            - Any appearances of `$CASE_NAME` or `$CASE_PATH` in the command will be replaced by the respective corresponding
              values
            
            The `progress` entry is similar to `command`, but also:
            - Runs every time the BO algorithm polls for trial completion, which is affected by the `orchestration_settings`
              section
            - Supports `$STEP` replacing with the progression step

            **Note: Since these commands are blocking-ops, and may run frequently, it's recommended to move any heavy-lifting
            to the case runner, and only consult logs with the metric commands**
            """,
        "optimization.objective": f"""
            A description of what to optimize.

            Single-objective minimization of a metric:
            ```yaml
            optimization:
                objective: "-metric"    # '-' for minimize, else maximizes
            ```

            Multi-objective optimization:
            ```yaml
            optimization:
                objective: "-metric1, metric2"
            ```
            """,
        "optimization.outcome_constraints": f"""
            A list of strings describing linear objective constraints

            The constraints can be relative to the baseline trial if provided:
            ```yaml
            optimization:
                outcome_constraints:
                    - "metric1 >= 50*baseline"    # 50% of the baseline metric1 value 
                    - "metric2 <= 10"             # An absolute bound value
            ```

            As a general rule, provide an upper bound for metrics that are minimized,
            and a lower bound otherwise. This is especially important in multi-objective settings
            """,
        "optimization.case_runner": f"""
            Configuration for trial case dispatch and monitoring.

            ```yaml
            optimization:
                case_runner:
                    template_case: "./case"          # The OpenFOAM case to use as a base for trials
                    trial_destination: /tmp/trials   # Where to put the new trials
                    artifacts_folder: /tmp/artifacts # Where to put reports and experiment state files
                    mode: local                      # Case run mode: local or remote
                    runner: "/tmp/run_case.sh"       # Everythign that applies to metric commands applies here too
                    log_runner: False                # Whether to write runner logs to log.runner file in trial folder
                                                     # Turn this on if your runner script generates a lot of text at STDOUT
                    # In case mode=remote, these two entries must be set:
                    remote_status_query: /tmp/cluster_job_state.sh  # Has to return SLURM-like job status
                    remote_early_stop: "scancel --name $CASE_NAME"  # This is needed only if early-stopping is on
                    # The next two entries define how to substitute parameter values in case files
                    file_substitution: []          # A list of files to replace with ready-made ones. More on this bellow 
                    variable_substitution: []      # A list of parameter scopes to substitute with values. More on this bellow
            ```

            File substitutions are designed to deal with choice parameters, especially the ones which
            have string values:
            ```yaml
            file_substitution:
                # Assume param1 has two possible values: GAMG, PCG
                # This will replace the trial's fvSolution file with either
                # system/fvSolution.GAMG or system/fvSolution.PCG
                # depending on the chosen value for the specific trial
                - parameter: param1
                  file_path: "/system/fvSolution"
            ```

            Variable substitution uses the `foamlib` library to directly assign the chosen values
            to the designated scope within the case file.

            These are organized by file to avoid excessive repetition:
            ```yaml
            variable_substitution:
                - file: /0orig/field
                  parameter_scopes:
                      x: subDict.x
                      y: y
                ... more files untill all parameters are substituted
            ```
            """,
        "orchestration_settings": "[Section] Controls for timeouts, poll times, early-stopping and convergence criteria",
        "orchestration_settings.max_trials": "Maximal number of trials to run, baseline included",
        "orchestration_settings.parallelism": "Maximal number of trials to be running at the same time",
        "orchestration_settings.initial_seconds_between_polls": "How many seconds to wait to poll trials initially",
        "orchestration_settings.seconds_between_polls_backoff_factor": "Enlarge/shrink initial waiting time between polls gradually",
        "orchestration_settings.timeout_hours": "Timeout in hours for the whole experiment; baseline excluded",
        "orchestration_settings.ttl_seconds_for_trials": "Timeout in seconds for single trials",
        "orchestration_settings.global_stopping_strategy": f"""
            Defines when to stop the optimization based on the best improvement so far.

            Stops coming up with new trials once the improvement in best observed objective over `n`
            consecutive trials becomes less than `bar * IQR` where
            - `IQR` is the interquartile range of observed objectives,
            - `bar` is a user-supplied setting as shown bellow
            ```yaml
            orchestration_settings:
              global_stopping_strategy:
                improvement_bar: 0.1
                min_trials: 30
                window_size: 10
            ```

            See {get_ax_path("ImprovementGlobalStoppingStrategy", "ax/global_stopping/strategies/improvement.py", [52, 55])}
            for possible configuration entries
            """,
        "orchestration_settings.early_stopping_strategy": f"""
            Stops trials mid-way if they prove to be not interesting enough for the objectives.

            The main goal is to save on resources, and to penalize harmful trials (eg. ones with inaccurate results).

            Ax provides two main strategies and logical operations ('and' and 'or') on them:
            - {get_ax_path("Percentile-based", "ax/early_stopping/strategies/percentile.py", [31, 37])}
              which looks at all feasible trials once they reach their `min_progression` step,
              compare trial metrics, and if one performs worse than the percentile threshold set,
              the trial is early-stopped.
            - {get_ax_path("Threshold-based", "ax/early_stopping/strategies/threshold.py", [29, 35])}
              which checks if the trial's metric beyond `min_progression` step is worse than a threshold.

            **None: "worse" in this case is defined by the objective, or `lower_is_better` entry if the chosen metric is not an objective**

            As a convenience option, `trial_indices_to_ignore` can be set through the YAML configuration with the
            custom range constructor:
            ```yaml
            orchestration_settings:
                early_stopping_strategy:
                    type: or
                    left:
                        type: percentile
                        metric_names: [ "metric1" ]
                        min_progression: 2
                        percentile_threshold: 25
                        trial_indices_to_ignore: !range [0, 30]
                    right:
                        type: threshold
                        metric_names: [ "metric2" ]
                        min_progression: 2
                        metric_threshold: 20
                        trial_indices_to_ignore: !range [0, 30]
            ```
            Note the `type` entry required to identify which strategy to pick.
            """,
        "store": "[Section] Controls where to store experiment state, and where to read it from",
        "store.save_to": f"Choose the storage backend: {fields(StoreOptions)[0].type}",
        "store.read_from": f"Choose the storage backend to read/load experiment state from: {fields(StoreOptions)[1].type}",
        "store.backend_options": f"""
            A dictionary used to supply backend-specific (SQL databases, or JSON files) settings:
            ```yaml
            store:
              read_from: nowhere
              save_to: sql
              backend_options:
                url: <SQL_database_URL>
            ```

            Note that the SQL store is currently not well tested.
            """,
        "loading_client_state": {
            "category": "Python snippet, not a configuration!",
            "content": f"""
            You can load an [Ax](https://ax.dev) client for your experiment with:
            ```python
            # This should run in the same folder holding the YAML configuration
            from foambo.common import *
            from foambo.orchestrate import StoreOptions
            set_experiment_name("MyExperiment") # experiment.name from your YAML config
            store = StoreOptions(save_to="nowhere", read_from="json", backend_options={{}})
            client = store.load()
            ```
                """,
        },
    }
    return docs
