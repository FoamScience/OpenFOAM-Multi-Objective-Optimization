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
        },
        "visualizer": {
            "sensitivity_callback": None
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
        "orchestration_settings.tolerated_trial_failure_rate": """
            Maximum fraction of failed trials before the experiment is terminated (default: 0.5).
            
            If more than this fraction of trials fail, the orchestrator will stop generating new trials.
            Set to 1.0 to never stop due to failures, or 0.0 to stop at the first failure.
            """,
        "orchestration_settings.min_failed_trials_for_failure_rate_check": """
            Minimum number of failed trials before checking the failure rate (default: 5).
            
            The failure rate check only triggers after at least this many trials have failed.
            This prevents early termination due to a few initial failures.
            """,
        "orchestration_settings.initial_seconds_between_polls": """
            Initial wait time in seconds before polling trial status (default: 1).
            
            This is the starting interval for checking if running trials have completed.
            The actual wait time increases with the backoff factor.
            """,
        "orchestration_settings.min_seconds_before_poll": """
            Minimum wait time in seconds before any poll (default: 1.0).
            
            Even with backoff, the orchestrator will wait at least this long between polls.
            Prevents excessive polling of trial status.
            """,
        "orchestration_settings.seconds_between_polls_backoff_factor": """
            Multiplier to increase wait time between polls (default: 1.5).
            
            After each poll, the wait time is multiplied by this factor (exponential backoff).
            - Values > 1.0: Wait time increases (recommended for long-running trials)
            - Value = 1.0: Constant wait time
            - Values < 1.0: Wait time decreases (not recommended)
            
            Example: With initial=60s and factor=1.5, polls happen at 60s, 90s, 135s, 202.5s, ...
            """,
        "orchestration_settings.timeout_hours": """
            Maximum runtime in hours for the entire experiment, excluding baseline trial (default: None).
            
            If set, the experiment will terminate after this many hours, even if max_trials
            has not been reached. Set to None for no timeout.
            """,
        "orchestration_settings.ttl_seconds_for_trials": """
            Time-to-live in seconds for individual trials (default: None).
            
            If a trial runs longer than this, it will be marked as failed and stopped.
            Set to None for no per-trial timeout. Useful for preventing runaway simulations.
            """,
        "orchestration_settings.fatal_error_on_metric_trouble": """
            Whether to terminate the experiment if metric evaluation fails (default: False).
            
            - True: Stop the entire experiment if any metric command fails or returns invalid data
            - False: Mark the trial as failed and continue with other trials
            
            Set to True for strict validation, False for robustness to occasional metric failures.
            """,
        "orchestration_settings.immutable_search_space": """
            Whether the search space is fixed after experiment creation (default: True).
            
            - True: Parameters and constraints cannot be modified after the first trial
            - False: Allows dynamic modification of the search space (advanced use only)
            
            Most users should keep this as True for reproducibility and consistency.
            """,
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
        "visualizer": "[Section] Settings for the web-based visualizer UI",
        "visualizer.sensitivity_callback": """
            Optional Python callback function to visualize parameter changes in sensitivity analysis.

            When configured, the visualizer will automatically generate a custom plot when the user
            clicks "Predict Metrics" in the sensitivity analysis section. This allows you to provide
            domain-specific visualizations of how parameter changes affect your system.

            **Configuration:**
            ```yaml
            visualizer:
              sensitivity_callback: "mymodule.myvisualizer.plot_function"
            ```

            **Callback Signature:**
            The callback function must accept a dictionary of parameters and return a base64-encoded image:
            ```python
            import matplotlib.pyplot as plt
            import io
            import base64

            def plot_function(parameters: dict) -> str:
                '''
                Visualize the system based on parameter values.

                Args:
                    parameters: Dictionary mapping parameter names to their values
                                e.g., {"x": 0.5, "y": "choice1", "z": 1.23}

                Returns:
                    A base64-encoded PNG image string
                '''
                # Extract parameters
                x = parameters.get('x', 0.0)
                y = parameters.get('y', 'default')

                # Generate visualization data
                # ... your custom logic here ...

                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot([...], [...])
                ax.set_title("Custom Visualization")
                ax.set_xlabel("X axis")
                ax.set_ylabel("Y axis")
                plt.tight_layout()

                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                return image_base64
            ```

            **Error Handling:**
            - If the callback raises an exception, the visualizer will display an error message
              but sensitivity analysis will still work
            - If the module cannot be imported, a warning is shown to the user
            - If the callback returns a non-string object, an error is displayed

            **Requirements:**
            - The callback module must be importable from the experiment directory
            - The function must return a base64-encoded image string
            - Matplotlib (or any plotting library that can generate base64 images) must be available

            **When to Use:**
            Use this feature when you want to help users understand the physical or geometric
            implications of parameter changes beyond just predicted metric values. Examples:
            - Visualizing geometry changes (airfoil shapes, duct profiles, etc.)
            - Showing control system responses (PID tuning, filter responses)
            - Displaying material property curves (stress-strain, phase diagrams)
            - Plotting field distributions (temperature, velocity, concentration)

            Set to `null` or omit to disable custom visualization (default behavior).
            """,
    }
    python_snippets = {
        "loading_client_state": {
            "category": "Python snippet",
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
    visualizer_api = {
        "visualizer_ui": {
            "category": "Visualizer API",
            "content": """
            Launch an interactive web UI to explore optimization results and run manual trials.
            
            ```python
            from foambo.visualize import visualizer_ui
            from foambo.common import load_config
            
            cfg = load_config("config.yaml")
            visualizer_ui(
                cfg,
                host="127.0.0.1",      # Server host address
                port=8099,             # Server port (auto-increments if in use)
                open_browser=True      # Automatically open browser
            )
            ```
            
            **Features:**
            - **Experiment Overview**: View trial history, parameters, and objectives
            - **Model Fitting**: Force model fitting to existing trials
            - **Pareto Frontier**: Pick optimal points from multi-objective optimization
            - **Sensitivity Analysis**: Adjust parameters and predict outcomes in real-time
            - **Manual Trials**: Run trials with custom parameter values from the UI
            - **Trial Visualization**: View OpenFOAM results with PyVista (if available)
            - **Diagnostics & Insights**: Generate automated analysis reports with Plotly
            
            **Port Handling:**
            If the specified port is in use, the visualizer automatically tries the next port
            (up to 3 attempts). For example, if port 8099 is busy, it tries 8100, then 8101.
            
            **Requirements:**
            - The experiment must already exist (run optimization first or load from storage)
            - For trial visualization: PyVista must be installed
            - For insights: Latest Ax version with `InsightsAnalysis` support
            
            **UI Tabs:**
            1. **Optimization**: Main workflow for Pareto analysis and predictions
            2. **Trial Visualization**: 3D visualization of OpenFOAM trial results
            3. **Diagnostics & Insights**: Automated experiment analysis and diagnostics
            
            **Manual Trial Workflow:**
            1. Click "Pick most interesting point" to select a Pareto-optimal configuration
            2. Adjust parameters in the sensitivity analysis section
            3. Click "Predict Metrics" to see expected outcomes
            4. Click "Run trial with selected params" to execute the trial
            5. Monitor trial status in real-time with automatic polling
            
            **Access the UI:**
            Once started, navigate to `http://127.0.0.1:8099` (or the displayed port) in your browser.
            """,
        },
        "visualizer_features": {
            "category": "Visualizer API",
            "content": """
            **Advanced Visualizer Features:**
            
            **1. Parameter Deduplication:**
            The visualizer automatically detects if you're trying to run a trial with parameters
            that match an existing trial. Instead of running a duplicate, it returns the existing
            trial index, saving computational resources.
            
            **2. Real-time Trial Status:**
            When running manual trials, the UI polls the server every 2 seconds to show:
            - "Queueing trial..." (yellow, spinner)
            - "Trial X is running..." (yellow, spinner)
            - "Trial X completed successfully" (green)
            - "Trial X failed: <error>" (red with full error message)
            
            **3. Delta Visualization:**
            When predicting metrics, the UI shows the change from the selected Pareto point:
            - Green ↓: Improvement (better than Pareto point)
            - Orange ↑: Worsening (worse than Pareto point)
            - Gray →: Neutral (< 0.05% change)
            
            **4. OpenFOAM Visualization Settings:**
            For trial visualization, you can configure:
            - Time step selection (latest or specific time)
            - Field variable to display
            - Decompose polyhedra (for complex cells)
            - Cell-to-point data conversion
            - Skip zero time directory
            
            **5. Plotly Dark Theme:**
            All diagnostic plots use Plotly's dark theme to match the UI aesthetic.
            
            **6. Error Handling:**
            - Full tracebacks logged to server console
            - User-friendly error messages in UI
            - Graceful degradation when features are unavailable
            
            **7. Full-Width Layout:**
            The UI spans the entire browser width for better visualization of:
            - Wide parameter tables
            - Large Plotly charts
            - OpenFOAM mesh visualizations
            """,
        },
        "visualizer_endpoints.api.experiment": {
            "category": "Visualizer API",
            "content": """
            `GET /api/experiment` - Get experiment metadata, parameters, and objectives.
            
            **Returns:**
            ```json
            {
                "name": "experiment_name",
                "trial_count": 42,
                "parameters": [
                    {"name": "x", "type": "range", "bounds": [0.0, 1.0]},
                    {"name": "y", "type": "choice", "values": ["a", "b"]}
                ],
                "objectives": [
                    {"name": "drag", "direction": "Minimize"},
                    {"name": "lift", "direction": "Maximize"}
                ]
            }
            ```
            
            **Example:**
            ```bash
            curl http://localhost:8099/api/experiment
            ```
            """,
        },
        "visualizer_endpoints.api.trials": {
            "category": "Visualizer API",
            "content": """
            `GET /api/trials` - List all trials with their status.
            
            **Returns:**
            ```json
            {
                "trials": [
                    {"index": 0, "status": "COMPLETED", "parameters": {...}},
                    {"index": 1, "status": "RUNNING", "parameters": {...}}
                ]
            }
            ```
            """,
        },
        "visualizer_endpoints.api.trial_list": {
            "category": "Visualizer API",
            "content": """
            `GET /api/trial_list` - Get detailed trial list with case paths.
            
            **Returns:**
            ```json
            {
                "trials": [
                    {
                        "index": 0,
                        "status": "COMPLETED",
                        "has_case_path": true,
                        "case_path": "/path/to/trial_0"
                    }
                ]
            }
            ```
            
            Used by the Trial Visualization tab to populate the trial selector.
            """,
        },
        "visualizer_endpoints.api.pareto": {
            "category": "Visualizer API",
            "content": """
            `GET /api/pareto?objective=<name>` - Get Pareto-optimal point for an objective.
            
            **Query Parameters:**
            - `objective` (required): Name of the objective metric to optimize
            
            **Returns:**
            ```json
            {
                "objective": "drag",
                "minimize": true,
                "parameters": {"x": 0.42, "y": "value1"}
            }
            ```
            
            Selects the point with the best value for the specified objective, considering:
            - Objective direction (minimize/maximize)
            - Prediction uncertainty (prefers lower SEM when values are similar)
            
            **Example:**
            ```bash
            curl "http://localhost:8099/api/pareto?objective=drag"
            ```
            """,
        },
        "visualizer_endpoints.api.sensitivity": {
            "category": "Visualizer API",
            "content": """
            `POST /api/sensitivity` - Predict metrics for parameter variations.
            
            **Request Body:**
            ```json
            {
                "base_parameters": {"x": 0.5, "y": "value1"},
                "variations": {}
            }
            ```
            
            **Returns:**
            ```json
            {
                "predicted_means": {"drag": 0.123, "lift": 2.456},
                "predicted_sems": {"drag": 0.012, "lift": 0.089}
            }
            ```
            
            Used by the Sensitivity Analysis section to show real-time predictions
            as parameters are adjusted.
            """,
        },
        "visualizer_endpoints.api.pareto_html": {
            "category": "Visualizer API",
            "content": """
            `GET /api/pareto_html` - Open Pareto frontier report in browser.
            
            **Returns:**
            ```json
            {"status": "opened"}
            ```
            
            Generates and opens an interactive Pareto frontier visualization in the default browser.
            The report includes all Pareto-optimal points and their trade-offs.
            """,
        },
        "visualizer_endpoints.api.run_trial": {
            "category": "Visualizer API",
            "content": """
            `POST /api/run_trial` - Queue a manual trial with custom parameters.
            
            **Request Body:**
            ```json
            {
                "parameters": {"x": 0.5, "y": "value1"}
            }
            ```
            
            **Returns (new trial):**
            ```json
            {
                "status": "queued",
                "trial_index": 42
            }
            ```
            
            **Returns (duplicate detected):**
            ```json
            {
                "status": "existing",
                "trial_index": 15,
                "message": "Trial 15 already exists with these parameters"
            }
            ```
            
            **Features:**
            - Automatic parameter deduplication (MD5 hash comparison)
            - Background execution with status tracking
            - Real-time status updates via `/api/trial_status/{index}`
            
            **Example:**
            ```bash
            curl -X POST http://localhost:8099/api/run_trial \
              -H "Content-Type: application/json" \
              -d '{"parameters": {"x": 0.5, "y": "value1"}}'
            ```
            """,
        },
        "visualizer_endpoints.api.trial_status": {
            "category": "Visualizer API",
            "content": """
            `GET /api/trial_status/{index}` - Get real-time status of a running trial.
            
            **Path Parameters:**
            - `index`: Trial index number
            
            **Returns:**
            ```json
            {
                "trial_index": 42,
                "status": "running"  // or "completed", "failed: <error>", "not_found"
            }
            ```
            
            **Status Values:**
            - `"running"`: Trial is currently executing
            - `"completed"`: Trial finished successfully
            - `"failed: <error>"`: Trial failed with error message (truncated to 200 chars)
            - `"not_found"`: Trial index doesn't exist
            - `"COMPLETED"`, `"FAILED"`, etc.: Ax trial status names (for non-manual trials)
            
            The UI polls this endpoint every 2 seconds to update trial status in real-time.
            """,
        },
        "visualizer_endpoints.api.fit_model": {
            "category": "Visualizer API",
            "content": """
            `POST /api/fit_model` - Force model fitting to existing trials.
            
            **Returns:**
            ```json
            {
                "status": "Model fitting step executed"
            }
            ```
            
            Triggers a generation step to force the Bayesian model to fit to existing trial data.
            Useful when predictions fail due to insufficient model training.
            
            **When to use:**
            - After loading an experiment from storage
            - When prediction errors mention "not enough trials"
            - Before running sensitivity analysis on a new experiment
            """,
        },
        "visualizer_endpoints.api.viz_settings": {
            "category": "Visualizer API",
            "content": """
            `GET /api/viz_settings` - Get current visualization settings.
            
            **Returns:**
            ```json
            {
                "time_step": "latest",
                "field": "auto",
                "mesh_parts": "internal",
                "camera_angle": "isometric",
                "decompose_polyhedra": true,
                "cell_to_point": true,
                "skip_zero_time": true
            }
            ```
            
            Settings are stored server-side and persist across all trial visualizations
            until the server is restarted.
            """,
        },
        "visualizer_endpoints.api.viz_settings_update": {
            "category": "Visualizer API",
            "content": """
            `POST /api/viz_settings` - Update visualization settings.
            
            **Request Body:**
            ```json
            {
                "time_step": "latest",
                "field": "U",
                "mesh_parts": "internal,inlet,outlet",
                "camera_angle": "xy",
                "decompose_polyhedra": true,
                "cell_to_point": true,
                "skip_zero_time": true
            }
            ```
            
            **Returns:**
            ```json
            {
                "status": "ok",
                "settings": { /* updated settings */ }
            }
            ```
            
            **Settings Options:**
            - `time_step`: "latest" or time index (e.g., "0", "1", "2")
            - `field`: "auto" or field name (e.g., "U", "p", "T")
            - `mesh_parts`: Comma-separated list (e.g., "internal", "all_patches", "inlet,outlet")
            - `camera_angle`: "isometric", "xy", "xz", "yz", "x", "y", "z"
            - `decompose_polyhedra`: true/false - Decompose complex cells
            - `cell_to_point`: true/false - Convert cell data to point data
            - `skip_zero_time`: true/false - Skip /0 time directory
            
            Settings are applied to all subsequent trial visualizations.
            """,
        },
        "visualizer_endpoints.api.trial_visualization": {
            "category": "Visualizer API",
            "content": """
            `GET /api/trial_visualization/{index}` - Get PyVista visualization HTML.
            
            **Path Parameters:**
            - `index`: Trial index number
            
            **Returns:**
            ```json
            {
                "html": "<div>...PyVista screenshot...</div>",
                "trial_index": 42,
                "case_path": "/path/to/trial_42",
                "n_points": 125000,
                "n_cells": 120000,
                "time_points": ["0", "100", "200"],
                "arrays": ["U", "p", "T"],
                "mesh_names": ["internal", "inlet", "outlet", "walls"]
            }
            ```
            
            **Visualization Settings:**
            Uses server-side settings from `/api/viz_settings`. The UI automatically:
            1. Loads settings on page load
            2. Updates settings when user clicks "Apply Settings"
            3. Uses saved settings for all trial visualizations
            
            **Mesh Parts:**
            - `"internal"`: Internal mesh only
            - `"all_patches"`: All boundary patches
            - `"inlet,outlet"`: Specific patches (comma-separated)
            - Multiple parts are rendered with different colors
            
            **Camera Angles:**
            - `"isometric"`: 3D angled view (default)
            - `"xy"`: Top view (XY plane)
            - `"xz"`: Front view (XZ plane)
            - `"yz"`: Side view (YZ plane)
            - `"x"`, `"y"`, `"z"`: Direct axis views
            
            **Requirements:**
            - PyVista must be installed
            - Trial must have a valid case path with OpenFOAM results
            - OpenFOAM case must have a `.foam` file (created automatically if missing)
            
            **Output:**
            The HTML contains a high-quality screenshot (1400x1000px) with mesh info overlay.
            Settings are persistent across trials for consistent visualization.
            """,
        },
        "visualizer_endpoints.api.insights": {
            "category": "Visualizer API",
            "content": """
            `GET /api/insights` - Generate diagnostic analysis and insights.
            
            **Returns:**
            ```json
            {
                "html": "<div>...Plotly charts and analysis...</div>"
            }
            ```
            
            **Features:**
            - Automated experiment diagnostics (DiagnosticAnalysis)
            - Optimization insights and recommendations (InsightsAnalysis)
            - Interactive Plotly charts with dark theme
            - Parameter importance analysis
            - Convergence diagnostics
            
            **Requirements:**
            - Latest Ax version with `InsightsAnalysis` support
            - Plotly for interactive charts
            
            The HTML includes embedded Plotly.js scripts that execute automatically in the UI.
            """,
        },
    }
    return {**docs, **python_snippets, **visualizer_api}
