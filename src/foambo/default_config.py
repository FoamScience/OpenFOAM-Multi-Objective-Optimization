from ._version import VERSION
from typing import Dict, Any, get_args
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

def harvest_defaults(models_with_paths: list[tuple[type, str]]) -> Dict[str, Any]:
    """Build a default config dict from Pydantic model field defaults and examples.

    For each field, uses examples[0] if available (illustrative config value),
    otherwise falls back to the field default. Sub-model fields and List[SubModel]
    fields are recursively harvested.

    Args:
        models_with_paths: list of (ModelClass, "yaml_section") tuples.
    """
    from pydantic.fields import PydanticUndefined
    from pydantic import BaseModel
    import typing, types

    def _is_model(tp):
        try:
            return isinstance(tp, type) and issubclass(tp, BaseModel)
        except TypeError:
            return False

    def _unwrap_optional(annotation):
        """Unwrap Optional[X] / X | None to X."""
        origin = typing.get_origin(annotation)
        if origin is types.UnionType or origin is typing.Union:
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                return args[0]
        return annotation

    def _harvest_model(cls):
        result = {}
        for name, field_info in cls.model_fields.items():
            annotation = _unwrap_optional(field_info.annotation)

            # List[SubModel] → generate [_harvest_model(SubModel)]
            if typing.get_origin(annotation) is list:
                args = typing.get_args(annotation)
                if args and _is_model(args[0]):
                    result[name] = [_harvest_model(args[0])]
                    continue

            # Direct sub-model → recurse
            if _is_model(annotation):
                result[name] = _harvest_model(annotation)
                continue

            # Union of models (e.g. JSONBackendOptions | SQLBackendOptions) → use example
            # Fall through to example/default logic below

            # Prefer example (illustrative config value), then default
            if field_info.examples:
                result[name] = field_info.examples[0]
            elif field_info.default is not PydanticUndefined:
                result[name] = field_info.default
            # Skip fields with neither
        return result

    config = {}
    for cls, section_key in models_with_paths:
        config[section_key] = _harvest_model(cls)
    return config


# Mapping from YAML section names to root model classes.
# Defined here so get_default_config() and tests can share it.
def _get_root_models():
    from .orchestrate import (
        ConfigOrchestratorOptions, ExperimentOptions, TrialGenerationOptions,
        OptimizationOptions, BaselineOptions, StoreOptions, ExistingTrialsOptions,
    )
    return [
        (ExperimentOptions,          "experiment"),
        (TrialGenerationOptions,     "trial_generation"),
        (ExistingTrialsOptions,      "existing_trials"),
        (BaselineOptions,            "baseline"),
        (OptimizationOptions,        "optimization"),
        (ConfigOrchestratorOptions,  "orchestration_settings"),
        (StoreOptions,               "store"),
    ]

def _get_doc_models():
    """All models for docs harvesting, including nested ones not in root config."""
    from .orchestrate import (
        ConfigOrchestratorOptions, ExperimentOptions, TrialGenerationOptions,
        ModelSpecConfig, GenerationNodeConfig, CenterGenerationNodeConfig,
        OptimizationOptions, FoamJobRunnerOptions, VariableSubstOptions,
        FileSubstOptions, BaselineOptions, StoreOptions, ExistingTrialsOptions,
        TrialSelector, TrialAction, TrialDependency,
    )
    from .metrics import FoamJob, LocalJobMetric
    return [
        (ExperimentOptions,          "experiment"),
        (TrialGenerationOptions,     "trial_generation"),
        (ModelSpecConfig,            "trial_generation.generation_nodes[].generator_specs[]"),
        (GenerationNodeConfig,       "trial_generation.generation_nodes[]"),
        (CenterGenerationNodeConfig, "trial_generation.generation_nodes[] (center)"),
        (ExistingTrialsOptions,      "existing_trials"),
        (BaselineOptions,            "baseline"),
        (OptimizationOptions,        "optimization"),
        (LocalJobMetric,             "optimization.metrics[]"),
        (FoamJobRunnerOptions,       "optimization.case_runner"),
        (VariableSubstOptions,       "optimization.case_runner.variable_substitution[]"),
        (FileSubstOptions,           "optimization.case_runner.file_substitution[]"),
        (ConfigOrchestratorOptions,  "orchestration_settings"),
        (StoreOptions,               "store"),
        (TrialDependency,            "trial_dependencies[]"),
        (TrialSelector,              "trial_dependencies[].source"),
        (TrialAction,                "trial_dependencies[].actions[]"),
        (FoamJob,                    "internal.FoamJob"),
    ]


def get_default_config() -> Dict[str, Any]:
    """
    Return a default configuration dict, auto-harvested from Pydantic model
    field defaults and examples, with manual overlays for complex structures.
    """
    default = harvest_defaults(_get_root_models())
    default["version"] = str(VERSION)

    # --- Manual overlays for structures that can't be model-derived ---

    # Experiment parameters: list of different parameter types (raw dicts)
    default["experiment"]["parameters"] = [
        {
            "name": "x",
            "bounds": [0.0, 1.0],
            "step_size": None,
            "scaling": None,
            "parameter_type": "float",
        },
        {
            "name": "y",
            "values": ["zero", "one"],
            "is_ordered": None,
            "dependent_parameters": None,
            "parameter_type": "str",
        },
    ]

    # Generation nodes: complex list of different node types
    default["trial_generation"]["generation_nodes"] = [
        {"next_node_name": "init"},
        {
            "node_name": "init",
            "generator_specs": [
                {"generator_enum": "SOBOL", "model_kwargs": {"seed": 12345}}
            ],
            "transition_criteria": [
                {"type": "max_trials", "threshold": 7,
                 "transition_to": "BOM", "use_all_trials_in_exp": True}
            ],
        },
        {
            "node_name": "BOM",
            "generator_specs": [{"generator_enum": "BOTORCH_MODULAR"}],
            "transition_criteria": [],
        },
    ]

    # Stopping strategies: parsed by validators from dicts, use RangeTag
    default["orchestration_settings"]["global_stopping_strategy"] = {
        "min_trials": 10, "window_size": 5, "improvement_bar": 0.1
    }
    default["orchestration_settings"]["early_stopping_strategy"] = {
        "type": "percentile",
        "metric_names": ["metric"],
        "percentile_threshold": 25,
        "min_progression": 5,
        "trial_indices_to_ignore": RangeTag(0, 10, 1),
    }

    # Trial dependencies: illustrative warm-start example
    default["trial_dependencies"] = [
        {
            "name": "warm_start",
            "enabled": False,
            "source": {
                "strategy": "best",
                "fallback": "skip",
            },
            "actions": [
                {
                    "type": "run_command",
                    "command": "cp -rT $SOURCE_TRIAL/0.5 $TARGET_TRIAL/0",
                },
            ],
        },
    ]

    # Visualizer section (no model)
    default["visualizer"] = {"sensitivity_callback": None}

    return default

def harvest_docs(models_with_paths: list[tuple[type, str]]) -> Dict[str, Any]:
    """Harvest documentation from Pydantic model Field descriptions and docstrings.

    Args:
        models_with_paths: list of (ModelClass, "yaml.config.path") tuples.
            The path maps the Python class to its location in the YAML config file.
    """
    docs = {}
    for cls, yaml_path in models_with_paths:
        if cls.__doc__:
            docs[yaml_path] = cls.__doc__.strip()
        for name, field_info in cls.model_fields.items():
            if field_info.description:
                docs[f"{yaml_path}.{name}"] = field_info.description
    return docs


GITHUB_DOCS = "https://github.com/FoamScience/OpenFOAM-Multi-Objective-Optimization/blob/main/docs"

# Short summaries for each tutorial (avoids loading large markdown into the TUI)
_TUTORIAL_SUMMARIES: Dict[str, str] = {
    "00-implementation": (
        "Overview of foamBO's internal architecture: how Ax, foamlib, and "
        "Pydantic models fit together."
    ),
    "01-single-objective": (
        "Quick-start: fit a Gaussian Process to the F1 benchmark function. "
        "Covers YAML config, case setup, and generating predictions from "
        "trained surrogate models."
    ),
    "02-multi-objective": (
        "Advanced multi-objective optimization on OpenFOAM's lid-driven "
        "cavity. Covers SLURM integration, early stopping, Pareto analysis, "
        "sensitivity plots, and parameter dependency."
    ),
}


def load_tutorial_docs() -> Dict[str, Any]:
    """Return lightweight tutorial entries with summaries and GitHub links."""
    import pathlib
    docs_dir = pathlib.Path(__file__).parent.parent.parent / "docs"
    tutorials: Dict[str, Any] = {}
    if docs_dir.is_dir():
        for md_file in sorted(docs_dir.glob("*.md")):
            stem = md_file.stem
            if stem == "README":
                continue
            title = stem
            if len(title) > 3 and title[2] == '-' and title[:2].isdigit():
                title = title[3:]
            title = title.replace("-", " ").replace("_", " ").title()
            summary = _TUTORIAL_SUMMARIES.get(stem, "")
            link = f"{GITHUB_DOCS}/{stem}.md"
            tutorials[f"tutorial.{title}"] = {
                "category": "Tutorial",
                "content": f"{summary}\n\nFull tutorial: [{stem}.md]({link})",
            }
    return tutorials


def get_config_docs() -> Dict[str, Any]:
    # Phase 1: Harvest field descriptions from all Pydantic models
    harvested = harvest_docs(_get_doc_models())

    # Phase 2: Merge cross-cutting examples into harvested entries
    # Appends YAML examples and extra context to the field descriptions
    import textwrap
    _examples = {
        "experiment.parameters": f"""
            Accepts any arguments to these Ax class constructors:
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
        "trial_generation.method": f"""
            For finer control, use the "custom" strategy with generation nodes:
            ```yaml
            trial_generation:
                method: custom
                generation_nodes:
                    - next_node_name: init
                    -   node_name: init
                        generator_specs:
                            -   generator_enum: SOBOL
                                model_kwargs:
                                    seed: 12345
                        transition_criteria:
                            -   type: max_trials
                                threshold: 5
                                transition_to: BOM
                                use_all_trials_in_exp: true
                    -   node_name: BOM
                        generator_specs:
                            -   generator_enum: BOTORCH_MODULAR
                                transition_criteria: []
            ```

            Each node is either a:
            - {get_ax_path("GenerationNode", "ax/generation_strategy/generation_node.py", [85, 119])}
            - {get_ax_path("CenterGenerationNode", "ax/generation_strategy/center_generation_node.py", [26, 27])}

            Transition options are listed in
            {get_ax_path("transition_criterion.py", "ax/generation_strategy/transition_criterion.py", [117, 844])}
            """,
        "optimization.objective": """
            Single-objective minimization:
            ```yaml
            objective: "-metric"    # '-' for minimize
            ```

            Multi-objective:
            ```yaml
            objective: "-metric1, metric2"
            ```
            """,
        "optimization.outcome_constraints": """
            Constraints can be relative to the baseline:
            ```yaml
            outcome_constraints:
                - "metric1 >= 0.5*baseline"
                - "metric2 <= 10"
            ```
            Provide upper bounds for minimized metrics, lower bounds otherwise.
            """,
        "optimization.case_runner": """
            File substitutions deal with choice parameters (string values):
            ```yaml
            file_substitution:
                - parameter: param1
                  file_path: "/system/fvSolution"
            # Replaces with fvSolution.GAMG or fvSolution.PCG based on param1's value
            ```

            Variable substitution uses foamlib to assign values directly:
            ```yaml
            variable_substitution:
                - file: /0orig/field
                  parameter_scopes:
                      x: subDict.x
                      y: y
            ```
            """,
        "orchestration_settings.global_stopping_strategy": f"""
            Stops generating trials once improvement falls below a threshold:
            ```yaml
            orchestration_settings:
              global_stopping_strategy:
                improvement_bar: 0.1
                min_trials: 30
                window_size: 10
            ```
            See {get_ax_path("ImprovementGlobalStoppingStrategy", "ax/global_stopping/strategies/improvement.py", [52, 55])}
            """,
        "orchestration_settings.early_stopping_strategy": f"""
            Ax provides two strategies and logical `and`/`or` combinators:
            - {get_ax_path("Percentile-based", "ax/early_stopping/strategies/percentile.py", [31, 37])}
            - {get_ax_path("Threshold-based", "ax/early_stopping/strategies/threshold.py", [29, 35])}

            `trial_indices_to_ignore` supports the `!range` YAML constructor:
            ```yaml
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
            """,
        "baseline.parameters": """
            Set to `null` to skip:
            ```yaml
            baseline:
                parameters: null
            ```

            If defined, a single trial runs before the generation strategy:
            ```yaml
            baseline:
                parameters:
                    param1: 1.0
                    param2: "value2"
            ```
            """,
        "store.backend_options": """
            For SQL storage:
            ```yaml
            store:
              read_from: nowhere
              save_to: sql
              backend_options:
                url: <SQL_database_URL>
            ```
            Note: SQL store is currently not well tested.
            """,
        "trial_dependencies[]": """
            Define trial-to-trial relationships. Each dependency selects a source trial
            and runs actions before the new trial's solver starts.

            Warm-start from the best trial's converged fields:
            ```yaml
            trial_dependencies:
              - name: warm_start
                source:
                  strategy: best
                  fallback: skip
                actions:
                  - type: run_command
                    command: "cp -rT $SOURCE_TRIAL/0.5 $TARGET_TRIAL/0"
            ```

            Copy mesh from the nearest completed trial:
            ```yaml
            trial_dependencies:
              - name: mesh_inherit
                source:
                  strategy: nearest
                  fallback: skip
                actions:
                  - type: run_command
                    command: "cp -rT $SOURCE_TRIAL/constant/polyMesh $TARGET_TRIAL/constant/polyMesh"
            ```

            Use OpenFOAM's mapFields to interpolate between different meshes:
            ```yaml
            trial_dependencies:
              - name: map_fields
                source:
                  strategy: best
                  fallback: skip
                actions:
                  - type: run_command
                    command: "mapFields $SOURCE_TRIAL -sourceTime latestTime -targetRegion $TARGET_TRIAL"
            ```

            Actions run in order after the template case is cloned and parameters are
            substituted, but before the runner command executes. The dependency resolution
            result is recorded in `run_metadata["dependencies"]` for traceability.
            """,
        "trial_generation.generation_nodes[].generator_specs[].transforms": """
            Ax applies a chain of data transforms before fitting the GP surrogate.
            The default chain can hurt accuracy on functions with large dynamic range
            or sharp features. You can override or filter it per generation node.

            **Available transforms (applied in order):**

            | Transform | What it does |
            |---|---|
            | `Cast` | Casts parameter types to match the search space |
            | `MapKeyToFloat` | Maps dict keys to float indices |
            | `RemoveFixed` | Removes fixed (non-optimizable) parameters |
            | `OrderedChoiceToIntegerRange` | Maps ordered choice params to integers |
            | `OneHot` | One-hot encodes unordered choice parameters |
            | `IntToFloat` | Converts log-scaled int params to floats |
            | `Log` | Applies log transform to log-scaled parameters |
            | `Logit` | Applies logit transform to logit-scaled parameters |
            | `Winsorize` | Clips outlier Y values to reduce their influence |
            | `Derelativize` | Converts relative constraints to absolute |
            | `BilogY` | Compresses Y via `sign(y)*log(1+|y|)` — reduces dynamic range |
            | `StandardizeY` | Standardizes Y to zero mean, unit variance |

            **Common issues and fixes:**

            `BilogY` compresses large values (e.g. 0.1 and 100 become 0.1 and 4.6),
            making it harder for the GP to distinguish them. `Winsorize` clips outliers,
            which can discard valid extreme observations.

            To disable both:
            ```yaml
            generator_specs:
              - generator_enum: BOTORCH_MODULAR
                exclude_transforms: [BilogY, Winsorize]
            ```

            For minimal transforms (best accuracy, less robustness):
            ```yaml
            generator_specs:
              - generator_enum: BOTORCH_MODULAR
                transforms: [StandardizeY]
            ```

            **Library API:**
            ```python
            FoamBO("Exp")
                .transforms(exclude=["BilogY", "Winsorize"])
                # or: .transforms(only=["StandardizeY"])
            ```

            **When to customize:**
            - GP predictions don't pass through trial points → try removing `BilogY`
            - Extreme values are being ignored → try removing `Winsorize`
            - Model fit warnings ("unable to be reliably fit") → try minimal transforms
            - Noisy real-world data with outliers → keep defaults (they add robustness)

            **Custom GP kernels (library API only):**

            The default Matérn-2.5 kernel uses a single lengthscale, which can't
            capture functions with both smooth trends and sharp local features.
            An additive kernel combining different lengthscales fixes this:

            ```python
            from gpytorch.kernels import ScaleKernel, MaternKernel, AdditiveKernel

            class AdditiveMatern(AdditiveKernel):
                def __init__(self, **kwargs):
                    super().__init__(
                        ScaleKernel(MaternKernel(nu=2.5)),  # smooth trend
                        ScaleKernel(MaternKernel(nu=0.5)),  # rough detail
                    )

            client = (
                FoamBO("Exp")
                .kernel(AdditiveMatern)
                .transforms(exclude=["BilogY", "Winsorize"])
                .minimize("metric", fn=my_fn)
                .run()
            )
            ```

            This is not available via YAML config — custom kernels require the
            Python library API since they involve gpytorch class definitions.

            **When to use a custom kernel:**
            - Single-objective with oscillatory or multi-modal objective functions
            - "Model fit is poor" warnings despite sufficient trials
            - GP predictions systematically miss trial points even with transform tuning
            """,
    }
    for key, extra in _examples.items():
        extra = textwrap.dedent(extra).strip()
        if key in harvested:
            harvested[key] = harvested[key] + "\n\n" + extra
        else:
            harvested[key] = extra

    # Phase 3: Standalone entries not tied to model fields
    harvested["version"] = f"The foamBO version to run this configuration with (v{str(VERSION)})"
    harvested["visualizer"] = "[Section] Settings for the web-based visualizer UI"
    harvested["visualizer.sensitivity_callback"] = textwrap.dedent("""
        Optional callback for custom parameter visualization in sensitivity analysis.
        The callback receives a `dict` of parameters and must return a base64-encoded PNG string.

        **YAML config (dotted import path):**
        ```yaml
        visualizer:
          sensitivity_callback: "mymodule.plot_function"
        ```

        **Library API (Python callable):**
        ```python
        def my_plot(parameters: dict) -> str:
            import matplotlib.pyplot as plt, io, base64
            fig, ax = plt.subplots()
            # ... render parameters ...
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            return base64.b64encode(buf.getvalue()).decode()

        FoamBO("Exp")
            .visualizer(sensitivity_fn=my_plot)
            .show()
        ```

        Set to `null` to disable.
    """).strip()

    harvested["python.library_usage"] = {
        "category": "Python snippet",
        "content": """
            foamBO can be used as a Python library with a fluent API:

            **With OpenFOAM case (shell commands):**
            ```python
            from foambo import FoamBO

            client = (
                FoamBO("MyExperiment", case="./case", trials="./trials", artifacts="./artifacts")
                .parameter("x", bounds=[-100.0, 200.0])
                .parameter("y", values=["A", "B", "C"])
                .minimize("cost", command="python3 evaluate.py")
                .maximize("quality", command="./compute_quality.sh")
                .track("residuals", command="./get_residuals.sh", lower_is_better=True)
                .substitute("/0orig/U", x="internalField")
                .file_substitute("y", "/system/fvSolution")
                .constraint("x <= 100")
                .outcome_constraint("cost <= 50")
                .baseline(x=0.5, y="A")
                .stop(max_trials=50, improvement_bar=0.1)
                .early_stop(type="percentile", metric_names=["residuals"],
                            percentile_threshold=25, min_progression=5)
                .depend("warm_start", source="best",
                        command="cp -rT $SOURCE_TRIAL/0.5 $TARGET_TRIAL/0")
                .run(parallelism=3, poll_interval=10, ttl=600)
            )
            predictions = client.predict([{"x": 10, "y": "A"}])
            ```

            **Pure Python (no case files, no OpenFOAM):**
            ```python
            from foambo import FoamBO

            def my_objective(parameters):
                return parameters["x"] ** 2 + parameters["y"] ** 2

            client = (
                FoamBO("PythonOpt")
                .parameter("x", bounds=[-10.0, 10.0])
                .parameter("y", bounds=[-10.0, 10.0])
                .minimize("loss", fn=my_objective)
                .stop(max_trials=50)
                .run(parallelism=3)
            )
            ```
            When all metrics use `fn=` and no `case=` is set, foamBO runs in
            caseless mode — no template cloning, no file substitution, no foamlib.

            **Conditional parameters:**
            ```python
            FoamBO("Exp")
                .parameter("solver", values=["PCG", "GAMG"],
                           depends={"PCG": ["pcg_tol"], "GAMG": ["gamg_tol"]})
                .parameter("pcg_tol", bounds=[1e-6, 1e-4], scaling="log")
                .parameter("gamg_tol", bounds=[1e-6, 1e-4], scaling="log")
            ```
            `depends=` maps each choice value to the parameters that are only
            active when that value is selected. Ax learns this structure natively.

            **Key methods:**
            - `.parameter(name, bounds=... | values=..., depends=...)` — add a parameter
            - `.minimize(name, command=... | fn=...)` / `.maximize(...)` — add an objective
            - `.track(name, command=... | fn=...)` — tracking metric (for early stopping)
            - `.substitute(file, param=path)` — map params to OpenFOAM fields
            - `.stop(max_trials, improvement_bar)` — global stopping
            - `.early_stop(type, ...)` — trial-level early stopping
            - `.depend(name, source, command)` — trial-to-trial dependencies
            - `.visualizer(sensitivity_fn=...)` — configure visualization callback
            - `.show()` — launch the web UI
            - `.preflight(dry_run=True)` — validate config before running
            - `.run(parallelism, ...)` — execute and return the Ax Client
            - `.build()` — return a `FoamBOConfig` without running
            """,
    }

    harvested["python.exploration_vs_exploitation"] = {
        "category": "Python snippet",
        "content": """
            **Controlling surrogate accuracy vs. optimum-seeking**

            Standard BO (Expected Improvement) aggressively hunts for the optimum,
            leaving large parameter regions unsampled. Two techniques to build
            a more globally accurate surrogate model:

            **1. Exploratory acquisition function (Upper Confidence Bound)**

            High `beta` in UCB means "sample where uncertainty is high" rather
            than "sample where improvement is likely":
            ```yaml
            trial_generation:
                method: custom
                generation_nodes:
                    - next_node_name: sobol
                    -   node_name: sobol
                        generator_specs:
                            - generator_enum: SOBOL
                        transition_criteria:
                            - type: max_trials
                              threshold: 10
                              transition_to: explore
                    -   node_name: explore
                        generator_specs:
                            - generator_enum: BOTORCH_MODULAR
                              model_kwargs:
                                  botorch_acqf_class: qUpperConfidenceBound
                                  acquisition_options:
                                      beta: 10.0
                        transition_criteria: []
            ```
            `beta=10` is very exploratory; `beta=0.1` is nearly pure exploitation.

            **1. Two-phase: explore then exploit**

            Build a globally accurate surrogate first, then switch to optimization:
            ```yaml
            trial_generation:
                method: custom
                generation_nodes:
                    - next_node_name: sobol
                    -   node_name: sobol
                        generator_specs:
                            - generator_enum: SOBOL
                        transition_criteria:
                            - type: max_trials
                              threshold: 10
                              transition_to: explore
                    -   node_name: explore
                        generator_specs:
                            - generator_enum: BOTORCH_MODULAR
                              model_kwargs:
                                  botorch_acqf_class: qUpperConfidenceBound
                                  acquisition_options:
                                      beta: 10.0
                        transition_criteria:
                            - type: max_trials
                              threshold: 30
                              transition_to: exploit
                    -   node_name: exploit
                        generator_specs:
                            - generator_enum: BOTORCH_MODULAR
                        transition_criteria: []
            ```
            Phase 1 (SOBOL) seeds the model. Phase 2 (high-beta UCB) fills gaps.
            Phase 3 (default EI) exploits the now-accurate surrogate.

            To validate surrogate accuracy, use ``FoamBO.cross_validate()``:
            ```python
            client = FoamBO("Exp").minimize("F1", fn=my_fn).stop(max_trials=50).run()

            cv = FoamBO.cross_validate(client)
            for row in cv:
                err = abs(row["observed_mean"] - row["predicted_mean"])
                print(f"Trial {row['trial_index']} {row['metric_name']}: "
                      f"observed={row['observed_mean']:.3f} "
                      f"predicted={row['predicted_mean']:.3f} error={err:.3f}")
            ```
            Large errors in specific regions indicate where the surrogate
            needs more samples. To improve coverage, add space-filling trials:
            ```python
            client = (
                FoamBO("Exp").minimize("F1", fn=my_fn)
                .generation(method="random_search")  # pure SOBOL
                .stop(max_trials=20).resume().run()
            )
            ```
            The analysis HTML reports also include cross-validation plots
            when a BO model has been fitted.
            """,
    }

    harvested["python.loading_client_state"] = {
        "category": "Python snippet",
        "content": f"""
            You can load an [Ax](https://ax.dev) client for your experiment with:
            ```python
            from foambo.common import set_experiment_name
            from foambo.orchestrate import StoreOptions
            set_experiment_name("MyExperiment")
            store = StoreOptions.model_validate({{"save_to": "json", "read_from": "json", "backend_options": {{}}}})
            client = store.load()
            ```
            """,
    }

    harvested["python.resume_from_library"] = {
        "category": "Python snippet",
        "content": """
            Resume a previously saved experiment from Python:

            ```python
            from foambo import FoamBO

            client = (
                FoamBO("MyExperiment", case="./case")
                .parameter("x", bounds=[0.0, 1.0])
                .minimize("metric", command="./evaluate.sh")
                .substitute("/FxDict", x="x")
                .stop(max_trials=100)
                .resume()              # load from saved JSON state
                .run(parallelism=4)
            )
            ```
            """,
    }

    harvested["visualizer_ui"] = {
        "category": "Visualizer API",
        "content": """
            Launch an interactive web UI to explore optimization results and run manual trials.

            ```python
            from foambo.visualize import visualizer_ui
            from foambo.config import load_config

            cfg = load_config("config.yaml")
            visualizer_ui(cfg, host="127.0.0.1", port=8099, open_browser=True)
            ```

            **Features:** Experiment overview, Pareto frontier, sensitivity analysis,
            manual trials, OpenFOAM visualization (PyVista), diagnostics & insights.

            **Endpoints:** `/api/experiment`, `/api/trials`, `/api/pareto`,
            `/api/sensitivity`, `/api/run_trial`, `/api/trial_status/{index}`,
            `/api/fit_model`, `/api/viz_settings`, `/api/trial_visualization/{index}`,
            `/api/insights`, `/api/pareto_html`
            """,
    }

    # Phase 4: Load tutorials from docs/*.md
    tutorials = load_tutorial_docs()
    harvested.update(tutorials)

    return harvested
