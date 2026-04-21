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
        OptimizationOptions, BaselineOptions, StoreOptions,
    )
    return [
        (ExperimentOptions,          "experiment"),
        (TrialGenerationOptions,     "trial_generation"),
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
        SeedDataNodeConfig,
        OptimizationOptions, FoamJobRunnerOptions, VariableSubstOptions,
        FileSubstOptions, BaselineOptions, StoreOptions,
        TrialSelector, TrialDependency,
        DimensionalityReductionOptions,
    )
    from .metrics import FoamJob, LocalJobMetric
    from .robustness import RobustOptimizationConfig
    return [
        (RobustOptimizationConfig,   "robust_optimization"),
        (ExperimentOptions,          "experiment"),
        (TrialGenerationOptions,     "trial_generation"),
        (ModelSpecConfig,            "trial_generation.generation_nodes[].generator_specs[]"),
        (GenerationNodeConfig,       "trial_generation.generation_nodes[]"),
        (CenterGenerationNodeConfig, "trial_generation.generation_nodes[] (center)"),
        (SeedDataNodeConfig,         "trial_generation.generation_nodes[] (seed_data)"),
        (BaselineOptions,            "baseline"),
        (OptimizationOptions,        "optimization"),
        (LocalJobMetric,             "optimization.metrics[]"),
        (FoamJobRunnerOptions,       "optimization.case_runner"),
        (VariableSubstOptions,       "optimization.case_runner.variable_substitution[]"),
        (FileSubstOptions,           "optimization.case_runner.file_substitution[]"),
        (ConfigOrchestratorOptions,  "orchestration_settings"),
        (DimensionalityReductionOptions, "orchestration_settings.dimensionality_reduction"),
        (StoreOptions,               "store"),
        (TrialDependency,            "trial_dependencies[]"),
        (TrialSelector,              "trial_dependencies[].source"),
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
            "groups": ["geometry"],
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

    # Trial dependencies: illustrative examples
    default["trial_dependencies"] = [
        {
            "name": "warm_start",
            "enabled": False,
            "source": {
                "strategy": "best",
                "fallback": "skip",
            },
        },
        {
            "name": "reuse_mesh",
            "enabled": False,
            "source": {
                "strategy": "matching_group",
                "group": "geometry",
                "fallback": "skip",
            },
        },
    ]

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
            - ``SeedDataNode`` (foamBO-specific): seeds the experiment with
              pre-existing trials from a CSV or a saved foamBO JSON state.
              Detected by the presence of a ``file_path`` key on the node dict.

            Example seeding from a prior foamBO JSON run, then continuing with BO:
            ```yaml
            trial_generation:
                method: custom
                generation_nodes:
                    -   file_path: ./artifacts/prior_run.json
                        drop_params: [c]            # c not in current search space
                        filter:
                            T: {{value: 350, tolerance: 5, direction: both}}
                        next_node: mbm
                    -   node_name: mbm
                        generator_specs:
                            -   generator_enum: BOTORCH_MODULAR
                        transition_criteria: []
            ```

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
        "optimization.objective_thresholds": """
            Reference point for multi-objective Pareto hypervolume computation.
            Defines "minimally acceptable" values for each objective.
            If not set, Ax infers them from data (which can crash with incomplete data
            from failed trials).

            Supports absolute values and baseline-relative expressions:
            ```yaml
            objective_thresholds:
                - "efficiency >= 0.3"
                - "pressureHead >= 0.9*baseline"
                - "torque <= 1.2*baseline"
            ```

            Use ``>=`` for maximized objectives, ``<=`` for minimized.
            Baseline expressions are resolved after the baseline trial completes.

            **Library API:**
            ```python
            FoamBO("Exp")
                .maximize("efficiency", ...)
                .minimize("torque", ...)
                .baseline(efficiency=0.45, torque=30.0)
                .objective_threshold("efficiency >= 0.8*baseline")
                .objective_threshold("torque <= 1.2*baseline")
            ```

            Strongly recommended for multi-objective optimization to avoid
            inference failures when trials produce incomplete metric data.

            **Objective thresholds vs outcome constraints:**

            These serve different purposes and interact in specific ways:

            - ``objective_thresholds`` define the **Pareto reference point** — the
              "worst acceptable" corner of objective space for hypervolume computation.
              Designs below a threshold still run and are evaluated, they just don't
              contribute to the Pareto hypervolume.

            - ``outcome_constraints`` are **hard feasibility filters** — designs that
              violate them are excluded from the Pareto front entirely, regardless of
              how good their other objectives are.

            When Ax computes the Pareto frontier:
            1. If explicit ``objective_thresholds`` are set, they are used directly
               as the reference point.
            2. If no thresholds are set, Ax infers them from the data — this can fail
               when trials have incomplete metrics (e.g. failed evaluations).
            3. If the resulting Pareto front is empty AND ``outcome_constraints`` exist,
               Ax retries without constraints (temporary relaxation) to find at least
               some frontier points.
            4. If the front is still empty and no constraints exist, Ax raises an error.

            Setting ``objective_thresholds`` explicitly avoids steps 2-4 entirely,
            making the optimization robust to trial failures and incomplete data.
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

            **Why objectives don't stream (and how to work around it):**

            Only non-objective (tracking) metrics are streamed during a trial's
            execution. Objective metrics are fetched once at trial completion.
            This is deliberate — streaming intermediate objective values would
            create ``MapData`` (time-series) that could confuse the GP surrogate
            and Pareto computation.

            If you need to early-stop based on an objective's behavior mid-run
            (e.g. detecting pressure divergence in a CFD solver), create a
            **separate tracking metric** with the same command:

            ```yaml
            metrics:
              - name: pressureHead               # objective (maximized)
                command: ["scripts/metric.sh", "pressureHead"]
              - name: inletPressure              # tracking (for early stopping)
                command: ["scripts/metric.sh", "inletPressure"]
                progress: ["scripts/metric.sh", "inletPressure"]
                progression_source: "foam_time:log.simpleFoam"
                lower_is_better: true             # important for threshold direction

            early_stopping_strategy:
              type: threshold
              metric_signatures: ["inletPressure"]
              metric_threshold: 10000             # stop if inlet pressure > 10k
            ```

            Note the ``lower_is_better`` setting matters for threshold direction:
            - ``lower_is_better: true`` + ``threshold: X`` → stops when value **> X**
            - ``lower_is_better: false`` + ``threshold: X`` → stops when value **< X**

            Using the objective directly would invert the threshold logic because
            the objective's maximize/minimize sense controls ``lower_is_better``.
            The tracking metric lets you set the direction independently.
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
            and runs actions either immediately (before the runner starts) or deferred
            to a hook phase that the runner invokes at the right moment.

            Warm-start from the best trial's converged fields (immediate, default):
            ```yaml
            trial_dependencies:
              - name: warm_start
                source:
                  strategy: best
                  fallback: skip
            ```

            Reuse mesh when geometry parameters haven't changed:
            ```yaml
            trial_dependencies:
              - name: reuse_mesh
                source:
                  strategy: matching_group
                  group: "geometry"
                  fallback: skip
              - name: warm_fields
                source:
                  strategy: nearest
                  similarity_threshold: 0.3
                  fallback: skip
            ```

            With parameters tagged by group:
            ```yaml
            experiment:
              parameters:
                - name: angle1
                  bounds: [20, 40]
                  groups: ["geometry"]
                - name: relaxation
                  bounds: [0.1, 0.9]
            ```

            **How it works:** foamBO resolves each dependency and writes
            `.foambo_deps.json` to the trial case directory.  The manifest
            path is available via `$FOAMBO_DEPS`.  The runner script reads
            it and decides what to do:

            ```bash
            #!/bin/bash
            deps="$FOAMBO_DEPS"

            # Mesh: reuse or generate
            if jq -e '.reuse_mesh.resolved' "$deps" >/dev/null 2>&1; then
                src=$(jq -r '.reuse_mesh.source_path' "$deps")
                cp -rT "$src/constant/polyMesh" constant/polyMesh
            else
                blockMesh
            fi

            # Fields: warm-start or cold start
            if jq -e '.warm_fields.resolved' "$deps" >/dev/null 2>&1; then
                src=$(jq -r '.warm_fields.source_path' "$deps")
                mapFields "$src" -sourceTime latestTime
            else
                potentialFoam -writep > log.potentialFoam 2>&1
            fi

            simpleFoam
            ```

            **Non-shell runners** (Python, binary):
            ```python
            import json, os
            deps = json.load(open(os.environ["FOAMBO_DEPS"]))
            if deps["reuse_mesh"]["resolved"]:
                shutil.copytree(deps["reuse_mesh"]["source_path"] + "/constant/polyMesh",
                                "constant/polyMesh", dirs_exist_ok=True)
            ```

            **Manifest format** (`.foambo_deps.json`):
            ```json
            {
              "reuse_mesh": {"resolved": true, "source_trial_index": 3, "source_path": "/path/to/trial_0003"},
              "warm_fields": {"resolved": false}
            }
            ```

            **Environment variables available to the runner:**
            - `$FOAMBO_DEPS` — path to `.foambo_deps.json`
            - `$FOAMBO_CASE_PATH` / `$FOAMBO_CASE_NAME` — trial case directory

            The dependency resolution result is recorded in
            `run_metadata["dependencies"]` for traceability.
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

    harvested["bootstrap"] = {
        "category": "Config",
        "content": textwrap.dedent("""
            Inherit configuration, search space, trials, and fitted GP from a
            previously saved run. The value is a path (absolute or relative to
            the YAML file) to a ``*_client_state.json`` produced by foamBO. The
            saved state embeds the original ``foambo_config``; the current YAML
            is merged on top via OmegaConf, so any key can be overridden.

            **Two typical workflows:**

            1. *Continue with more trials under a new name:*
            ```yaml
            bootstrap: ./artifacts/PumpOpt_client_state.json

            experiment:
              name: PumpOpt_continued   # new save target

            orchestration_settings:
              n_trials: 60              # extend the budget
            ```

            2. *Specialize a robust run to a fixed context point:*
            ```yaml
            bootstrap: ./artifacts/PumpOpt_client_state.json

            experiment:
              name: PumpOpt_spec_op1

            robust_optimization: null   # disable robust mode

            specialize:
              flowRate: 0.022
              rpm: 2811

            orchestration_settings:
              n_trials: 30
            ```

            **Merge rules:**
            - Dict keys merge deeply; scalars in YAML override parent values.
            - Setting a block to ``null`` in YAML clears it (e.g., dropping
              ``robust_optimization``).
            - Parent ``experiment.parameters`` is inherited verbatim unless
              explicitly overridden. Changing parameter bounds / types is not
              safe when trials already exist — use ``specialize`` instead.

            **What gets reused automatically:**
            The loaded Ax Client retains the full trial history and fitted GP.
            ``_model.pt`` alongside the state JSON warm-starts the surrogate
            (0.15s vs 9s cold fit). Analysis cards, generation-node progress,
            and attached metadata are all preserved.
        """).strip(),
    }

    harvested["specialize"] = {
        "category": "Config",
        "content": textwrap.dedent("""
            Only meaningful when ``bootstrap`` is set. A mapping of parameter
            name → fixed value. Each listed parameter is converted to an Ax
            ``FixedParameter`` in the inherited search space; all recorded
            trial arms are rewritten so the fixed parameter equals the pinned
            value. The GP refits on this clamped dataset at the next
            generation call, and subsequent candidates only vary the remaining
            design dimensions.

            ```yaml
            specialize:
              flowRate: 0.022    # context variable from the parent robust run
              rpm: 2811
            ```

            **Intended use:** after a robust optimization covers a context
            distribution, pick a concrete operating point and continue tuning
            design parameters under that point. Pairs naturally with
            ``robust_optimization: null``.

            **Trade-off:** rewriting arm parameters discards information about
            how the specialized variable influenced outcomes — past trials
            now all appear to share the fixed value. This is the correct
            behavior for conditional optimization; if you want the GP to
            retain context sensitivity, use ``bootstrap`` alone and keep
            ``robust_optimization``.

            **Requirements:** each specialized key must exist in the parent
            search space. Parent experiments produced by foamBO carry Ax's
            ``immutable_search_space_and_opt_config=True`` flag (it only means
            generator runs didn't cache per-trial search-space copies); the
            bootstrap loader clears that flag automatically before mutating.
        """).strip(),
    }
    harvested["experiment.parameters[].is_fidelity"] = {
        "category": "Config",
        "content": textwrap.dedent("""
            Mark a parameter as the fidelity dimension for multi-fidelity BO.
            The parameter represents evaluation quality — e.g. 0 = cheap
            meanline, 1 = expensive CFD. Exactly one parameter should have
            this flag. ``target_value`` is the high-fidelity level the
            optimizer ultimately cares about.

            ```yaml
            experiment:
              parameters:
                - name: fidelity
                  parameter_type: float
                  bounds: [0.0, 1.0]
                  is_fidelity: true
                  target_value: 1.0
            ```

            When detected, foamBO auto-selects ``SingleTaskMultiFidelityGP``
            as the surrogate and extracts ``target_fidelities`` for the
            acquisition function. The runner should branch on the fidelity
            value to dispatch cheap vs expensive evaluations.
        """).strip(),
    }

    harvested["optimization.metrics[].is_cost"] = {
        "category": "Config",
        "content": textwrap.dedent("""
            Mark a metric as the cost signal for multi-fidelity acquisition.
            The metric value should represent actual execution cost (e.g.
            wall-clock seconds) and must be emitted by the runner at every
            fidelity level.

            ```yaml
            optimization:
              metrics:
                - name: executionTime
                  command: ["scripts/metric.sh", "executionTime"]
                  is_cost: true
            ```

            Exactly one metric should have ``is_cost: true``. foamBO learns
            per-fidelity mean cost from observed values and updates the
            acquisition function's cost model each callback cycle. Without
            an ``is_cost`` metric, MF acquisition uses uniform cost
            (no cost-aware fidelity selection).
        """).strip(),
    }

    harvested["trial_generation.generation_nodes[].generator_specs[].model_kwargs.botorch_acqf_class"] = {
        "category": "Config",
        "content": textwrap.dedent("""
            Override the default BoTorch acquisition function class. Accepts
            a string class name that foamBO resolves to the actual Python
            class. Supported MF acquisition functions:

            ``qMultiFidelityHypervolumeKnowledgeGradient`` (recommended):
            - Multi-objective, cost-aware, one-step lookahead.
            - Maximizes expected hypervolume improvement at target fidelity.
            - Cost model auto-wired from ``is_cost`` metric via
              ``cost_intercept`` + ``fidelity_weights``.

            ``MOMF`` (alternative):
            - Multi-objective multi-fidelity via fidelity pseudo-objective.
            - Faster candidate generation, less sample efficient.
            - Requires manual ``cost_call`` in ``botorch_acqf_options``.

            ```yaml
            trial_generation:
              method: custom
              generation_nodes:
                - node_name: MF
                  generator_specs:
                    - generator_enum: BOTORCH_MODULAR
                      model_kwargs:
                        botorch_acqf_class: "qMultiFidelityHypervolumeKnowledgeGradient"
            ```

            With ``method: fast``, setting ``botorch_acqf_class`` is not
            needed — foamBO auto-selects ``qMultiFidelityHypervolumeKnowledgeGradient``
            when an ``is_fidelity`` parameter is detected. Custom generation
            nodes are only needed to override the default (e.g. pick MOMF).

            **Runner dispatch (recommended):** use ``file_substitution`` with
            ``value_map`` to map numeric fidelity to file suffixes. Place
            ``Allrun.meanline`` + ``Allrun.CFD`` in the template case:
            ```yaml
            experiment:
              parameters:
                - name: fidelity
                  bounds: [0, 1]
                  parameter_type: int
                  is_fidelity: true
                  target_value: 1

            optimization:
              case_runner:
                file_substitution:
                  - parameter: fidelity
                    file_path: /Allrun
                    value_map: {0: "meanline", 1: "CFD"}
            ```

            Other resolvable names: ``qMultiFidelityKnowledgeGradient``
            (single-objective MF), ``qMultiFidelityMaxValueEntropy``.
        """).strip(),
    }

    harvested["dashboard"] = {
        "category": "Dashboard",
        "content": textwrap.dedent("""
            Web dashboard and REST API settings.

            foamBO starts a web dashboard automatically during optimization (unless ``--no-ui``).
            The dashboard provides real-time monitoring of trials, objectives, streaming metrics,
            generation strategy progress, and on-demand model analysis.

            **Configuration** (via ``orchestration_settings``):
            - ``api_host``: Bind address (default: ``127.0.0.1``)
            - ``api_port``: Port number (default: ``8098``; auto-picks a free port if in use; ``0`` to disable)

            **CLI flags:**
            - ``--no-ui``: Disable the web dashboard entirely
            - ``--no-opt``: Load a saved experiment and launch the dashboard without running optimization

            **Post-optimization viewing:**
            ```bash
            foamBO --no-opt --config MyOpt.yaml ++store.read_from=json
            ```

            **Trial visualization** supports uploading a ``pvpython`` script for custom rendering:
            - ``sys.argv[1]`` → case folder path
            - ``sys.argv[2]`` → screenshot filename (saved inside case folder)

            **Analysis tab** ("Generate Insights") refits the surrogate model on demand and provides:
            sensitivity analysis, parallel coordinates, cross-validation, contour plots, and healthchecks.
        """).strip(),
    }

    harvested["orchestration_settings.api_host"] = {
        "category": "Config",
        "content": textwrap.dedent("""
            foamBO uses Ax's native poll-based orchestration. Ax sleeps between
            poll cycles and checks subprocess status via the configured runner.
            Remote trials (SLURM/SSH) that can't be polled locally notify
            completion by POSTing to the live REST API. The runner's
            ``poll_trial`` consumes push state on Ax's next poll cycle.

            **Remote runner configuration:**
            For SLURM or SSH-based runners, ``api_host`` must be set to an address
            reachable by all compute nodes (e.g. the login node hostname, or
            ``0.0.0.0`` to bind all interfaces):
            ```yaml
            orchestration_settings:
              api_host: login01.cluster.local   # reachable by compute nodes
              api_port: 8098
            ```

            **Environment variables** injected into every trial subprocess:
            - ``FOAMBO_API_ENDPOINT`` — full API URL (e.g. ``http://login01:8098/api/v1``)
            - ``FOAMBO_TRIAL_INDEX`` — the trial number
            - ``FOAMBO_SESSION_ID`` — unique ID for this foamBO instance (for rogue job detection)

            **Allrun completion notification** (POST sets status override that
            ``FoamJobRunner.poll_trial`` consumes on next poll cycle):
            ```bash
            curl -s -X POST "$FOAMBO_API_ENDPOINT/trials/$FOAMBO_TRIAL_INDEX/push/status" \\
              -H "Content-Type: application/json" \\
              -d '{"status":"completed","exit_code":'$?',"session_id":"'$FOAMBO_SESSION_ID'"}'
            ```

            **Streaming metrics from OpenFOAM function objects** (attached via
            the push queue for early-stopping decisions):
            ```bash
            curl -s -X POST "$FOAMBO_API_ENDPOINT/trials/$FOAMBO_TRIAL_INDEX/push/metrics" \\
              -H "Content-Type: application/json" \\
              -d '{"metrics":{"continuityErrors":'$VALUE'},"step":'$TIME',"session_id":"'$FOAMBO_SESSION_ID'"}'
            ```

            Harvest latency is bounded by ``init_seconds_between_polls`` in the
            Ax scheduler options.
        """).strip(),
    }

    # Merge concept docs from separate file
    from .docs_concepts import CONCEPTS
    harvested.update(CONCEPTS)

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
                .depend("warm_start", source="best")
                .depend("reuse_mesh", source="matching_group", group="geometry")
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
            - `.kernel(cls)` — custom GP kernel (gpytorch)
            - `.transforms(exclude=[...])` — control Ax transform pipeline
            - `.stop(max_trials, improvement_bar)` — global stopping
            - `.early_stop(type, ...)` — trial-level early stopping
            - `.reduce(after_trials, min_importance)` — auto-fix irrelevant parameters
            - `.depend(name, source)` — trial-to-trial dependencies
            - `.preflight(dry_run=True)` — validate config before running
            - `.run(parallelism, ...)` — execute and return the Ax Client
            - `FoamBO.load(name)` — load a saved experiment for analysis
            - `FoamBO.cross_validate(client)` — leave-one-out CV on the surrogate
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
            needs more samples. Re-run with more trials or a space-filling
            strategy to improve coverage. The analysis HTML reports also
            include cross-validation plots when a BO model has been fitted.
            """,
    }

    harvested["python.dimensionality_reduction"] = {
        "category": "Python snippet",
        "content": """
            **Automatic parameter screening (Dakota-style)**

            When the parameter space is large, many parameters may have
            negligible influence on the objectives. foamBO can detect and
            fix these mid-optimization using Sobol sensitivity indices.

            **Library API:**
            ```python
            FoamBO("Exp")
                .parameter("x", bounds=[0, 1])
                .parameter("y", bounds=[0, 1])
                .parameter("z", bounds=[0, 1])
                .minimize("metric", fn=my_fn)
                .reduce(after_trials=15, min_importance=0.05, fix_at="best")
                .run()
            ```

            **YAML config:**
            ```yaml
            orchestration_settings:
              dimensionality_reduction:
                enabled: true
                after_trials: 15
                min_importance: 0.05
                fix_at: best           # or "center"
                max_fix_fraction: 0.5
            ```

            **How it works:**
            1. Optimization runs normally for ``after_trials`` trials
            2. GP model fit is checked via cross-validation — if the model is
               too inaccurate, reduction is deferred until the next poll cycle
            3. Sobol first-order sensitivity indices are computed directly from
               the BoTorch GP model (works for both single and multi-objective)
            4. Parameters with index below ``min_importance`` are fixed at
               their best observed value (or center of bounds)
            5. At most ``max_fix_fraction`` of parameters are fixed, and at
               least one parameter always remains active
            6. Optimization continues in the reduced search space

            **Settings:**
            - ``after_trials``: must be past the SOBOL init phase (BO model needed)
            - ``min_importance``: threshold [0, 1]. 0.05 = drop params < 5%% variance
            - ``fix_at``: ``"best"`` (from best trial) or ``"center"`` (midpoint)
            - ``max_fix_fraction``: safety cap (default 0.5 = at most half)

            **Safety guards:**
            - Model fit is validated before trusting Sobol indices — poor GP
              fit causes the reduction to be deferred, not skipped permanently
            - If sensitivity analysis fails 3 times, reduction is disabled
              for the remainder of the run
            - At least one parameter always remains active

            The reduced state persists across save/load — fixed parameters
            remain fixed on restart.
            """,
    }

    harvested["python.loading_client_state"] = {
        "category": "Python snippet",
        "content": """
            Load a saved experiment for analysis or predictions:
            ```python
            from foambo import FoamBO

            result = FoamBO.load("MyExperiment")
            predictions = result.predict([{"x": 10}])
            cv = result.cross_validate()
            ```

            The underlying Ax Client is available as ``result.client`` for
            advanced use (e.g. ``result.client.get_pareto_frontier()``).

            To browse results in the web dashboard:
            ```bash
            foamBO --no-opt --config MyOpt.yaml ++store.read_from=json
            ```
            """,
    }

    harvested["python.resume_from_library"] = {
        "category": "Python snippet",
        "content": """
            Load a saved experiment for analysis or predictions:

            ```python
            from foambo import FoamBO

            result = FoamBO.load("MyExperiment")
            result.predict([{"x": 10}])
            result.cross_validate()
            ```

            To browse results in the web dashboard:
            ```bash
            foamBO --no-opt --config MyOpt.yaml ++store.read_from=json
            ```

            To continue optimization with more trials:
            ```bash
            foamBO --config MyOpt.yaml \\
                ++store.read_from=json \\
                ++orchestration_settings.max_trials=200
            ```
            """,
    }

    # Phase 4: Load tutorials from docs/*.md
    tutorials = load_tutorial_docs()
    harvested.update(tutorials)

    return harvested
