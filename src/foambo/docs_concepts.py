"""Concept documentation for foamBO --docs.

These are reference/educational entries, not config field docs.
Each entry is {key: {"category": "Concept", "content": str}}.
"""

CONCEPTS = {
    "concept.sensitivity_indices": {
        "category": "Concept",
        "content": """\
Understanding Sobol sensitivity indices in the Analysis tab.

Ax uses a GP (Gaussian Process) surrogate model to estimate how much each
parameter contributes to the objective variance. There are three types:

**First-order index** (e.g. ``angle1: 0.35``):
- "35% of the objective variance is explained by ``angle1`` alone"
- Measures the direct effect of changing that parameter while averaging over all others

**Second-order index** (e.g. ``angle1 & angle2: 0.12``):
- "12% of the objective variance comes from the *interaction* between ``angle1`` and ``angle2``"
- This is variance that can't be attributed to either parameter alone — it only appears when both change together
- Example: ``angle1=30`` is great when ``angle2=25`` but terrible when ``angle2=40``. Neither parameter's individual effect captures this — it's a joint effect

**Total-order index** (e.g. ``angle1: 0.52``):
- First-order + all interactions involving ``angle1``
- "52% of variance involves ``angle1`` in some way"
- Always >= first-order. The gap tells you how much this parameter interacts with others

**Practical interpretation:**
- High first-order, low interactions → parameter has a clean, independent effect. Easy to optimize.
- Low first-order, high total-order → parameter mostly matters through interactions. Hard to optimize in isolation.
- ``param1 & param2`` being large → these two parameters should be tuned together, not independently.

**Parameter groups** amplify this analysis. When parameters have ``groups`` tags
(e.g. ``groups: ["geometry"]``), foamBO can aggregate sensitivity by group —
showing "geometry vs solver" importance rather than 15 individual parameter bars.""",
    },

    "concept.parameter_groups": {
        "category": "Concept",
        "content": """\
Parameter groups allow tagging parameters by domain (e.g. geometry, solver, mesh).

**Definition** (in experiment parameters):
```yaml
experiment:
  parameters:
    - name: angle1_hub
      bounds: [20, 40]
      groups: ["geometry"]
    - name: meshResolution
      bounds: [10, 50]
      groups: ["geometry", "mesh"]
    - name: relaxationFactor
      bounds: [0.1, 0.9]
      # no group
```

**Uses:**
- **Group sensitivity analysis**: aggregate Sobol indices by group instead of individual parameters
- **Group interaction heatmap**: see which groups interact strongly (should be optimized jointly)
- **Group-conditional best**: "given this geometry, what's the best solver config?"
- **What-if freezing**: lock a group's params in the Predict tab, explore others
- **matching_group dependency**: reuse mesh when geometry params haven't changed

Parameters can belong to multiple groups. The ``groups`` field is stripped before
passing parameters to Ax — it's a foamBO-only annotation.""",
    },

    "concept.trial_dependencies": {
        "category": "Concept",
        "content": """\
Trial dependencies let new trials reuse work from previous ones.

**Strategies** for picking the source trial:
- ``best``: trial with the best objective value
- ``nearest``: closest in parameter space (normalized L2 distance)
- ``latest``: most recently completed trial
- ``baseline``: the baseline trial (index 0)
- ``matching_group``: latest trial where a parameter group's values match exactly
- ``by_index``: specific trial number
- ``custom``: external command that prints a trial index

**Phases** control when actions execute:
- ``immediate``: before the runner starts (default)
- ``pre_init`` / ``pre_mesh`` / ``pre_solve`` / ``post_solve``: deferred to hook scripts

The runner exposes hook scripts via environment variables (``$FOAMBO_PRE_MESH``, etc.)
that the Allrun script calls at the right moment.""",
    },

    "concept.early_stopping": {
        "category": "Concept",
        "content": """\
Early stopping terminates underperforming trials before they complete.

**Strategies:**
- ``threshold``: stop if a streaming metric exceeds a fixed value
- ``percentile``: stop if a trial's metric is worse than the Nth percentile of all trials at that step

**Composite strategies** combine multiple rules:
- ``or``: stop if ANY sub-strategy triggers
- ``and``: stop only if ALL sub-strategies trigger

**Requirements:**
- Metrics must stream intermediate values (``progress`` field in metric config)
- ``FoamJobMetric.is_available_while_running()`` must return True
- ``min_progression``: minimum steps before early stopping can trigger
- ``min_curves``: minimum number of trials before percentile is meaningful

The dashboard's Streaming tab shows threshold/percentile lines overlaid on metric curves.""",
    },

    "concept.dimensionality_reduction": {
        "category": "Concept",
        "content": """\
Automatic dimensionality reduction fixes unimportant parameters to shrink the search space.

**How it works:**
1. After ``after_trials`` completed trials, compute total-order Sobol sensitivity indices
2. Parameters with total importance below ``min_importance`` are candidates for fixing
3. Fixed value is chosen by ``fix_at`` strategy (``best`` = value from best trial)
4. At most ``max_fix_fraction`` of parameters can be fixed (at least one stays free)

**Total-order** indices are used (not first-order) because a parameter that only matters
through interactions with others should not be fixed — fixing it would break those interactions.

The reduction persists across save/load. Fixed parameters appear as ``FixedParameter`` in Ax.""",
    },

    "concept.dashboard": {
        "category": "Concept",
        "content": """\
The web dashboard provides real-time monitoring and post-optimization analysis.

**During optimization:** starts automatically on port 8098 (``--no-ui`` to disable).
**After optimization:** ``foamBO --no-opt --config MyOpt.yaml ++store.read_from=json``

**Tabs:**
- **Overview**: trial counts, generation strategy progress, budget
- **Trials**: sortable table with inline detail (params, logs, visualization)
- **Objectives**: per-metric progression, Pareto frontier, hypervolume trace
- **Analysis**: on-demand model insights (sensitivity, parallel coords, cross-validation, contour, healthchecks, group analysis)
- **Streaming**: per-trial metric curves with early stopping thresholds
- **Dependencies**: trial dependency tree
- **Predict**: model predictions with uncertainty, what-if group freezing
- **Config**: live config viewer/editor for mutable settings

**Visualization:** upload a pvpython script for custom case rendering.
**Performance:** ETag caching, tab-aware fetching, 3s polling.

**REST API:** The dashboard is powered by an OpenAPI-compatible REST API.
- Swagger UI: ``/api/docs`` (e.g. ``http://localhost:8098/api/docs``)
- OpenAPI spec: ``/api/openapi.json``
- ReDoc: ``/redoc``""",
    },

    "concept.pareto_frontier": {
        "category": "Concept",
        "content": """\
The Pareto frontier is the set of solutions where no objective can be improved
without worsening another. A trial is **Pareto-optimal** if no other trial
dominates it (i.e. is better in all objectives simultaneously).

**Hypervolume** measures the quality of the Pareto frontier — it's the volume of
objective space dominated by the frontier relative to a reference point. Larger
is better. The **hypervolume trace** (in the Objectives tab) plots this over
trials; a plateauing trace signals convergence.

**In foamBO:**
- The Objectives tab shows Pareto scatter plots for all objective pairs
- Pareto-optimal trials are marked as diamonds
- The "Pick Interesting Point" buttons in the Predict tab select Pareto points
- ``use_model_predictions=True`` (default) uses the GP model to estimate the
  frontier, which is smoother but requires a fitted model""",
    },

    "concept.objective_thresholds": {
        "category": "Concept",
        "content": """\
Objective thresholds define the "unacceptable" region in multi-objective optimization.
They set the reference point for hypervolume computation.

**Why they matter:**
- Without thresholds, Ax must infer the reference point (less reliable)
- They focus the optimizer on the region of interest
- Hypervolume is computed relative to these bounds

**Configuration:**
```yaml
optimization:
  objective_thresholds:
    - "pressure >= 100"        # absolute threshold
    - "efficiency >= baseline"  # relative to baseline trial
```

**Baseline-relative thresholds** (e.g. ``baseline + 10%``) are resolved after
the baseline trial completes. This is useful when you don't know the scale of
your objectives upfront.

Thresholds are shown as dashed lines in the Objectives tab charts.""",
    },

    "concept.constraints_vs_thresholds": {
        "category": "Concept",
        "content": """\
Outcome constraints and objective thresholds serve different purposes in MOO.

**Outcome constraints** are hard feasibility boundaries:
- Trials violating them are marked **infeasible**
- Infeasible trials are observed by the model but excluded from the Pareto frontier
- Example: ``"stress <= 500"`` — any trial exceeding 500 MPa stress is rejected
- The feasibility rate tracks what fraction of trials satisfy all constraints

**Objective thresholds** are soft reference bounds:
- They define the reference point for hypervolume computation
- Trials beyond thresholds are still valid, just in an "uninteresting" region
- Example: ``"efficiency >= 0.8"`` — we only care about the frontier above 80%

**Rule of thumb:**
- Use **constraints** for physical limits (material failure, divergence, cost caps)
- Use **thresholds** for "we want at least this good" preferences""",
    },

    "concept.acquisition_functions": {
        "category": "Concept",
        "content": """\
In multi-objective BO, the acquisition function decides which trial to run next.

**Expected Hypervolume Improvement (EHVI)** — what Ax/foamBO uses:
- Estimates how much a candidate point would increase the Pareto hypervolume
- Naturally balances exploration (uncertain regions) and exploitation (near frontier)
- Explores the frontier uniformly — doesn't favor one objective over another
- Accounts for model uncertainty via the GP posterior

**Comparison with MOGA (Multi-Objective Genetic Algorithm):**
- MOGA uses a population of solutions evolved via crossover/mutation
- MOGA is gradient-free and parallelizes trivially but needs many evaluations (100s–1000s)
- EHVI is **sample-efficient** — finds good frontiers in 20–50 evaluations
- EHVI uses a surrogate model, so each trial is informed by all previous data
- MOGA treats each evaluation independently (no model, no learning between generations)
- For expensive simulations (CFD, FEA), EHVI is vastly more practical

**When MOGA is better:** cheap evaluations, highly discrete/combinatorial spaces,
or when the objective landscape is so noisy that a GP can't fit it.""",
    },

    "concept.cross_validation": {
        "category": "Concept",
        "content": """\
Cross-validation assesses how well the GP surrogate model predicts unseen trials.

**How it works in Ax:**
1. Leave one trial out (or use k-fold)
2. Fit the GP on the remaining trials
3. Predict the held-out trial
4. Compare predicted vs observed values

**The CV plot** (in the Analysis tab) shows observed values on x-axis,
predicted values on y-axis. Points near the diagonal = good model fit.

**Interpreting results:**
- Tight cluster along diagonal → model is reliable, trust its predictions
- Wide scatter → model is uncertain, more data needed before trusting analysis
- Systematic bias (all above/below diagonal) → model is mis-specified

**In foamBO:**
- Dimensionality reduction gates on CV quality (won't fix params if fit is poor)
- The Predict tab's uncertainty bars come from the GP posterior
- Poor CV means sensitivity analysis and contour plots may be misleading""",
    },

    "concept.generation_strategy": {
        "category": "Concept",
        "content": """\
The generation strategy controls how new trial parameterizations are chosen.

**Phases (typical ``method: fast`` flow):**
1. **CenterOfSearchSpace** (1 trial): first trial at the midpoint of all parameter ranges
2. **Sobol** (quasi-random, N trials): space-filling initialization to give the GP data
3. **MBM** (model-based, remaining trials): Bayesian Optimization using the fitted GP

**Why initialization matters:**
- The GP needs diverse data before it can make useful predictions
- Sobol sequences fill the space more uniformly than random sampling
- Too few init trials → poor model fit → bad BO suggestions
- ``initialization_budget`` controls how many Sobol trials run before BO starts

**User-provided baseline:**
- If ``baseline.parameters`` is configured, a baseline trial runs *before* the
  generation strategy starts (under a separate ``ManualGenerationNode``)
- The baseline becomes ``status_quo`` — all other trials are compared against it
- Objective thresholds can reference ``baseline`` (e.g. ``baseline + 10%``)
- The baseline trial counts toward ``max_trials`` but not ``initialization_budget``

**Custom strategies (``method: custom``):**
- Define ``generation_nodes`` with explicit generator algorithms and transition criteria
- Supports ``exclude_transforms`` (e.g. skip ``BilogY`` for non-negative objectives)
- Full control over Sobol seeds, transition thresholds, and model configuration

**Seeding from prior data (``SeedDataNode``):**
- Add a node with a ``file_path`` key to seed the experiment with completed
  trials from a CSV or a foamBO JSON state file before BO starts
- Supports ``filter`` (per-parameter value+tolerance, directions: both/left/right)
  and ``drop_params`` (strip params not present in the target search space)
- Enables robust→non-robust workflows: run a robust CVaR study, then continue
  with a non-robust focused run seeded on the Pareto-filtered robust data
- See concept: ``SeedDataNode``

The Overview tab shows per-node trial counts and the currently active node.""",
    },

    "concept.seed_data_node": {
        "category": "Concept",
        "content": """\
``SeedDataNode`` is a generation node that loads pre-existing trial data into
the experiment before optimization starts. It replaces the old ``existing_trials``
top-level section (removed in v1.3).

**Why a node, not a top-level section?**
- Composable in ``trial_generation.generation_nodes`` — visible in the strategy printout
- Follows the same ``next_node`` transition chain as other nodes
- Supports filtering and parameter dropping for cross-experiment workflows
  (e.g. robust study → non-robust focused study on the Pareto front)

**Sources:**
- **CSV** — same format as the legacy ``existing_trials`` loader: columns for
  parameters and metrics, scalar metrics or ``(mean, sem)`` tuples. Rows with
  identical parameter values are grouped as progressions of one trial.
- **foamBO JSON state** — a full saved Client state, loaded via ``FoamBO.load``.
  Completed trials are iterated, filtered, and re-attached to the current client.

**Configuration keys:**
- ``file_path`` (required): path to the CSV or JSON file
- ``next_node`` (required): generation node to transition to after seeding
- ``filter`` (optional): per-parameter filter dict. Scalar form is exact match;
  dict form ``{value, tolerance, direction}`` supports ``both`` (default),
  ``left``, and ``right`` one-sided ranges around the anchor value.
- ``drop_params`` (optional): list of parameter names to remove from imported
  data. The target experiment MUST NOT contain these parameters in its search
  space — otherwise ``attach_trial`` will reject the row as missing a required
  parameter.

**Runtime flow:**
1. During strategy setup, foamBO scans the strategy for ``SeedDataNode`` instances
   and pre-loads their data into the client (attaches completed trials).
2. When the scheduler reaches the node, ``gen()`` emits an empty ``GeneratorRun``
   and ``AutoTransitionAfterGen`` immediately advances to ``next_node``.

**Budget caveat:**
Seeded trials are real Ax trials and count toward experiment-wide ``MaxTrials``
/ ``MinTrials`` transition criteria on downstream nodes. Reduce downstream node
budgets accordingly when seeding. There is no per-origin trial accounting today.

**Example — robust to non-robust handoff:**

```yaml
trial_generation:
  method: custom
  generation_nodes:
    - file_path: "./artifacts/robust_study.json"
      drop_params: ["c"]           # c was a context dim in the robust run
      filter:
        c: {value: 350, tolerance: 5, direction: both}
      next_node: mbm
    - node_name: mbm
      generator_specs:
        - generator_enum: BOTORCH_MODULAR
      transition_criteria: []
```

The non-robust experiment here has a smaller search space (no ``c``), and only
trials from the robust run with ``T`` near 350 K are re-attached before BO
resumes on the reduced space.""",
    },

    "concept.warm_starting_moo": {
        "category": "Concept",
        "content": """\
Warm-starting in multi-objective optimization requires careful source trial selection.

**The challenge:** in SOO, "best trial" is unambiguous. In MOO, the best trial
depends on which objective you prioritize — every Pareto-optimal trial is "best"
in some trade-off.

**Strategy recommendations for MOO:**

``nearest`` — copy fields from the most similar trial in parameter space:
- Best for solver warm-starting (fields from a similar configuration converge faster)
- Use ``similarity_threshold`` to skip if the nearest trial is too far away

``matching_group`` — reuse work when a parameter group hasn't changed:
- Ideal for mesh reuse: if geometry params match, copy the mesh
- Saves the most compute when geometry is expensive but solver settings change often

``best`` — uses the trial with the best *primary* objective:
- In MOO, this is the first objective listed in ``optimization.objective``
- May not represent the best trade-off — consider ``nearest`` instead

``latest`` — most recently completed trial:
- Simple and effective when consecutive trials are similar (small parameter changes)
- Less useful with parallel trials (multiple running simultaneously)

**Combining strategies:**
```yaml
trial_dependencies:
  - name: reuse_mesh
    source: {strategy: matching_group, group: geometry, fallback: skip}
    actions: [{type: run_command, phase: pre_mesh,
               command: "cp -rT $FOAMBO_SOURCE_TRIAL/constant/polyMesh $FOAMBO_TARGET_TRIAL/constant/polyMesh"}]
  - name: warm_fields
    source: {strategy: nearest, similarity_threshold: 0.3, fallback: skip}
    actions: [{type: run_command, phase: pre_solve,
               command: "mapFields $FOAMBO_SOURCE_TRIAL -sourceTime latestTime -case $FOAMBO_TARGET_TRIAL"}]
```
This reuses the mesh when geometry matches AND warm-starts solver fields from
the nearest trial — two independent dependencies working together.""",
    },

    "concept.orchestration": {
        "category": "Concept",
        "content": """\
foamBO uses Ax's native poll-based orchestrator (``ax.api.client.Orchestrator``)
to dispatch, monitor, and harvest trials. The main loop dispatches up to
``parallelism`` trials, then repeatedly sleeps and polls each runner for
completion. Completed trials have their metrics fetched and are handed to the
generation strategy for the next round of proposals.

**Harvest latency** = ``init_seconds_between_polls`` (Ax scheduler option).
For OpenFOAM trials lasting minutes, a small interval (1–5s) adds negligible
overhead while keeping the model responsive.

**Remote runner push integration:**

For local subprocess runners, ``FoamJobRunner.poll_trial`` checks the child
process directly — nothing special is needed. For remote runners (SLURM, SSH),
the local runner has no direct visibility into the remote job state, so trials
notify completion by POSTing to the live REST API. The runner's ``poll_trial``
consumes the push queue at the top of each poll cycle:

1. Trial script POSTs to ``/api/v1/trials/{idx}/push/status`` on completion.
2. The endpoint writes to ``_state.trial_status_overrides[idx]``.
3. On the next Ax poll cycle, ``FoamJobRunner.poll_trial`` pops the override
   and returns ``TrialStatus.COMPLETED`` / ``FAILED`` immediately.
4. Ax fetches metrics and proceeds normally.

**Push endpoints (optional, for remote runners):**
- ``POST /api/v1/trials/{idx}/push/status`` — report completion or failure
- ``POST /api/v1/trials/{idx}/push/metrics`` — push streaming metric values
- ``POST /api/v1/trials/{idx}/push/heartbeat`` — signal the trial is alive
- ``GET  /api/v1/events`` — dashboard-facing event feed (last 500 events)

**Environment variables** injected into every trial subprocess:
- ``FOAMBO_API_ENDPOINT`` — full API URL (e.g. ``http://login01:8098/api/v1``)
- ``FOAMBO_TRIAL_INDEX`` — the trial number
- ``FOAMBO_SESSION_ID`` — unique ID for this foamBO instance

**Allrun integration example (remote runner):**
```bash
#!/bin/bash
blockMesh
$FOAMBO_PRE_MESH
snappyHexMesh
$FOAMBO_PRE_SOLVE
simpleFoam
EXIT_CODE=$?
$FOAMBO_POST_SOLVE

# Notify foamBO of completion
curl -s -X POST "$FOAMBO_API_ENDPOINT/trials/$FOAMBO_TRIAL_INDEX/push/status" \\
  -H "Content-Type: application/json" \\
  -d "{\\\"status\\\":\\\"completed\\\",\\\"exit_code\\\":$EXIT_CODE,\\\"session_id\\\":\\\"$FOAMBO_SESSION_ID\\\"}"
```

**OpenFOAM function object** for streaming metrics (pushed values are attached
to the Ax client via ``_streaming_client.attach_data`` during the next poll
cycle, enabling early stopping on remote trials):
```bash
# Called from controlDict writeInterval, e.g. via a coded function object
curl -s -X POST "$FOAMBO_API_ENDPOINT/trials/$FOAMBO_TRIAL_INDEX/push/metrics" \\
  -H "Content-Type: application/json" \\
  -d "{\\\"metrics\\\":{\\\"continuityErrors\\\":$VALUE},\\\"step\\\":$TIME,\\\"session_id\\\":\\\"$FOAMBO_SESSION_ID\\\"}"
```

**Local mode:** The runner detects subprocess completion automatically —
Allrun curl is optional (useful for sub-phase notifications like "meshing done"
that show up on the dashboard event feed).

**Remote mode (SLURM, SSH):** The Allrun curl is essential — without it, a
remote trial's exit would only be detected via the (slow) SSH-based status
probe. ``api_host`` must be set to an address reachable by all compute nodes
(e.g. the login node hostname). Example: ``api_host: login01.cluster.local``
or ``api_host: 0.0.0.0`` to bind all interfaces.

**Session ID and rogue jobs:** Each foamBO instance has a unique session ID.
Push endpoints validate this ID — if a stale job from a crashed run pushes
to a new foamBO instance, the push is rejected (HTTP 409). The session ID
is available as ``FOAMBO_SESSION_ID`` in the trial environment.

**Event log:** All orchestration events are timestamped and stored:
trial dispatched, streaming metrics, completion, failure, early stopping,
objective values. The log drives the dashboard's live event feed and is
available via ``GET /api/v1/events``.""",
    },

    "concept.robust_optimization": {
        "category": "Concept",
        "content": """\
Robust optimization — designing for performance under context uncertainty.

Instead of optimizing at one nominal operating point, robust mode samples a
distribution of context values (flowRate, temperature, inlet profile, …) at
every candidate design and collapses the resulting outcome distribution via a
risk measure. The optimizer then maximizes the risk-adjusted score, favoring
designs that stay good across contexts rather than designs that peak at one
and tank at others.

**YAML:**
```yaml
robust_optimization:
  context_groups: [operating_point]   # parameter groups treated as contexts
  risk_measure: auto                  # or: cvar, mars
  robustness: 0.8                     # alpha; higher = more conservative
  context_samples: 4                  # samples per candidate
  # context_points: auto-generated via Sobol unless given explicitly
```

**Risk measures:**
- ``cvar`` (Conditional Value-at-Risk) — average of the worst ``(1-alpha)``
  fraction of context outcomes. Single-objective only.
- ``mars`` (MVaR Approximation via Random Scalarizations) — multi-objective
  VaR built from random weight vectors; each scalarization reduces the problem
  to single-objective CVaR, then Ax aggregates. Multi-objective only.
- ``auto`` — picks CVaR when the optimization has one objective and MARS when
  it has multiple.

**Mechanism:** foamBO's ``SubstituteContextFeatures`` input transform swaps
the candidate's context slot for each Sobol-drawn context point at acquisition
evaluation time. The GP is evaluated once per context sample and the risk
measure reduces the ``n_contexts`` values to a single score that
``optimize_acqf`` maximizes.

**Parallelism:** robust trials require ``context_samples`` design evaluations
each. With ``parallelism: N``, each Ax orchestrator slot still corresponds to
one design point (the context sweep runs inside the acquisition, not the
runner), so the runner budget is unchanged.

**Full details, risk-measure math, MOO suitability matrix, and comparisons
to SAASBO / InputPerturbation:** see ``docs/risk-measures-guide.md``.

**Specialization:** once a robust run converges, pin the context with
``bootstrap: + specialize: {ctx: value}`` to continue optimizing design
variables at a concrete operating point. See
``concept.bootstrap_and_specialize``.""",
    },

    "concept.bootstrap_and_specialize": {
        "category": "Concept",
        "content": """\
Reusing a prior experiment: ``bootstrap`` and ``specialize``.

A top-level ``bootstrap: path/to/<name>_client_state.json`` key in a YAML
config tells foamBO to inherit the parent run's full state: search space,
trial history, fitted GP, generation strategy progress, and every field of
the original ``foambo_config``. The current YAML is deep-merged on top via
OmegaConf — any field can be overridden, and setting a block to ``null``
clears it.

**Two workflows this unlocks:**

1. **Continue** under a new experiment name with more trials:
```yaml
bootstrap: ./artifacts/PumpOpt_client_state.json
experiment:
  name: PumpOpt_continued
orchestration_settings:
  n_trials: 60
```

The parent ``_model.pt`` warm-starts the GP (~0.15s) and the orchestrator
resumes at the next free trial index. Everything else stays identical.

2. **Specialize** a robust run to a concrete operating point. Add a
``specialize: {param: value, ...}`` map; each listed parameter becomes an
Ax ``FixedParameter`` in the inherited search space, and every recorded
trial arm is rewritten so that parameter equals the pinned value. The GP
refits on this clamped dataset at the next generation call, and future
candidates vary only the remaining design dimensions.
```yaml
bootstrap: ./artifacts/PumpOpt_client_state.json
experiment:
  name: PumpOpt_spec_op1
robust_optimization: null
specialize:
  flowRate: 0.022
  rpm: 2811
orchestration_settings:
  n_trials: 30
```

**Why rewrite arms instead of dropping non-matching trials?**
Dropping loses the design-variable information that is perfectly valid at
any context. Rewriting keeps every trial as a data point at the pinned
context — the GP loses sensitivity to the specialized parameter (which is
correct: it is no longer a variable) but retains everything it learned
about the rest of the space.

**Boundaries:**
- Parent experiments produced by ``Client.configure_experiment`` carry the
  ``immutable_search_space_and_opt_config`` flag (Ax's default). The bootstrap
  loader clears it automatically so the search space can be mutated for
  specialization. Past generator runs remain valid because the flag only
  meant "don't cache per-trial search-space copies."
- ``specialize`` keys must exist in the parent search space.
- Relative ``bootstrap`` paths resolve against the YAML file's directory.
- Bootstrap does not re-run the seed step from ``SeedDataNode`` or attach
  duplicate trials; it is strictly *continuation*, not *re-seeding*.

**Contrast with ``SeedDataNode``:** ``SeedDataNode`` imports selected trials
from a CSV or JSON into a fresh experiment with a new search space and no
prior GP. ``bootstrap`` keeps the parent experiment and GP wholesale. Use
``SeedDataNode`` when the search space changes shape; use ``bootstrap`` when
it does not (or only changes via ``specialize``).""",
    },

}
