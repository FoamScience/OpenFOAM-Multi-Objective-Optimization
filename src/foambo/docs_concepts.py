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
Interactive API documentation (Swagger UI) is available at ``/docs`` on the
running server (e.g. ``http://localhost:8098/docs``). The raw OpenAPI spec
is at ``/openapi.json``.""",
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

The Overview tab shows per-node trial counts and the currently active node.""",
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
}
