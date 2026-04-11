"""Robust optimization wiring: context features, risk measures, and round-robin context assignment.

Extracted from optimize.py for maintainability.  The main entry points are:

- ``_param_bounds``: extract (lo, hi) from various parameter config types
- ``_resolve_context_points``: auto-generate context points via Sobol
- ``_wire_robust_optimization``: monkey-patch Ax/BoTorch for context-aware BO
"""

import logging
import numpy as np

log = logging.getLogger("ax.foambo.robustness")


def _param_bounds(p):
    """Extract (lower, upper) from RangeParameterConfig or Ax RangeParameter."""
    if hasattr(p, 'bounds'):
        return float(p.bounds[0]), float(p.bounds[1])
    return float(p.lower), float(p.upper)


def _resolve_context_points(robust_cfg, exp_cfg, log):
    """Auto-generate context_points via Sobol if not explicitly provided."""
    if robust_cfg.context_points is not None:
        return  # already explicit
    from torch.quasirandom import SobolEngine

    context_names = exp_cfg.get_context_param_names(robust_cfg)
    d = len(context_names)
    if d == 0:
        raise ValueError(
            f"No parameters found in context_groups {robust_cfg.context_groups}. "
            "Check that parameter 'groups' tags match context_groups names.")

    # Collect bounds and types for context params
    param_map = {p.name: p for p in exp_cfg.parameters}
    bounds_lo, bounds_hi, is_int = [], [], []
    for cn in context_names:
        p = param_map[cn]
        lo_val, hi_val = _param_bounds(p)
        bounds_lo.append(lo_val)
        bounds_hi.append(hi_val)
        is_int.append(getattr(p, 'parameter_type', 'float') == 'int')
    lo = np.array(bounds_lo, dtype=np.float64)
    hi = np.array(bounds_hi, dtype=np.float64)

    # Generate Sobol samples, over-sample to handle constraint filtering
    n_target = robust_cfg.context_samples
    sobol = SobolEngine(dimension=d, scramble=True)
    # Over-sample 3x to leave room for constraint rejection
    n_raw = n_target * 3
    raw = sobol.draw(n_raw).numpy()  # (n_raw, d) in [0,1]
    scaled = lo + raw * (hi - lo)

    # Round integer params
    for j, is_i in enumerate(is_int):
        if is_i:
            scaled[:, j] = np.round(scaled[:, j])

    # Build dicts
    points = [{cn: float(scaled[i, j]) for j, cn in enumerate(context_names)}
              for i in range(n_raw)]

    # Apply context_constraints filter
    if robust_cfg.context_constraints:
        from foambo.constraints import _parse_inequality
        import sympy
        symbols = [sympy.Symbol(n) for n in context_names]
        compiled = []
        for expr_str in robust_cfg.context_constraints:
            normalized, _ = _parse_inequality(expr_str)
            fn = sympy.lambdify(symbols, normalized, modules=["numpy"])
            compiled.append(fn)
        filtered = []
        for pt in points:
            vals = [pt[cn] for cn in context_names]
            if all(fn(*vals) >= 0 for fn in compiled):
                filtered.append(pt)
        points = filtered

    if len(points) < 2:
        raise ValueError(
            f"Only {len(points)} context points survived constraint filtering "
            f"(need >= 2). Relax context_constraints or increase context_samples.")

    # Take first n_target
    points = points[:n_target]
    robust_cfg.context_points = points
    log.debug("Generated %d context points via Sobol from %s parameter bounds",
             len(points), robust_cfg.context_groups)
    for i, pt in enumerate(points):
        log.debug("  context[%d]: %s", i, pt)


def _wire_robust_optimization(client, exp_cfg, robust_cfg, log):
    """Wire risk measure + SubstituteContextFeatures + fixed_features + round-robin.

    Makes BO context-aware for robust optimization:
    1. Builds normalized feature_set tensor from context_points
    2. Patches Surrogate.fit to attach SubstituteContextFeatures + cache baseline_Y
    3. Patches get_botorch_objective_and_transform to inject risk measure
       - SOO: CVaR
       - MOO + mars/auto: MARS (VaR of Chebyshev scalarizations, Daulton 2022)
       - MOO + cvar: ParEGO-style Chebyshev scalarization + CVaR
    4. Sets up fixed_features for context dims
    5. Patches GenerationStrategy.gen for round-robin context + weight randomization
    """
    import torch
    from botorch.acquisition.risk_measures import CVaR
    from botorch.acquisition.objective import GenericMCObjective
    from ax.adapter.registry import Generators

    context_names = exp_cfg.get_context_param_names(robust_cfg)
    context_indices = exp_cfg.get_context_dim_indices(robust_cfg)
    param_map = {p.name: p for p in exp_cfg.parameters}
    n_w = robust_cfg.n_w
    alpha = robust_cfg.alpha

    # --- Detect SOO vs MOO ---
    from ax.core.objective import MultiObjective
    is_moo = isinstance(
        client._experiment.optimization_config.objective, MultiObjective
    )
    n_objectives = (len(client._experiment.optimization_config.objective.objectives)
                    if is_moo else 1)

    # --- Determine risk measure ---
    use_mars = is_moo and robust_cfg.risk_measure in ("mars", "auto")
    robust_cfg._resolved_risk_measure = "mars" if use_mars else "cvar"

    # --- MARS setup: preprocessing to align objectives for maximization ---
    _align_fn = None
    _signs = None
    if use_mars:
        from botorch.acquisition.multi_objective.multi_output_risk_measures import (
            MARS as _MARS,
        )
        _obj_list = client._experiment.optimization_config.objective.objectives
        _minimize_mask = [obj.minimize for obj in _obj_list]
        _signs = torch.tensor(
            [-1.0 if m else 1.0 for m in _minimize_mask], dtype=torch.double)

        class _AlignMaximize(torch.nn.Module):
            """Negate minimization objectives to align everything for maximization."""
            def forward(self, samples, X=None):
                return samples * _signs.to(samples)

        _align_fn = _AlignMaximize()

    # --- 1. Build normalized feature_set tensor (n_w x d_context) ---
    bounds_lo = np.array([_param_bounds(param_map[cn])[0] for cn in context_names])
    bounds_hi = np.array([_param_bounds(param_map[cn])[1] for cn in context_names])
    raw = np.array([[pt[cn] for cn in context_names]
                    for pt in robust_cfg.context_points])
    normalized = (raw - bounds_lo) / (bounds_hi - bounds_lo + 1e-12)
    feature_set = torch.tensor(normalized, dtype=torch.double)

    _rm_label = "MARS" if use_mars else "CVaR"
    log.info("Robust optimization: %s(alpha=%.2f, n_w=%d), context_params=%s, MOO=%s",
             _rm_label, alpha, n_w, context_names, is_moo)

    # --- MOO state (mutable, closure-captured) ---
    _state = {"baseline_Y": None, "chebyshev_scalarize": None, "_mars_obj": None}

    def _sample_chebyshev_weights():
        """Sample random weights from Dirichlet(1,...,1) = uniform on simplex."""
        return torch.distributions.Dirichlet(
            torch.ones(n_objectives, dtype=torch.double)).sample()

    def _compute_best_f():
        """Compute best scalarized training value for qLogEI (MOO only)."""
        bY = _state.get("baseline_Y")
        if bY is None or bY.shape[0] == 0:
            return 0.0
        if use_mars:
            mars_obj = _state.get("_mars_obj")
            if mars_obj is None:
                return 0.0
            try:
                # chebyshev_objective applies preprocessing (negate) + normalize
                vals = mars_obj.chebyshev_objective(bY, None)
                return vals.max().item()
            except Exception:
                return 0.0
        fn = _state.get("chebyshev_scalarize")
        if fn is None:
            return 0.0
        vals = fn(bY.unsqueeze(0), None)  # (1, n)
        return vals.max().item()

    # --- 2. Patch Surrogate.fit: attach SubstituteContextFeatures + cache baseline_Y ---
    from ax.generators.torch.botorch_modular.surrogate import Surrogate
    _orig_surrogate_fit = Surrogate.fit

    def _fit_with_context_features(self, *args, **kwargs):
        from foambo.constraints import clear_candidate_cache
        clear_candidate_cache()
        result = _orig_surrogate_fit(self, *args, **kwargs)
        if self.model is not None:
            from botorch.models.transforms.input import InputTransform, ChainedInputTransform

            class _SubstituteContextFeatures(InputTransform, torch.nn.Module):
                """Replace context dims with n_w scenarios (keeps same d)."""
                def __init__(self, ctx_indices, ctx_values):
                    super().__init__()
                    self.transform_on_train = False
                    self.transform_on_eval = True
                    self.transform_on_fantasize = True
                    self.register_buffer("ctx_indices", torch.tensor(ctx_indices, dtype=torch.long))
                    self.register_buffer("ctx_values", ctx_values)  # (n_w, d_ctx)
                    self._n_w = ctx_values.shape[0]

                def transform(self, X):
                    # X: (..., q, d) → (..., q*n_w, d)
                    shape = X.shape
                    q, d = shape[-2], shape[-1]
                    batch = shape[:-2]
                    X_exp = X.unsqueeze(-2).expand(*batch, q, self._n_w, d).clone()
                    X_exp[..., self.ctx_indices] = self.ctx_values.to(X_exp)
                    return X_exp.reshape(*batch, q * self._n_w, d)

            def _attach_ctx_tf(model):
                tf = _SubstituteContextFeatures(context_indices, feature_set)
                if hasattr(model, 'input_transform') and model.input_transform is not None:
                    existing = model.input_transform
                    model.input_transform = ChainedInputTransform(
                        tf_existing=existing, tf_ctx=tf)
                else:
                    model.input_transform = tf

            if hasattr(self.model, 'models'):
                for sub_model in self.model.models:
                    _attach_ctx_tf(sub_model)
                log.debug("Attached SubstituteContextFeatures(n_w=%d) to %d sub-models",
                          n_w, len(self.model.models))
            else:
                _attach_ctx_tf(self.model)
                log.debug("Attached SubstituteContextFeatures(n_w=%d) to model", n_w)

            # Cache baseline_Y for MOO Chebyshev scalarization
            if is_moo:
                try:
                    # ModelList: each sub-model has its own train_targets (1 per objective)
                    if hasattr(self.model, 'models'):
                        cols = []
                        for sub_m in self.model.models[:n_objectives]:
                            t = sub_m.train_targets
                            if t is not None:
                                cols.append(t.squeeze(-1) if t.dim() > 1 else t)
                        if cols:
                            Y = torch.stack(cols, dim=-1)
                            _state["baseline_Y"] = Y
                            log.debug("Cached baseline_Y (%d points, %d objectives)",
                                      Y.shape[0], Y.shape[1])
                    elif hasattr(self.model, 'train_targets') and self.model.train_targets is not None:
                        Y = self.model.train_targets
                        if Y.dim() == 1:
                            Y = Y.unsqueeze(-1)
                        _state["baseline_Y"] = Y[:, :n_objectives]
                        log.debug("Cached baseline_Y (%d points, %d objectives)",
                                  Y.shape[0], Y.shape[1])
                except Exception as e:
                    log.warning("Failed to cache baseline_Y: %s", e)
        return result

    Surrogate.fit = _fit_with_context_features

    # --- 3. Patch get_botorch_objective_and_transform for CVaR ---
    import ax.generators.torch.utils as _torch_utils
    _orig_get_obj = _torch_utils.get_botorch_objective_and_transform

    def _get_obj_with_risk(*args, **kwargs):
        objective, posterior_transform = _orig_get_obj(*args, **kwargs)
        if is_moo and use_mars:
            # MARS: VaR of Chebyshev scalarization (Daulton 2022)
            weights = _state.get("_next_weights")
            if weights is None:
                weights = _sample_chebyshev_weights()
            bY = _state["baseline_Y"]
            # baseline_Y for MARS normalization must be in maximization-aligned space
            if bY is not None and bY.shape[0] > 0:
                bY_aligned = (bY * _signs.to(bY))
                # Filter to non-dominated for cleaner normalization bounds
                from botorch.utils.multi_objective.pareto import is_non_dominated
                mask = is_non_dominated(bY_aligned)
                if mask.any():
                    bY_aligned = bY_aligned[mask]
            else:
                bY_aligned = None
            mars_obj = _MARS(
                alpha=alpha, n_w=n_w,
                chebyshev_weights=weights,
                baseline_Y=bY_aligned,
                preprocessing_function=_align_fn,
            )
            _state["_mars_obj"] = mars_obj
            log.debug("MARS: weights=%s, alpha=%.2f, n_w=%d, baseline_Y=%s",
                      weights.tolist(), alpha, n_w,
                      f"{bY_aligned.shape}" if bY_aligned is not None else "None")
            return mars_obj, posterior_transform
        elif is_moo:
            # ParEGO-style: Chebyshev scalarization → CVaR (legacy path)
            from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
            weights = _state.get("_next_weights")
            if weights is None:
                weights = _sample_chebyshev_weights()
            bY = _state["baseline_Y"]
            if bY is None:
                bY = torch.empty(0, n_objectives, dtype=torch.double)
            scalarize = get_chebyshev_scalarization(weights, Y=bY)
            _state["chebyshev_scalarize"] = scalarize
            preprocessing = GenericMCObjective(scalarize)
            log.debug("ParEGO CVaR: weights=%s, baseline_Y=%s",
                      weights.tolist(),
                      f"{bY.shape}" if bY.numel() > 0 else "None")
        else:
            preprocessing = objective
        risk_obj = CVaR(alpha=alpha, n_w=n_w, preprocessing_function=preprocessing)
        log.debug("Injected CVaR(alpha=%.2f, n_w=%d) as MC objective", alpha, n_w)
        return risk_obj, posterior_transform

    _torch_utils.get_botorch_objective_and_transform = _get_obj_with_risk
    import ax.generators.torch.botorch_modular.acquisition as _acq_mod
    _acq_mod.get_botorch_objective_and_transform = _get_obj_with_risk

    # --- 4. Set up fixed_features for context dims ---
    first_pt = robust_cfg.context_points[0]
    initial_ff = {}
    for idx, cn in zip(context_indices, context_names):
        lo, hi = _param_bounds(param_map[cn])
        val = (first_pt[cn] - lo) / (hi - lo + 1e-12)
        initial_ff[idx] = val
    from foambo.constraints import patch_optimize_acqf_robust
    patch_optimize_acqf_robust(initial_ff)

    # --- 5. For MOO: qLogEI + best_f injection (ParEGO scalarizes to single obj) ---
    # SAASBO is included: same ModularBoTorchGenerator pipeline, only the surrogate
    # differs (fully-Bayesian MCMC). Input transforms apply before the MCMC broadcast,
    # baseline_Y caching works via the 1D train_targets path, and the qLogEI input
    # constructor patch is keyed on the acqf class so it applies uniformly.
    # SAAS_MTGP is deliberately excluded: its task-feature dim collides with the
    # context-feature indexing used by fixed_features cycling.
    _BO_GENERATORS = {
        Generators.BOTORCH_MODULAR,
        Generators.BO_MIXED,
        Generators.SAASBO,
    }
    if is_moo:
        for node in client._generation_strategy._nodes:
            for spec in node.generator_specs:
                if (spec.generator_enum not in _BO_GENERATORS
                        and spec.generator_enum not in (
                            Generators.SOBOL, Generators.UNIFORM, Generators.FACTORIAL)):
                    log.warning(
                        "Robust MOO is not compatible with generator '%s'. "
                        "Only BOTORCH_MODULAR, BO_MIXED, and SAASBO are supported.",
                        spec.generator_enum.name)

        # qLogNoisyExpectedImprovement concatenates baseline+candidate then splits
        # back — incompatible with one-to-many transforms. Use qLogEI instead.
        from botorch.acquisition.logei import qLogExpectedImprovement
        for node in client._generation_strategy._nodes:
            for spec in node.generator_specs:
                if spec.generator_enum in _BO_GENERATORS:
                    spec.generator_kwargs["botorch_acqf_class"] = qLogExpectedImprovement

        # Patch input constructor: get_best_f_mc fails with non-block designs
        # (ModelListGP). Inject best_f from Chebyshev-scalarized training data.
        from botorch.acquisition.input_constructors import (
            construct_inputs_qLogEI as _orig_construct_qlei,
            ACQF_INPUT_CONSTRUCTOR_REGISTRY,
        )

        import inspect as _inspect
        _qlei_params = set(_inspect.signature(_orig_construct_qlei).parameters)

        def _construct_inputs_qLogEI_with_best_f(*args, **kwargs):
            # Ax passes MOO-specific kwargs that construct_inputs_qLogEI rejects;
            # filter to only accepted parameters
            kwargs = {k: v for k, v in kwargs.items() if k in _qlei_params}
            if kwargs.get("best_f") is None:
                kwargs["best_f"] = _compute_best_f()
                log.debug("qLogEI: best_f=%.4f", kwargs["best_f"])
            return _orig_construct_qlei(*args, **kwargs)

        ACQF_INPUT_CONSTRUCTOR_REGISTRY[qLogExpectedImprovement] = _construct_inputs_qLogEI_with_best_f
        log.debug("MOO: qLogExpectedImprovement + %s", _rm_label)

    # --- 6. Patch GenerationStrategy.gen for round-robin context ---
    from foambo.constraints import set_fixed_features

    _ctx_idx = [0]
    _orig_gs_gen = client._generation_strategy.gen
    ctx_dim_map = dict(zip(context_indices, context_names))

    def _gen_with_context(*args, **kwargs):
        gs = client._generation_strategy
        is_bo = any(
            s.generator_enum in _BO_GENERATORS
            for s in gs.current_node.generator_specs
        )
        if not is_bo:
            return _orig_gs_gen(*args, **kwargs)

        # Fix context dims so optimizer doesn't search them
        first_pt = robust_cfg.context_points[0]
        ff = {}
        for dim_idx, cn in ctx_dim_map.items():
            lo, hi = _param_bounds(param_map[cn])
            ff[dim_idx] = (first_pt[cn] - lo) / (hi - lo + 1e-12)
        set_fixed_features(ff)

        # Pre-sample Chebyshev weights for this gen call (MOO ParEGO)
        if is_moo:
            _state["_next_weights"] = _sample_chebyshev_weights()
            log.debug("Chebyshev weights: %s", _state["_next_weights"].tolist())

        import time as _t
        _gen_t0 = _t.perf_counter()
        result = _orig_gs_gen(*args, **kwargs)
        _gen_elapsed = _t.perf_counter() - _gen_t0

        # Assign different context points per trial (round-robin)
        for trial_grs in result:
            idx = _ctx_idx[0] % n_w
            ctx_pt = robust_cfg.context_points[idx]
            _ctx_idx[0] += 1
            for gr in trial_grs:
                for arm in gr.arms:
                    for cn in context_names:
                        arm.parameters[cn] = ctx_pt[cn]
            ctx_desc = ", ".join(f"{cn}={ctx_pt[cn]:.4g}" for cn in context_names)
            log.debug("Trial assigned context #%d: %s", idx, ctx_desc)

        log.debug("Generated %d candidate(s) in %.1fs", len(result), _gen_elapsed)
        return result

    client._generation_strategy.gen = _gen_with_context

    # --- 7. Store robust metadata on runner ---
    runner = client._experiment.runner
    runner._robust_cfg = robust_cfg
    runner._context_param_names = context_names
