"""Robust optimization: context features, risk measures, and generator augmentation.

Clean reimplementation — no monkey-patching.  The main entry points are:

- ``RobustOptimizationConfig``: Pydantic config for robust optimization
- ``SubstituteContextFeatures``: BoTorch input transform (ModelListGP-compatible)
- ``RobustAcquisition``: Ax Acquisition subclass injecting risk measures
- ``resolve_context_points``: auto-generate context points via Sobol
- ``augment_generator_specs``: wire robust config into user's generation strategy

Risk measures:
- SOO: CVaR (Conditional Value at Risk) — wraps the scalar objective.
- MOO: MARS (MVaR Approximated by Random Scalarizations) — scalarizes via
  random Chebyshev weights, then applies VaR.  Uses qLogNEI instead of qLogNEHVI.
  Over many iterations, random weights explore the Pareto front.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field
from torch import Tensor

from botorch.acquisition.risk_measures import CVaR
from botorch.models.transforms.input import InputTransform

from ax.generators.torch.botorch_modular.acquisition import Acquisition

log = logging.getLogger("ax.foambo.robustness")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _foambo_base():
    """Lazy import to avoid circular dependency."""
    from foambo.orchestrate import FoamBOBaseModel
    return FoamBOBaseModel


def RobustOptimizationConfig_factory():
    """Build the config class lazily to avoid import-time circular refs."""
    Base = _foambo_base()

    class _RobustOptimizationConfig(Base):
        """Robust optimization across environmental/context variables."""
        context_groups: list[str] = Field(
            description="Parameter group names treated as context (environmental) variables")
        risk_measure: Literal["auto", "cvar", "mars"] = Field(
            default="auto",
            description="Risk measure. 'auto' picks MARS for MOO, CVaR for SOO")
        robustness: float = Field(
            default=0.5, ge=0.0, le=1.0,
            description=(
                "0=risk-neutral, 1=most conservative.  Maps directly to "
                "CVaR/MARS alpha (clamped >= 0.05).  Higher alpha optimises over "
                "the worst (1-alpha) fraction of context scenarios."))
        context_points: list[dict[str, float]] | None = Field(
            default=None,
            description="Explicit context scenarios. If None, auto-generated via Sobol")
        context_samples: int = Field(
            default=10, ge=2,
            description="Number of Sobol samples when auto-generating context points")
        context_constraints: list[str] = Field(
            default=[],
            description="Inequality filters on context points (e.g. 'flowRate >= 0.01')")

    return _RobustOptimizationConfig


# Module-level class (created once)
RobustOptimizationConfig = RobustOptimizationConfig_factory()


# ---------------------------------------------------------------------------
# SubstituteContextFeatures — ModelListGP-compatible input transform
# ---------------------------------------------------------------------------

class SubstituteContextFeatures(InputTransform, torch.nn.Module):
    """Replace context dims with n_w scenarios.  Keeps tensor dim d unchanged.

    Unlike ``AppendFeatures`` (which changes d and breaks ``ModelListGP``),
    this replaces the context columns in-place, replicating each input
    ``q`` times ``n_w`` to produce ``(q * n_w, d)`` outputs.

    Args:
        context_indices: indices of context dimensions in the feature tensor.
        feature_set: ``(n_w, d_ctx)`` tensor of context scenarios (normalized).
    """

    is_one_to_many: bool = True

    def __init__(
        self,
        context_indices: list[int],
        feature_set: Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.transform_on_train = False
        self.transform_on_eval = True
        self.transform_on_fantasize = True
        self.register_buffer("ctx_indices", torch.tensor(context_indices, dtype=torch.long))
        self.register_buffer("feature_set", feature_set)  # (n_w, d_ctx)
        self._n_w = feature_set.shape[0]

    @property
    def n_w(self) -> int:
        return self._n_w

    def transform(self, X: Tensor) -> Tensor:
        # X: (..., q, d) → (..., q * n_w, d)
        shape = X.shape
        q, d = shape[-2], shape[-1]
        batch = shape[:-2]
        X_exp = X.unsqueeze(-2).expand(*batch, q, self._n_w, d).clone()
        for i, idx in enumerate(self.ctx_indices):
            X_exp[..., idx] = self.feature_set[:, i]
        return X_exp.reshape(*batch, q * self._n_w, d)

    def untransform(self, X: Tensor) -> Tensor:
        raise NotImplementedError("SubstituteContextFeatures is not reversible")


# Register with Ax's input_transform_argparse dispatcher so ModelConfig
# can construct this transform via input_transform_classes / input_transform_options.
try:
    from ax.generators.torch.botorch_modular.input_constructors.input_transforms import (
        input_transform_argparse,
    )

    @input_transform_argparse.register(SubstituteContextFeatures)
    def _argparse_substitute_context(
        input_transform_class,
        dataset=None,
        search_space_digest=None,
        input_transform_options=None,
        **kwargs,
    ):
        return input_transform_options or {}

except ImportError:
    pass  # Ax not installed — skip registration


# ---------------------------------------------------------------------------
# RobustAcquisition — injects CVaR (SOO) or MARS (MOO)
# ---------------------------------------------------------------------------

class RobustAcquisition(Acquisition):
    """Acquisition subclass that wraps the objective in a risk measure.

    SOO: CVaR wraps the scalar objective → works with qLogNEI.
    MOO: MARS (random Chebyshev scalarization + VaR) → swaps acqf to qLogNEI.
         New random weights each generation explore the Pareto front.

    Instantiated by passing ``acquisition_class=RobustAcquisition`` and
    ``acquisition_options={"risk_config": {...}}`` in GeneratorSpec kwargs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Extract risk_config before super().__init__ because it calls
        # _instantiate_acquisition → get_botorch_objective_and_transform
        # which needs _risk_config already set.
        # Ax passes acquisition_options as the `options` kwarg.
        opts = kwargs.get("options") or {}
        self._risk_config: dict[str, Any] = opts.pop("risk_config", {})
        super().__init__(*args, **kwargs)

    def _construct_botorch_acquisition(self, botorch_acqf_class, botorch_acqf_options, model):
        """Swap MOO acqf → SOO acqf when using MARS.

        MARS scalarizes multi-objective via Chebyshev weights → scalar output.
        The MOO acqf (qLogNEHVI) expects m-dimensional output, so we swap to
        qLogNEI.
        """
        if self._risk_config.get("is_moo", False):
            from botorch.acquisition.logei import qLogNoisyExpectedImprovement
            from botorch.acquisition.multi_objective.logei import (
                qLogNoisyExpectedHypervolumeImprovement,
            )
            if issubclass(botorch_acqf_class, qLogNoisyExpectedHypervolumeImprovement):
                botorch_acqf_class = qLogNoisyExpectedImprovement
                log.debug("MARS: swapped acqf to qLogNoisyExpectedImprovement")
        return super()._construct_botorch_acquisition(
            botorch_acqf_class=botorch_acqf_class,
            botorch_acqf_options=botorch_acqf_options,
            model=model,
        )

    def get_botorch_objective_and_transform(self, botorch_acqf_class, model,
                                            objective_weights, outcome_constraints=None,
                                            X_observed=None,
                                            learned_objective_preference_model=None):
        is_moo = self._risk_config.get("is_moo", False)

        # For MOO, tell Ax to build a SOO objective so MARS can wrap it
        if is_moo:
            from botorch.acquisition.logei import qLogNoisyExpectedImprovement
            botorch_acqf_class = qLogNoisyExpectedImprovement

        from ax.generators.torch.utils import get_botorch_objective_and_transform as _get_obj
        objective, posterior_transform = _get_obj(
            botorch_acqf_class=botorch_acqf_class,
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
            learned_objective_preference_model=learned_objective_preference_model,
        )

        alpha = self._risk_config.get("alpha", 0.5)
        n_w = self._risk_config.get("n_w", 10)

        if is_moo:
            return self._build_mars(
                model=model,
                objective_weights=objective_weights,
                X_observed=X_observed,
                alpha=alpha,
                n_w=n_w,
            )
        else:
            preprocessing = objective if objective is not None else None
            risk_obj = CVaR(alpha=alpha, n_w=n_w, preprocessing_function=preprocessing)
            log.debug("Injected CVaR(alpha=%.2f, n_w=%d)", alpha, n_w)
            return risk_obj, posterior_transform

    def _build_mars(self, model, objective_weights, X_observed, alpha, n_w):
        """Build MARS objective following BoTorch tutorial pattern.

        1. Sample random Chebyshev weights on the simplex
        2. Create MARS with ref_point
        3. Call set_baseline_Y to compute normalization from training data
        """
        from botorch.acquisition.multi_objective.multi_output_risk_measures import (
            MARS,
        )
        from botorch.utils.sampling import sample_simplex

        # Number of objectives from objective_weights
        flat_w = (objective_weights.sum(dim=0) if objective_weights.dim() > 1
                  else objective_weights)
        n_obj = (flat_w != 0).sum().item()

        # Random Chebyshev weights — each generation explores a different
        # trade-off direction on the Pareto front
        weights = sample_simplex(
            d=n_obj, n=1,
            dtype=torch.double,
        ).squeeze(0)

        # Reference point: zeros (in normalized space)
        ref_point = torch.zeros(n_obj, dtype=torch.double)

        # Alignment: flip signs for minimized objectives so MARS sees
        # all-maximize space
        nonzero_mask = flat_w != 0
        signs = torch.sign(flat_w[nonzero_mask])

        def _align_fn(samples: Tensor, X=None) -> Tensor:
            # Select only the objective outputs and align to maximization
            if samples.shape[-1] > n_obj:
                samples = samples[..., :n_obj]
            return samples * signs.to(samples)

        mars = MARS(
            alpha=alpha,
            n_w=n_w,
            chebyshev_weights=weights,
            ref_point=ref_point,
            preprocessing_function=_align_fn,
        )

        # Set normalization bounds from training data (tutorial pattern)
        if X_observed is not None:
            try:
                mars.set_baseline_Y(model=model, X_baseline=X_observed)
            except Exception as e:
                log.warning("Failed to set MARS baseline_Y: %s", e)

        log.debug("MARS: weights=%s, alpha=%.2f, n_w=%d, n_obj=%d",
                  weights.tolist(), alpha, n_w, n_obj)

        # No posterior_transform — MARS handles everything
        return mars, None


# ---------------------------------------------------------------------------
# Context point resolution
# ---------------------------------------------------------------------------

def _param_bounds(p) -> tuple[float, float]:
    """Extract (lower, upper) from RangeParameterConfig or Ax RangeParameter."""
    if hasattr(p, 'bounds'):
        return float(p.bounds[0]), float(p.bounds[1])
    return float(p.lower), float(p.upper)


def resolve_context_points(robust_cfg, exp_cfg) -> None:
    """Auto-generate context_points via Sobol if not explicitly provided.

    Modifies ``robust_cfg.context_points`` in place.
    """
    if robust_cfg.context_points is not None:
        return  # already explicit

    from torch.quasirandom import SobolEngine

    context_names = _get_context_param_names(robust_cfg, exp_cfg)
    d = len(context_names)
    if d == 0:
        raise ValueError(
            f"No parameters found in context_groups {robust_cfg.context_groups}. "
            "Check that parameter 'groups' tags match context_groups names.")

    param_map = {p.name: p for p in exp_cfg.parameters}
    bounds_lo, bounds_hi, is_int = [], [], []
    for cn in context_names:
        p = param_map[cn]
        lo, hi = _param_bounds(p)
        bounds_lo.append(lo)
        bounds_hi.append(hi)
        is_int.append(getattr(p, 'parameter_type', 'float') == 'int')

    n_raw = max(robust_cfg.context_samples * 3, 20)  # oversample for filtering
    engine = SobolEngine(dimension=d, scramble=True)
    raw = engine.draw(n_raw).numpy()

    bounds_lo = np.array(bounds_lo)
    bounds_hi = np.array(bounds_hi)
    scaled = bounds_lo + raw * (bounds_hi - bounds_lo)

    for j, is_i in enumerate(is_int):
        if is_i:
            scaled[:, j] = np.round(scaled[:, j])

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

    n_target = robust_cfg.context_samples
    points = points[:n_target]
    robust_cfg.context_points = points
    log.debug("Generated %d context points via Sobol from %s bounds",
              len(points), robust_cfg.context_groups)
    for i, pt in enumerate(points):
        log.debug("  context[%d]: %s", i, pt)


# ---------------------------------------------------------------------------
# Helper: context param names and indices
# ---------------------------------------------------------------------------

def _get_context_param_names(robust_cfg, exp_cfg) -> list[str]:
    """Ordered list of parameter names belonging to context_groups."""
    names = []
    for p in exp_cfg.parameters:
        p_groups = exp_cfg.get_parameter_groups().get(p.name, [])
        if any(g in robust_cfg.context_groups for g in p_groups):
            names.append(p.name)
    return names


def _get_context_dim_indices(robust_cfg, exp_cfg) -> list[int]:
    """Integer indices of context parameters in the feature tensor."""
    all_names = [p.name for p in exp_cfg.parameters]
    ctx_names = _get_context_param_names(robust_cfg, exp_cfg)
    return [all_names.index(n) for n in ctx_names]


# ---------------------------------------------------------------------------
# Generator augmentation — the main integration point
# ---------------------------------------------------------------------------

def build_context_tensor(robust_cfg, exp_cfg) -> Tensor:
    """Build normalized (n_w, d_ctx) tensor from context_points."""
    ctx_names = _get_context_param_names(robust_cfg, exp_cfg)
    param_map = {p.name: p for p in exp_cfg.parameters}

    bounds_lo, bounds_hi = [], []
    for cn in ctx_names:
        lo, hi = _param_bounds(param_map[cn])
        bounds_lo.append(lo)
        bounds_hi.append(hi)

    bounds_lo = np.array(bounds_lo)
    bounds_hi = np.array(bounds_hi)

    raw = np.array([[pt[cn] for cn in ctx_names] for pt in robust_cfg.context_points])
    normalized = (raw - bounds_lo) / (bounds_hi - bounds_lo + 1e-12)
    return torch.tensor(normalized, dtype=torch.double)


def augment_generator_specs(client, exp_cfg, robust_cfg) -> dict[str, Any]:
    """Augment BO GeneratorSpecs in the client's generation strategy for robust opt.

    Non-destructive: preserves user's node structure, transitions, and model config.
    Only modifies BO generator specs to add:
    - SubstituteContextFeatures input transform
    - RobustAcquisition class (CVaR for SOO, MARS for MOO)
    - Fixed features for context dimensions

    Returns a state dict for use by the optimization loop (context cycling etc).
    """
    from ax.adapter.registry import Generators
    from ax.generators.torch.botorch_modular.surrogate import (
        SurrogateSpec, ModelConfig,
    )
    from botorch.models.transforms.input import Normalize

    # Resolve context points if needed
    resolve_context_points(robust_cfg, exp_cfg)

    ctx_names = _get_context_param_names(robust_cfg, exp_cfg)
    ctx_indices = _get_context_dim_indices(robust_cfg, exp_cfg)
    feature_set = build_context_tensor(robust_cfg, exp_cfg)
    n_w = feature_set.shape[0]

    # BoTorch convention: higher alpha = more conservative (optimises over
    # worst 1-alpha fraction).  User's `robustness` maps directly to alpha.
    alpha = max(robust_cfg.robustness, 0.05)

    # Detect MOO
    from ax.core.objective import MultiObjective
    is_moo = isinstance(
        client._experiment.optimization_config.objective, MultiObjective
    )

    risk_config = {
        "alpha": alpha,
        "n_w": n_w,
        "is_moo": is_moo,
        "context_names": ctx_names,
        "context_indices": ctx_indices,
    }

    # Build SurrogateSpec with SubstituteContextFeatures
    surrogate_spec = SurrogateSpec(
        model_configs=[ModelConfig(
            input_transform_classes=[Normalize, SubstituteContextFeatures],
            input_transform_options={
                "SubstituteContextFeatures": {
                    "context_indices": ctx_indices,
                    "feature_set": feature_set,
                },
            },
        )],
    )

    # Fixed features: pin context dims to first context point (normalized)
    from ax.core.observation import ObservationFeatures
    first_ctx_normalized = {
        ctx_names[i]: float(feature_set[0, i])
        for i in range(len(ctx_names))
    }

    _BO_GENERATORS = {
        Generators.BOTORCH_MODULAR,
        Generators.BO_MIXED,
        Generators.SAASBO,
    }

    augmented = 0
    for node in client._generation_strategy._nodes:
        for spec in node.generator_specs:
            if spec.generator_enum in _BO_GENERATORS:
                spec.generator_kwargs["surrogate_spec"] = surrogate_spec
                spec.generator_kwargs["acquisition_class"] = RobustAcquisition
                spec.generator_kwargs["acquisition_options"] = {
                    "risk_config": risk_config,
                }
                spec.fixed_features = ObservationFeatures(
                    parameters=first_ctx_normalized
                )
                augmented += 1

    rm_label = "MARS" if is_moo else "CVaR"
    log.info("Robust optimization: %s(alpha=%.2f, n_w=%d), context_params=%s, "
             "augmented %d generator specs",
             rm_label, alpha, n_w, ctx_names, augmented)

    return {
        "robust_cfg": robust_cfg,
        "ctx_names": ctx_names,
        "ctx_indices": ctx_indices,
        "feature_set": feature_set,
        "n_w": n_w,
        "risk_config": risk_config,
        "is_moo": is_moo,
    }


def cycle_context(client, robust_state: dict, trial_index: int) -> None:
    """Cycle fixed_features to the next context point (round-robin).

    Called from the optimization loop before each gen() to rotate
    through context scenarios.
    """
    from ax.adapter.registry import Generators
    from ax.core.observation import ObservationFeatures

    feature_set = robust_state["feature_set"]
    ctx_names = robust_state["ctx_names"]
    n_w = robust_state["n_w"]

    ctx_idx = trial_index % n_w
    ctx_values = {
        ctx_names[i]: float(feature_set[ctx_idx, i])
        for i in range(len(ctx_names))
    }

    _BO_GENERATORS = {
        Generators.BOTORCH_MODULAR,
        Generators.BO_MIXED,
        Generators.SAASBO,
    }

    for node in client._generation_strategy._nodes:
        for spec in node.generator_specs:
            if spec.generator_enum in _BO_GENERATORS:
                spec.fixed_features = ObservationFeatures(parameters=ctx_values)

    log.debug("Context cycle: trial=%d → context[%d]=%s", trial_index, ctx_idx, ctx_values)


# ---------------------------------------------------------------------------
# Ax JSON serialization registry — must run at import time so that
# loading saved experiments (--no-opt, analysis, etc.) can deserialize
# SubstituteContextFeatures and RobustAcquisition.
# ---------------------------------------------------------------------------
try:
    from ax.storage.botorch_modular_registry import (
        CLASS_TO_REGISTRY, CLASS_TO_REVERSE_REGISTRY,
        register_acquisition,
    )
    from botorch.models.transforms.input import InputTransform as _InputTransform

    CLASS_TO_REGISTRY[_InputTransform][SubstituteContextFeatures] = "SubstituteContextFeatures"
    CLASS_TO_REVERSE_REGISTRY[_InputTransform]["SubstituteContextFeatures"] = SubstituteContextFeatures
    register_acquisition(RobustAcquisition)
except ImportError:
    pass
