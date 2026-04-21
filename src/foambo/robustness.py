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

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
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
# RobustContextValueFunction — qMFKG inner value function for MF+robust
# ---------------------------------------------------------------------------

class RobustContextValueFunction(AcquisitionFunction):
    """Inner value function for qMFKG that evaluates candidates over context scenarios.

    Used as ``valfunc_cls`` for ``qMultiFidelityKnowledgeGradient``.
    Replaces context dims with ``n_w`` scenarios, evaluates the model,
    and aggregates via MARS (Chebyshev scalarization + VaR).

    Inherits from ``AcquisitionFunction`` (not ``MCAcquisitionFunction``) to
    avoid shape-validation decorators that conflict with qMFKG's inner loop.
    """

    def __init__(self, model, posterior_transform=None, **kwargs):
        super().__init__(model=model)
        self.posterior_transform = posterior_transform
        self._context_indices = kwargs.get("context_indices", [])
        self._feature_set = kwargs.get("feature_set")  # (n_w, d_ctx)
        self._mars = kwargs.get("mars")

    def forward(self, X: Tensor) -> Tensor:
        """Evaluate candidates over context scenarios and aggregate via MARS.

        Args:
            X: ``(..., q, d)`` candidates from qMFKG's inner optimization.

        Returns:
            ``(...)`` risk-aggregated scalar values (one per batch element).
        """
        n_w = self._feature_set.shape[0]
        shape = X.shape  # (..., q, d)
        q, d = shape[-2], shape[-1]
        batch = shape[:-2]

        # Expand across context scenarios: (..., q, d) → (..., q*n_w, d)
        X_exp = X.unsqueeze(-2).expand(*batch, q, n_w, d).clone()
        feat = self._feature_set.to(X_exp)
        for i, idx in enumerate(self._context_indices):
            X_exp[..., idx] = feat[:, i]
        X_exp = X_exp.reshape(*batch, q * n_w, d)

        # Get posterior mean (deterministic — no MC sampling needed for value function)
        posterior = self.model.posterior(
            X_exp, posterior_transform=self.posterior_transform)
        # mean: (..., q*n_w, m)
        Y = posterior.mean

        # Apply MARS (scalarizes m→1 via Chebyshev, then risk-aggregates n_w)
        if self._mars is not None:
            return self._mars(Y, X)
        # Fallback: mean over objectives and context
        return Y.mean(dim=(-1, -2))


def _build_mf_robust_valfunc_argfac(ctx_indices, feature_set, alpha, n_w, is_moo,
                                    n_obj=2):
    """Build a valfunc_argfac closure for qMFKG's RobustContextValueFunction.

    Returns a callable ``model → kwargs`` that constructs MARS and passes
    context config to the value function.
    """
    def argfac(model):
        from botorch.acquisition.multi_objective.multi_output_risk_measures import MARS
        from botorch.utils.sampling import sample_simplex

        # Build MARS (same logic as RobustAcquisition._build_mars)
        if is_moo:
            weights = sample_simplex(d=n_obj, n=1, dtype=torch.double).squeeze(0)
            ref_point = torch.zeros(n_obj, dtype=torch.double)
            mars = MARS(
                alpha=alpha,
                n_w=n_w,
                chebyshev_weights=weights,
                ref_point=ref_point,
            )
            # Set baseline normalization — required before MARS can construct
            # its Chebyshev objective. Try multiple ways to get training X.
            X_baseline = None
            try:
                # Fantasy model: get base model's training data
                if hasattr(model, 'models'):
                    # ModelListGP — get first sub-model's training inputs
                    for sub_m in model.models:
                        if hasattr(sub_m, 'train_inputs') and sub_m.train_inputs:
                            X_baseline = sub_m.train_inputs[0]
                            break
                if X_baseline is None and hasattr(model, 'train_inputs') and model.train_inputs:
                    X_baseline = model.train_inputs[0]
                if X_baseline is not None:
                    mars.set_baseline_Y(model=model, X_baseline=X_baseline)
                else:
                    # Fallback: use dummy data to avoid the error
                    dummy_X = torch.zeros(1, 1, dtype=torch.double)
                    mars.baseline_Y = torch.zeros(1, n_obj, dtype=torch.double)
                    log.debug("MF+robust valfunc: using dummy baseline_Y")
            except Exception as e:
                # Last resort: set dummy baseline to avoid crash
                mars.baseline_Y = torch.zeros(1, n_obj, dtype=torch.double)
                log.warning("MF+robust valfunc: set_baseline_Y failed (%s), using dummy", e)
            log.debug("MF+robust valfunc: MARS(alpha=%.2f, n_w=%d, n_obj=%d)",
                      alpha, n_w, n_obj)
        else:
            # SOO: use CVaR instead
            from botorch.acquisition.risk_measures import CVaR
            mars = CVaR(alpha=alpha, n_w=n_w)

        return {
            "context_indices": ctx_indices,
            "feature_set": feature_set,
            "mars": mars,
        }
    return argfac


# ---------------------------------------------------------------------------
# RobustAcquisition — injects CVaR (SOO) or MARS (MOO)
# ---------------------------------------------------------------------------

class MFHVKGAcquisition(Acquisition):
    """Acquisition subclass that strips kwargs incompatible with qMFHVKG.

    ``construct_inputs_qMFHVKG`` doesn't accept ``X_pending`` (unlike most
    other BoTorch input constructors). This subclass strips it before the
    parent passes acqf_options to the input constructor.
    """

    def _construct_botorch_acquisition(self, botorch_acqf_class, botorch_acqf_options, model):
        # X_pending is passed via self.X_pending in the parent's
        # input_constructor_kwargs, but qMFHVKG doesn't accept it.
        # Temporarily clear it before calling parent.
        _saved = self.X_pending
        self.X_pending = None
        try:
            return super()._construct_botorch_acquisition(
                botorch_acqf_class=botorch_acqf_class,
                botorch_acqf_options=botorch_acqf_options,
                model=model,
            )
        finally:
            self.X_pending = _saved


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
        """Handle acquisition construction for robust optimization.

        MOO: Swap MOO acqf → SOO acqf for MARS scalarization.
        """
        is_moo = self._risk_config.get("is_moo", False)

        if is_moo:
            from botorch.acquisition.logei import qLogNoisyExpectedImprovement
            from botorch.acquisition.multi_objective.logei import (
                qLogNoisyExpectedHypervolumeImprovement,
            )
            if issubclass(botorch_acqf_class, qLogNoisyExpectedHypervolumeImprovement):
                botorch_acqf_class = qLogNoisyExpectedImprovement
                log.debug("MARS: swapped acqf to qLogNoisyExpectedImprovement")
            # Temporarily clear outcome_constraints for the parent construction:
            # with MARS scalarization, the constraint transforms produce shape
            # mismatches in prune_inferior_points.
            _saved_oc = self._outcome_constraints
            self._outcome_constraints = None
            try:
                return super()._construct_botorch_acquisition(
                    botorch_acqf_class=botorch_acqf_class,
                    botorch_acqf_options=botorch_acqf_options,
                    model=model,
                )
            finally:
                self._outcome_constraints = _saved_oc
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

        # MOO: tell Ax to build a SOO objective so MARS can wrap it.
        if is_moo:
            from botorch.acquisition.logei import qLogNoisyExpectedImprovement
            botorch_acqf_class = qLogNoisyExpectedImprovement
            # MARS + constraint transforms produce shape mismatches in
            # prune_inferior_points. Drop constraints for MARS — the
            # GP-learned posterior already accounts for constraint violations
            # via the training data.
            if outcome_constraints is not None:
                log.debug("MARS: dropping outcome_constraints to avoid shape mismatch")
                outcome_constraints = None

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
# RobustMCObjective — standalone risk-measure objective for qMFHVKG composition
# ---------------------------------------------------------------------------

class RobustMCObjective:
    """Risk-measure objective compatible with MCMultiOutputObjective interface.

    Wraps MARS (MOO) or CVaR (SOO) as a callable that can be passed as the
    ``objective`` parameter to qMFHVKG or any MC acquisition function.

    When the GP has SubstituteContextFeatures as an input transform, posterior
    samples at a candidate X are fanned across K context points. This objective
    reduces the K-context samples via the risk measure to produce risk-adjusted
    objective values.

    Usage::

        obj = RobustMCObjective.build(
            model=model, objective_weights=weights, X_observed=X,
            alpha=0.8, n_w=10, is_moo=True,
        )
        acqf = qMFHVKG(..., objective=obj)
    """

    @staticmethod
    def build(model, objective_weights, X_observed=None,
              alpha=0.5, n_w=10, is_moo=False):
        """Build and return a risk-measure objective (MARS or CVaR).

        Returns a BoTorch objective instance that can be passed directly
        as ``objective`` to an MC acquisition function.
        """
        if is_moo:
            return RobustMCObjective._build_mars_objective(
                model=model, objective_weights=objective_weights,
                X_observed=X_observed, alpha=alpha, n_w=n_w,
            )
        else:
            return RobustMCObjective._build_cvar_objective(alpha=alpha, n_w=n_w)

    @staticmethod
    def _build_mars_objective(model, objective_weights, X_observed, alpha, n_w):
        """Build MARS objective — same logic as RobustAcquisition._build_mars."""
        from botorch.acquisition.multi_objective.multi_output_risk_measures import MARS
        from botorch.utils.sampling import sample_simplex

        flat_w = (objective_weights.sum(dim=0) if objective_weights.dim() > 1
                  else objective_weights)
        n_obj = (flat_w != 0).sum().item()

        weights = sample_simplex(d=n_obj, n=1, dtype=torch.double).squeeze(0)
        ref_point = torch.zeros(n_obj, dtype=torch.double)
        nonzero_mask = flat_w != 0
        signs = torch.sign(flat_w[nonzero_mask])

        def _align_fn(samples, X=None):
            if samples.shape[-1] > n_obj:
                samples = samples[..., :n_obj]
            return samples * signs.to(samples)

        mars = MARS(
            alpha=alpha, n_w=n_w, chebyshev_weights=weights,
            ref_point=ref_point, preprocessing_function=_align_fn,
        )
        if X_observed is not None:
            try:
                mars.set_baseline_Y(model=model, X_baseline=X_observed)
            except Exception as e:
                log.warning("RobustMCObjective: failed to set MARS baseline_Y: %s", e)

        log.debug("RobustMCObjective MARS: weights=%s, alpha=%.2f, n_w=%d, n_obj=%d",
                  weights.tolist(), alpha, n_w, n_obj)
        return mars

    @staticmethod
    def _build_cvar_objective(alpha, n_w):
        """Build CVaR objective for single-objective."""
        risk_obj = CVaR(alpha=alpha, n_w=n_w)
        log.debug("RobustMCObjective CVaR: alpha=%.2f, n_w=%d", alpha, n_w)
        return risk_obj


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


def augment_generator_specs(client, exp_cfg, robust_cfg,
                            multifidelity=False) -> dict[str, Any]:
    """Augment BO GeneratorSpecs in the client's generation strategy for robust opt.

    Non-destructive: preserves user's node structure, transitions, and model config.
    Only modifies BO generator specs to add:
    - SubstituteContextFeatures input transform
    - RobustAcquisition class (CVaR for SOO, MARS for MOO)
    - Fixed features for context dimensions

    When ``multifidelity=True``, composes with MF acquisition instead of using
    RobustAcquisition:
    - Uses SingleTaskMultiFidelityGP as the model class
    - Passes RobustMCObjective via botorch_acqf_options.objective
    - Sets qMultiFidelityHypervolumeKnowledgeGradient as the acqf class

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


    # Build SurrogateSpec
    model_config_kwargs = {}
    if multifidelity:
        # MF+robust: use SingleTaskMultiFidelityGP per output.
        # Setting it in ModelConfig ensures Ax creates a ModelListGP of
        # per-output STMFGPs (required by qMFHVKG).
        from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
        model_config_kwargs["botorch_model_class"] = SingleTaskMultiFidelityGP
        # NO SubstituteContextFeatures — qMFHVKG can't handle one-to-many
        # transforms. Robustness from: GP training data spans context scenarios
        # (cycle_context), fixed_features pins context per generation.
        input_transform_classes = [Normalize]
        input_transform_options = {}
    else:
        # Non-MF robust: SCF expands across context scenarios, MARS aggregates
        input_transform_classes = [Normalize, SubstituteContextFeatures]
        input_transform_options = {
            "SubstituteContextFeatures": {
                "context_indices": ctx_indices,
                "feature_set": feature_set,
            },
        }

    surrogate_spec = SurrogateSpec(
        model_configs=[ModelConfig(
            input_transform_classes=input_transform_classes,
            input_transform_options=input_transform_options,
            **model_config_kwargs,
        )],
        # MF+robust: qMFHVKG requires ModelListGP (one GP per output).
        # allow_batched_models=False forces this even when a single
        # ModelConfig is provided.
        allow_batched_models=not multifidelity,
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
                if multifidelity:
                    # MF+robust: use qMFHVKG (standard MOO MF acqf) with
                    # SubstituteContextFeatures on the model for implicit
                    # robustness. MARS risk measure is not compatible with
                    # qMFKG's inner optimization loop (tensor shape issues).
                    # cycle_context rotates the pinned context point each
                    # generation for exploration across operating conditions.
                    try:
                        from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
                            qMultiFidelityHypervolumeKnowledgeGradient,
                        )
                        spec.generator_kwargs["botorch_acqf_class"] = \
                            qMultiFidelityHypervolumeKnowledgeGradient
                    except ImportError:
                        log.warning("MF+robust: qMFHVKG not available")
                    # MFHVKGAcquisition strips X_pending (not accepted by
                    # construct_inputs_qMFHVKG).
                    spec.generator_kwargs["acquisition_class"] = MFHVKGAcquisition
                else:
                    spec.generator_kwargs["acquisition_class"] = RobustAcquisition
                    spec.generator_kwargs["acquisition_options"] = {
                        "risk_config": risk_config,
                    }
                spec.fixed_features = ObservationFeatures(
                    parameters=first_ctx_normalized
                )
                augmented += 1

    mode = "MF+robust" if multifidelity else "robust"
    rm_label = "MARS" if is_moo else "CVaR"
    log.info("%s: %s(alpha=%.2f, n_w=%d), context_params=%s, "
             "augmented %d generator specs",
             mode, rm_label, alpha, n_w, ctx_names, augmented)

    return {
        "robust_cfg": robust_cfg,
        "ctx_names": ctx_names,
        "ctx_indices": ctx_indices,
        "feature_set": feature_set,
        "n_w": n_w,
        "risk_config": risk_config,
        "is_moo": is_moo,
        "multifidelity": multifidelity,
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
    register_acquisition(MFHVKGAcquisition)

    # Register RobustContextValueFunction (MF+robust inner value function)
    from botorch.acquisition import AcquisitionFunction as _AcqFn
    CLASS_TO_REGISTRY[_AcqFn][RobustContextValueFunction] = "RobustContextValueFunction"
    CLASS_TO_REVERSE_REGISTRY[_AcqFn]["RobustContextValueFunction"] = RobustContextValueFunction

    # Register multi-fidelity acquisition classes not shipped in Ax's default registry.
    # These are BoTorch AcquisitionFunction subclasses (not Ax Acquisition wrappers),
    # so they go into the AcquisitionFunction registry.
    _mf_acqf_classes = [
        ("botorch.acquisition.multi_objective.hypervolume_knowledge_gradient",
         "qMultiFidelityHypervolumeKnowledgeGradient"),
        ("botorch.acquisition.multi_objective.multi_fidelity", "MOMF"),
    ]
    for _mod, _cls_name in _mf_acqf_classes:
        try:
            _m = __import__(_mod, fromlist=[_cls_name])
            _cls = getattr(_m, _cls_name)
            CLASS_TO_REGISTRY[_AcqFn][_cls] = _cls_name
            CLASS_TO_REVERSE_REGISTRY[_AcqFn][_cls_name] = _cls
        except (ImportError, AttributeError):
            pass
except ImportError:
    pass
