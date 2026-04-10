"""Constraint parsing, classification (linear vs nonlinear), and BoTorch callable builder."""

from __future__ import annotations

import logging
import re
from typing import Callable

import sympy
import torch

log = logging.getLogger(__name__)

# Pattern: "lhs <= rhs" or "lhs >= rhs"
_INEQ_RE = re.compile(r"^(.+?)\s*(<=|>=)\s*(.+)$")


def _parse_inequality(expr_str: str) -> tuple[sympy.Expr, str]:
    """Parse ``"lhs op rhs"`` into a sympy expression normalised to ``expr >= 0``.

    Returns (sympy_expr, original_string).
    """
    m = _INEQ_RE.match(expr_str.strip())
    if not m:
        raise ValueError(
            f"Constraint must use '<=' or '>=': {expr_str!r}"
        )
    lhs_str, op, rhs_str = m.group(1), m.group(2), m.group(3)
    lhs = sympy.sympify(lhs_str)
    rhs = sympy.sympify(rhs_str)
    # Normalise to "expr >= 0" (feasible when >= 0, matching BoTorch convention)
    if op == ">=":
        normalized = lhs - rhs
    else:
        normalized = rhs - lhs
    return normalized, expr_str


def is_linear(expr: sympy.Expr, symbols: set[sympy.Symbol]) -> bool:
    """Return True if *expr* is linear (degree <= 1) in *symbols*."""
    poly = sympy.Poly(expr, *symbols, domain="RR")
    return poly.total_degree() <= 1


def classify_constraints(
    constraints: list[str],
    param_names: list[str],
) -> tuple[list[str], list[str]]:
    """Split constraints into linear and nonlinear lists.

    Parameters
    ----------
    constraints:
        Raw constraint strings, e.g. ``["x1 + x2 <= 10", "x1 * x2 <= 5"]``.
    param_names:
        Ordered parameter names from the search space.

    Returns
    -------
    (linear, nonlinear) : tuple of string lists
    """
    symbols = {sympy.Symbol(n) for n in param_names}
    linear, nonlinear = [], []
    for c in constraints:
        normalized, original = _parse_inequality(c)
        if is_linear(normalized, symbols):
            linear.append(original)
        else:
            nonlinear.append(original)
    return linear, nonlinear


def build_nonlinear_callable(
    expr_str: str,
    param_names: list[str],
) -> tuple[Callable[[torch.Tensor], torch.Tensor], bool]:
    """Build a BoTorch-compatible constraint callable from a string expression.

    Parameters
    ----------
    expr_str:
        Inequality string, e.g. ``"x1 * x2 <= 10"``.
    param_names:
        Ordered parameter names matching the search-space column order.

    Returns
    -------
    (callable, intra_point) tuple for ``nonlinear_inequality_constraints``.
    The callable maps a ``(1 x d)`` tensor to a scalar tensor;
    feasible when ``>= 0``.  ``intra_point`` is always ``True``
    (these are parameter-space constraints, not inter-point).
    """
    normalized, _ = _parse_inequality(expr_str)
    symbols = [sympy.Symbol(n) for n in param_names]
    # lambdify to a numpy/torch-compatible function
    fn = sympy.lambdify(symbols, normalized, modules=["numpy"])

    def constraint_callable(X: torch.Tensor) -> torch.Tensor:
        # X shape: (1 x d) or (d,)
        x = X.squeeze(0) if X.dim() > 1 else X
        vals = {s.name: x[i].item() for i, s in enumerate(symbols)}
        result = fn(**{s.name: vals[s.name] for s in symbols})
        return torch.tensor(float(result), dtype=X.dtype, device=X.device)

    return constraint_callable, True


def build_nonlinear_constraints(
    constraint_strs: list[str],
    param_names: list[str],
) -> list[tuple[Callable[[torch.Tensor], torch.Tensor], bool]]:
    """Build all nonlinear constraint callables for ``optimize_acqf``.

    Parameters
    ----------
    constraint_strs:
        List of nonlinear inequality strings.
    param_names:
        Ordered parameter names from the search space.

    Returns
    -------
    List of ``(callable, intra_point)`` tuples suitable for
    ``optimize_acqf(nonlinear_inequality_constraints=...)``.
    """
    result = []
    for c in constraint_strs:
        result.append(build_nonlinear_callable(c, param_names))
        log.info("Nonlinear constraint registered: %s", c)
    return result


# ---------------------------------------------------------------------------
# Monkey-patch optimize_acqf to inject nonlinear constraints and/or
# fixed_features (for CVaR robust optimization) at call time.
# This avoids storing non-serializable callables in Ax's GenerationStrategy.
# ---------------------------------------------------------------------------

_active_nl_constraints: list[tuple[Callable[[torch.Tensor], torch.Tensor], bool]] | None = None
_active_fixed_features: dict[int, float] | None = None
_orig_optimize_acqf = None
_batch_size: int = 1
_candidate_cache: list = []
_compile_acqf: bool = False


def _patched_optimize_acqf(*args, **kwargs):
    global _candidate_cache

    # --- Inject constraints / fixed_features ---
    if _active_nl_constraints and "nonlinear_inequality_constraints" not in kwargs:
        kwargs["nonlinear_inequality_constraints"] = _active_nl_constraints
        log.debug("Injected %d nonlinear constraint(s) into optimize_acqf",
                  len(_active_nl_constraints))
    if _active_fixed_features and "fixed_features" not in kwargs:
        kwargs["fixed_features"] = _active_fixed_features
        log.debug("Injected %d fixed_feature(s) into optimize_acqf",
                  len(_active_fixed_features))

    # --- Return cached candidate from previous batch ---
    if _candidate_cache:
        log.debug("Returning cached candidate (%d remaining)", len(_candidate_cache) - 1)
        return _candidate_cache.pop(0)

    # --- torch.compile on acquisition function ---
    if _compile_acqf:
        acq = kwargs.get('acq_function') or (args[0] if args else None)
        if acq is not None and not getattr(acq, '_foambo_compiled', False):
            import torch as _torch
            log.info("Compiling acquisition function with torch.compile (first call may be slow)")
            compiled_acq = _torch.compile(acq, dynamic=False)
            compiled_acq._foambo_compiled = True
            if 'acq_function' in kwargs:
                kwargs['acq_function'] = compiled_acq
            elif args:
                args = (compiled_acq,) + args[1:]

    # Batch generation (q>1) not viable for CVaR + mixed discrete params:
    # optimize_acqf_mixed loops over discrete combos, each running q*d-dim
    # joint SLSQP with q*n_w GP evals per step. Intractable at scale.

    return _orig_optimize_acqf(*args, **kwargs)


def _ensure_patched() -> None:
    """Apply the optimize_acqf monkey-patch once (idempotent)."""
    global _orig_optimize_acqf
    if _orig_optimize_acqf is not None:
        return
    import botorch.optim.optimize as _botorch_optim
    _orig_optimize_acqf = _botorch_optim.optimize_acqf
    _botorch_optim.optimize_acqf = _patched_optimize_acqf


def patch_optimize_acqf(
    nl_constraints: list[tuple[Callable[[torch.Tensor], torch.Tensor], bool]],
) -> None:
    """Patch ``botorch.optim.optimize.optimize_acqf`` to inject *nl_constraints*.

    The patch is applied once; subsequent calls update the active constraints.
    """
    global _active_nl_constraints
    _active_nl_constraints = nl_constraints
    _ensure_patched()


def set_fixed_features(fixed: dict[int, float] | None) -> None:
    """Update active fixed_features for context dims. Called per-trial for round-robin."""
    global _active_fixed_features
    _active_fixed_features = fixed


def patch_optimize_acqf_robust(fixed_features: dict[int, float]) -> None:
    """Ensure patch is applied and set initial fixed_features for robust optimization."""
    set_fixed_features(fixed_features)
    _ensure_patched()


def set_batch_size(size: int) -> None:
    """Set batch candidate generation size (q>1 joint optimization)."""
    global _batch_size
    _batch_size = max(size, 1)
    if _batch_size > 1:
        log.info("Batch candidate generation enabled: q=%d", _batch_size)


def enable_acqf_compile(enabled: bool = True) -> None:
    """Enable/disable torch.compile on acquisition function."""
    global _compile_acqf
    _compile_acqf = enabled


def clear_candidate_cache() -> None:
    """Flush cached candidates (call when model is refitted)."""
    global _candidate_cache
    if _candidate_cache:
        log.debug("Cleared %d cached candidates (model refitted)", len(_candidate_cache))
    _candidate_cache = []
