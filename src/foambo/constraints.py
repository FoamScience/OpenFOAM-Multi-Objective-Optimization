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
# Monkey-patch optimize_acqf to inject nonlinear constraints at call time.
# This avoids storing non-serializable callables in Ax's GenerationStrategy.
# ---------------------------------------------------------------------------

_active_nl_constraints: list[tuple[Callable[[torch.Tensor], torch.Tensor], bool]] | None = None
_orig_optimize_acqf = None


def patch_optimize_acqf(
    nl_constraints: list[tuple[Callable[[torch.Tensor], torch.Tensor], bool]],
) -> None:
    """Patch ``botorch.optim.optimize.optimize_acqf`` to inject *nl_constraints*.

    The patch is applied once; subsequent calls update the active constraints.
    """
    global _active_nl_constraints, _orig_optimize_acqf
    _active_nl_constraints = nl_constraints

    if _orig_optimize_acqf is not None:
        return  # already patched

    import botorch.optim.optimize as _botorch_optim
    _orig_optimize_acqf = _botorch_optim.optimize_acqf

    def _patched_optimize_acqf(*args, **kwargs):
        if _active_nl_constraints and "nonlinear_inequality_constraints" not in kwargs:
            kwargs["nonlinear_inequality_constraints"] = _active_nl_constraints
            log.debug("Injected %d nonlinear constraint(s) into optimize_acqf",
                      len(_active_nl_constraints))
        return _orig_optimize_acqf(*args, **kwargs)

    _botorch_optim.optimize_acqf = _patched_optimize_acqf
