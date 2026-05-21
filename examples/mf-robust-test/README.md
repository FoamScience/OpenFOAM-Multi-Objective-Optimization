# MultiFidelity + Robust Benchmark

Toy MOO benchmark comparing multi-fidelity, robust, and combined strategies.

## Objectives

- **F1(x)**: polynomial + sin — min at x=0
- **F2(x)**: F3(x-50) log + cos — min at x=50
- Pareto tradeoff: can't minimize both simultaneously

## Fidelity

- **Expensive** (fidelity=1): true F1/F2
- **Cheap** (fidelity=0): Gaussian-smoothed (sigma=15) — same trend, less detail

## Operating Conditions (Robustness)

Two context parameters model operating-point uncertainty (like flowRate/rpm in pump optimization):

- **load** in [0.8, 1.2]: scales F1 amplitude (higher load = worse F1, F2 unaffected)
- **speed** in [0.9, 1.1]: shifts F2 optimum by 10*(speed-1) (F1 unaffected)

Each affects a different objective differently — realistic tradeoff under uncertainty.

## Configs

| Config | MF | Robust | Description |
|--------|:--:|:------:|-------------|
| `base` | | | Plain MOO |
| `robust` | | X | SCF + MARS |
| `mf` | X | | qMFHVKG |
| `mf_robust` | X | X | qMFHVKG + cycle_context |
| `robust_spec_n0` | | | Specialize: nominal (load=1, speed=1) |
| `robust_spec_n3` | | | Specialize: off-design (load=1.2, speed=0.9) |
| `mf_robust_spec_n0` | | | Specialize: nominal from MF+robust parent |
| `mf_robust_spec_n3` | | | Specialize: off-design from MF+robust parent |

## Run

```bash
# All configs + compare + plots
uv run python3 run_benchmark.py --all

# Base configs only
uv run python3 run_benchmark.py --configs base,robust,mf,mf_robust

# Replot existing results
uv run python3 run_benchmark.py --compare --plot --all
```

## Plots

Output in `results/plots/`:
- `pareto_fronts.png` — empirical + theoretical PFs (non-robust, robust, cheap fidelity)
- `convergence.png` — HV over trials with theoretical HV bounds
- `landscape.png` — F1/F2 curves (cheap + expensive fidelity) with trial evals
- `objectives.png` — F1/F2 per trial
- `specialization.png` — parent vs specialized trials
