# Closed loop vs. detached loop

foamBO can drive an optimization campaign in two ways:

1. **Closed loop** (the default): one long-lived process owns the whole
   campaign. Ax's `Orchestrator` loops `ask → run → poll → tell` until the
   budget is spent.
   ```bash
   foamBO --config study.yaml
   ```
2. **Detached loop**: the campaign is decomposed into independently-invokable,
   stateless steps coupled only through the filesystem — the trial dirs plus
   the state store *are* the queue. Each step loads the client from the store,
   does its piece, saves, and exits.
   ```bash
   foamBO --config study.yaml --step   # brain: tell finished results, ask + materialize next batch, exit
   # ... run the emitted cases (foamBO --run, or your own scheduler) ...
   foamBO --config study.yaml --step   # ingest, ask next batch ... repeat to budget
   ```

Why detached? The closed-loop process (torch + Ax + gpytorch) sits mostly idle
for hours between multi-hour CFD trials, often pinned to a cluster allocation.
In detached mode the "brain" runs for seconds per batch on a login node, and
the cases run wherever and however you like (SLURM array, k8s, by hand). The
optimizer environment and the solver environment never have to coexist — they
communicate by files.

## The three commands

All three are idempotent and safe to re-run; the state store is authoritative.

| Command | Role |
|---|---|
| `foamBO --config c.yaml --step` | **Brain.** Ingest any finished results (tell), then ask the next batch (bounded by `orchestration_settings.parallelism`), materialize the trial case dirs *without dispatching them*, save, exit. |
| `foamBO --config c.yaml --run` | **Executor.** Dispatch every pending materialized trial via the configured `case_runner`, wait for the whole batch, ingest results, save, exit. |
| `foamBO --config c.yaml --tell` | **Ingest-only.** Complete trials whose results are on disk, save, exit — no ask. For pipelines that separate harvest from generate; `--step` already does tell-then-ask. |

Each command prints one machine-readable JSON line on stdout for scripting:
`--step` → `{"told": [...], "failed": [...], "emitted": [{"idx": N, "case_path": "..."}]}`,
`--run` → `{"dispatched": [...], "told": [...], "failed": [...]}`,
`--tell` → `{"told": [...], "failed": [...]}`.

## Configuration

Detached mode needs a persistent store so successive invocations see the same
experiment:

```yaml
store:
  save_to: json
  read_from: json   # a missing state file on the first --step is fine: it bootstraps fresh
```

The seam between "foamBO executes" and "my scheduler executes" is the existing
`optimization.case_runner.runner` field — no dedicated detached settings:

- **foamBO executes** — set `runner: ./Allrun` (or `mode: remote` with
  `remote_status_query`). Alternate `--step` and `--run`:
  ```bash
  while true; do
    foamBO --config study.yaml --step
    foamBO --config study.yaml --run
  done
  ```
- **Bring-your-own scheduler** — set `runner: null`. `--step` only
  materializes the case dirs and stops; you run the cases between calls, and
  the next `--step` ingests whatever finished.

## How results are matched back (the handshake)

The brain creates trials with real Ax indices; results must come back against
those indices. Two identity sources, both always written:

- `trial_index → case_path` is recorded in the trial's `run_metadata` inside
  the state store — enough for `--run`/`--tell` on the same host.
- every materialized case dir gets a `.foambo_trial.json` stamp (and no-case
  sidecars carry `__foambo_trial_index__`) so a fully-external scheduler with
  no store access can still write back `{trial_index, metrics}`.

There is no explicit "done" flag: a trial is ready to tell when **every
declared metric fetches a finite value** from its case dir (a metric command
that fails or returns NaN means the case is still cooking, and the trial is
skipped until the next step). Trials stay RUNNING until `--step`/`--tell`
completes them; a trial whose completion errors out is marked FAILED.

## Limitations

- **Multi-fidelity and robust campaigns: use the closed loop.** Their
  between-trial hooks (MF cost-model updates, robust context cycling) run only
  from the closed-loop Orchestrator callback. A detached step on such a config
  warns loudly and would otherwise use a stale cost model / context point.
- `--run` with `mode: local` must dispatch and wait within one invocation (the
  process holds the live subprocess handles), so it drives the whole batch to
  completion before exiting. For dispatch-in-one-process / poll-in-another use
  `mode: remote`.
- Wall-clock control: `timeout_hours` and per-trial timeouts belong to the
  closed loop. In detached mode your scheduler owns wall-time; foamBO's trial
  timeout applies only inside `foamBO --run`.

The full design rationale lives in [the detached-mode design doc](./detached_mode_design.md).
