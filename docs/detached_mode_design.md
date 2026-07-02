# foamBO detached mode (ask / run / tell) — design

**Status:** design only — implementation in a later session.
**Author trace:** end-to-end read of `src/foambo/{optimize,orchestrate,metrics,common,api,api_server,archive,analysis}.py` (ax-platform modern `Client`). All file:line refs are against the current tree.

---

## 1. What & why

Today foamBO runs one long-lived process that owns the whole campaign: the Ax
**Orchestrator** drives `ask → run → poll → tell` in a loop until the budget is
spent (`optimize.py:1197` `TimedOrchestrator`, `:1221` `run_all_trials`). For CFD
that process — torch + Ax + gpytorch + pyvista — sits **mostly idle for hours to
days**, often pinned to a cluster allocation, just to fit a GP for a few seconds
between multi-hour trials.

**Detached mode** decomposes that single loop into **independently-invokable,
stateless steps coupled only through the filesystem** (the trial dirs + the state
store *are* the queue):

```
foamBO --config study.yaml --step     # brain: ingest finished results → ask N → write trial sidecars → save → exit   (seconds, login node)
   … run the N cases (foamBO --run, OR your own scheduler) …
foamBO --config study.yaml --step     # ingest, ask next batch … repeat to budget
```

Goals:
- **Cost.** The brain runs for seconds per batch on a login node; nothing
  heavyweight is pinned to the cluster for the campaign.
- **Bring-your-own scheduler.** A user can drive the cases between `--step`
  calls however they like (SLURM array, k8s, by hand) — config-only, no code.
- **Env decoupling.** The optimizer env (numpy 2 / torch) and the solver env
  never have to coexist; they communicate by files. (Directly relevant to
  CENTRIPUMP-235 B1/B3, where foamBO's numpy-2 stack cannot share the
  impelworks numpy<2 / cadquery-ocp env.)
- **foamBO still ships both halves.** The executor half (`--run`) is a
  first-class command reusing the existing runner — so out of the box you get a
  full loop, but either half is replaceable.

This is the same shape `universal-parameter-optimizer` uses ("run simulations and
call the script again"), except native and reusing Ax's `Client` ask/tell instead
of reimplementing a BO half-loop.

---

## 2. Command surface

Add to the existing mutually-exclusive arg group in `main()` (`optimize.py:1404`),
each mirroring the `--no-opt` block (`optimize.py:1557-1611`) for config load +
client/runner restore, then calling one new function.

| Command | Role | Idempotent? |
|---|---|---|
| `foamBO --config c.yaml --step` | **Brain.** Ingest any finished results (tell), then `get_next_trials(N)` (ask), materialize sidecars, save, exit. | Yes — re-running ingests stragglers, then emits. Self-healing. |
| `foamBO --config c.yaml --run` | **Executor.** Dispatch every pending (RUNNING, has `case_path`) trial via the existing runner, wait to completion, write fitness artifact. | Yes — skips trials already done. |
| `foamBO --config c.yaml --tell` | *(optional)* Ingest-only, no ask. For explicit SLURM-array pipelines that separate harvest from generate. | Yes |

**Recommendation:** ship `--step` + `--run` for v1; add `--tell` only if a user
needs harvest/generate split (`--step` already does tell-then-ask). `ponytail:`
one idempotent `--step` beats a forced `--ask`/`--tell` split — the split doubles
the CLI and the load/save round-trip for no gain on the common path.

### Two usage patterns, config-only switch

The seam for "my scheduler vs foamBO's executor" is the **existing**
`optimization.case_runner.runner` + `.mode` field (`orchestrate.py:1059,1069`) —
no new code:

- **BYO scheduler** — `case_runner.runner: ""`. `--step` materializes sidecars
  and stops (empty runner already short-circuits the Popen,
  `metrics.py:90` `if cmd:`). You run the cases; the next `--step` ingests them.
- **foamBO executes** — `case_runner.runner: "./Allrun"` (local) or `mode: remote`
  + `remote_status_query`/`remote_early_stop` (SLURM). `foamBO --run` executes
  the batch; or the classic closed loop still works unchanged.

---

## 3. Reuse map — what already exists

Everything below is callable standalone (proven by existing standalone callers):

| Need | Existing call | Evidence |
|---|---|---|
| ask N trials | `client.get_next_trials(max_trials=N)` → `{idx: params}`, trials created RUNNING, no dispatch | `api_server.py:2530` |
| tell one trial | `client.complete_trial(trial_index, raw_data, progression)` | `orchestrate.py:854` |
| read scalar fitness from a finished dir | `FoamJobMetric.fetch(idx, meta)` → `FoamJob.local_metric` → `parse_outcome_for_metric` | `metrics.py:265,130`, `common.py:73` |
| materialize sidecar, no execution | `runner.preprocessor.setup(params, runner.cfg)` (or `run_trial` with `runner=""`) | `metrics.py:791`, `common.py:224` |
| persist / resume full Client | `store_cfg.save(client)` / `store_cfg.load()` | `orchestrate.py:1235/1221`; standalone use at `archive.py:219`, `analysis.py:364` |
| executor dispatch + poll | `FoamJobRunner.run_trial` + `poll_trial` + `FoamJob.local_case_run`/`local_status_query` | `metrics.py:789,837,65,160` |
| full client+runner setup | the block at `optimize.py:337-655` | — |
| ingest-external pattern (reference) | `SeedDataNode._load_csv/_load_foambo_json` | `orchestrate.py:823,859` |

### What must be written (the whole diff)

1. **Refactor `_build_client(cfg) -> (client, runner, opt_cfg, store_cfg, …)`** —
   lift `optimize.py:337-655` verbatim so both `optimize()` and the new steps
   share identical setup. Pure refactor, no behavior change.
2. **Three thin entry functions** (`_do_step`, `_do_run`, optional `_do_tell`) +
   their `if args.X:` CLI blocks.
3. **Stamp `trial_index` into the sidecar** — one line in `NoCasePreprocessor`
   payload (`common.py:244`) + a tiny `.foambo_trial.json` next to cloned cases.
   Only needed for the fully-external-scheduler write-back (see §5).

That's it. ~1 refactor + 3 entry blocks + 1 line.

---

## 4. Step sequences (→ existing calls)

### `--step` (brain: tell-then-ask)
1. `load_config` + apply `overrides` (copy `--no-opt` block `optimize.py:1562-1566`).
2. `client, runner, opt_cfg, store_cfg = _build_client(cfg)` — `store.read_from=json`
   resumes the prior state (see fork D4 for first-run).
3. **Tell:** for each RUNNING trial that has a finished result (fitness file
   present, or `FoamJobMetric.fetch` succeeds against its `case_path`):
   `raw_data = {m.name: m.fetch(idx, meta)[1] for m in metrics}`;
   `client.complete_trial(idx, raw_data)`. Failures → `client.mark_trial_failed(idx)`
   (new use; Ax supports it, foamBO doesn't call it yet).
4. **Ask:** `trials = client.get_next_trials(max_trials=N)` (N = `parallelism`).
5. **Materialize:** for `idx, params in trials.items()`:
   `cd = runner.preprocessor.setup(params, runner.cfg)`;
   `client._experiment.trials[idx].update_run_metadata({"case_path": str(cd['case'].path), "parameters": params})`;
   stamp `idx` into the sidecar.
6. **Persist:** `store_cfg.save(client)`; refresh `runner.trial_registry`
   (copy `callback` `:1072-1081`); optionally write report CSV (`callback` `:1119-1139`).
7. Exit. Stdout/JSON lists the emitted `{idx, case_path}` for the user/executor.

### `--run` (executor: dispatch + wait + write fitness)
1-2. as above.
3. Scan `client._experiment.trials` for RUNNING trials with a `case_path` in
   `run_metadata`.
4. For each: `runner.dispatcher[mode](params, case_path, runner.cfg['template_case'])`
   then **wait** — `mode: local` Popen is async (`local_case_run:95`
   `start_new_session=True` returns the pid immediately), so loop
   `runner.poll_trial(idx, meta)` until COMPLETED/FAILED (or `jobs[pid].wait()`).
5. On finish, write the fitness artifact (or `complete_trial` directly — see fork D2).
6. `store_cfg.save(client)`.

> **In-process handle:** the `jobs` dict (`metrics.py:107`) is memory-only — the
> process that dispatches a `local` job must be the one that waits. So `--run`
> runs a batch to completion in one process. For dispatch-in-one-process /
> poll-in-another, use `mode: remote`, where status comes from a command, not a
> live `Popen` handle.

### `--tell` (optional, ingest-only)
Steps 1-2, then step 3 of `--step`, then `store_cfg.save`. Identity via
`run_metadata["case_path"]` (from store) or stamped `trial_index` (external).

---

## 5. Identity — the filesystem-as-queue handshake

The brain creates trials with `get_next_trials` (real Ax indices). The executor /
user must return results against **those existing indices** — not create new ones.
Two identity sources:

- **From the store (default, foamBO executor or `--tell` on the same host):**
  `trial_index → case_path` lives in `run_metadata` (set at materialize, step 5)
  and `runner.trial_registry`. Fully recoverable; nothing extra needed.
- **From the sidecar (fully-external scheduler with no store access):** the
  sidecar today is **hash-named and params-only** (`NoCasePreprocessor`,
  `common.py:244`) — the index is *not* in it. **Stamp `trial_index`** so an
  external writer can produce `{trial_index, metrics}` results the next `--step`
  reads back. `ponytail:` stamp always — it's one line and removes a whole class
  of reconciliation bugs.

Minimal handshake:
```
--step  writes  <trial_dir>/...json  ⊇ { __foambo_trial_index__: idx, <params> }
executor reads it, runs the case, writes  <trial_dir>/<fitness artifact>   (or {trial_index, metrics})
--step  reads it back → complete_trial(idx, raw_data)
```
The **presence of the fitness artifact is the "done" flag** the tell-step scans
for. Trial status stays RUNNING (set by `get_next_trials`) until `complete_trial`
flips it COMPLETED — so RUNNING-with-result = "ready to tell", RUNNING-without =
"still cooking", COMPLETED = "told". No new status machinery.

---

## 6. Resolved design decisions

| # | Fork | Decision | Why |
|---|---|---|---|
| D1 | single `--step` vs split `--ask`/`--tell` | **Single idempotent `--step`** (`--tell` optional extra) | fewer commands, self-healing, one load/save round-trip |
| D2 | result discovery: re-`fetch` vs result sidecar | **Both, by usage:** foamBO executor / same-host → re-`FoamJobMetric.fetch` against `case_path` (zero new format); fully-external → `{trial_index, metrics}` sidecar | reuse metric commands when the store is reachable; explicit file when it isn't |
| D3 | identity key | **Stamp `trial_index` into sidecar AND keep store `case_path` map** | store covers local; stamp covers external; both cheap |
| D4 | first-run bootstrap | small load helper: `read_from=json` but **treat missing file as fresh `Client()`** | avoids the two-config dance (`StoreOptions.load` currently raises on missing, `orchestrate.py:1221`) |

---

## 7. Landmines (must-replicate — beyond "Ax supports ask/tell")

These are things the Orchestrator does implicitly that a detached step must do by
hand. **All confirmed in the trace.**

1. **`get_next_trials` does NOT set foamBO `run_metadata`.** The Orchestrator gets
   `case_path` only because it calls `runner.run_trial`. In `--step` you **must**
   `trial.update_run_metadata({"case_path":…, "parameters":…})` after
   `preprocessor.setup`, or the idx→dir map is lost and tell/executor can't find
   the case.
2. **`_fn_registry` is module-level and not persisted** (`optimize.py:525-532`).
   Every detached invocation must re-run `m.to_metric()` for expression metrics,
   else `FoamJobMetric.fetch` tries to exec the `__expr__<name>` sentinel. → put
   it in `_build_client`.
3. **`runner.trial_registry` is runtime-only** (`metrics.py:580`), seeded from the
   experiment at `optimize.py:550-562` and refreshed in `callback:1072`. Skip it
   and trial-dependency resolution (`_resolve_source_trial`, `metrics.py:621`;
   `nearest/best/matching_group`) breaks. → reseed in every step.
4. **Advanced-mode hooks are `callback`-driven, not in `get_next_trials`.** MF cost
   update (`_update_cost_state`), robust `cycle_context`, dim-reduction /
   region-screening / baseline / SeedDataNode all run from the Orchestrator
   `callback` (`optimize.py:1060-1070`). Basic SOO/MOO detached works as-is; if MF
   or robust is configured, `--step` must call those hooks **between tell and ask**
   or they silently no-op. → see §8 scope.
5. **Timeouts:** the Orchestrator owns `timeout_hours` (`run_all_trials`) and
   per-trial timeout (`poll_trial:867-891`). In detached mode the user's scheduler
   owns wall-time; foamBO's `trial_timeout` only applies inside `foamBO --run`.
6. **In-process `jobs` dict** (`metrics.py:107`) — already covered in §4: dispatch
   and poll of a `local` job must be the same process.

---

## 8. Scope

**v1 (this design):** basic single- and multi-objective detached `--step` + `--run`
+ BYO-scheduler via `runner=""`. Covers CENTRIPUMP B4 (basic) and the B1/B3
cost-effective dispatch path.

**v2 (deferred, landmine #4):** multi-fidelity and robust in detached mode — needs
the Orchestrator `callback` hooks (`_update_cost_state`, `cycle_context`,
dim-reduction) factored into a `_between_tell_and_ask(client, cfg)` helper called
by `--step`. Until then, MF/robust campaigns use the closed loop. Flag clearly:
`--step` with MF/robust configured should **warn** that cost/robust hooks are not
yet wired, not silently no-op.

**Non-goals:** changing the closed-loop default behavior; a new store format; a
network protocol (the filesystem is the queue).

---

## 9. Acceptance / tests

- **Refactor parity:** `optimize()` closed-loop run produces identical results
  before/after `_build_client` extraction (golden Branin SOO + ZDT1 MOO, reuse
  `tests/integration`).
- **Detached round-trip:** `--step` (fresh) emits N sidecars with stamped indices
  + `case_path` in store; manually drop fitness files; `--step` again ingests all
  N (`complete_trial` called N×) and emits the next batch; experiment data matches
  a closed-loop run of the same seed within noise.
- **Executor:** `--run` with `runner="./echo_fitness.sh"` dispatches + writes
  fitness for every pending trial; subsequent `--step` ingests them.
- **BYO seam:** `runner=""` → `--step` materializes only, runs no process
  (assert no Popen); external write-back via stamped index ingests correctly.
- **Resume:** kill between `--step` calls, re-run — no duplicate trials, no lost
  data (store is authoritative).

---

## 10. Open questions for owner

1. CLI naming: `--step`/`--run`/`--tell` vs `--ask`/`--tell`/`--run` vs a single
   `--detached <step|run>` subcommand. (Recommend `--step`/`--run`.)
2. Default result-discovery for the foamBO executor: re-`fetch` (D2a) writes
   nothing extra; do we also want it to drop a canonical `fitness.json` for
   downstream tooling (CENTRIPUMP B2 schema)? Cheap to add in `--run`.
3. Should `--step` print machine-readable JSON (`{emitted:[{idx,case_path}], told:[idx]}`)
   on stdout for scripting the BYO-scheduler loop? (Recommend yes.)
4. MF/robust in detached (v2) priority — wire the `callback` hooks now, or ship
   basic-only and defer?
