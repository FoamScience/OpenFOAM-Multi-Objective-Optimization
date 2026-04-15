# Code design

foamBO = Ax + pydantic v2 + foamlib + FastAPI.

## Config tree

Root model: `FoamBOConfig` in `orchestrate.py`. Sub-models validated via `@field_validator`.
All inherit `FoamBOBaseModel`. Field defaults + `examples=[...]` harvested by `default_config.py::get_default_config()`.

## Subsystems

| File | Job |
|---|---|
| `orchestrate.py` | pydantic models (config tree) |
| `optimize.py` | CLI entry, Ax client lifecycle |
| `api.py` | `FoamBO` fluent builder |
| `api_server.py` | FastAPI dashboard + REST + config builder |
| `preflight.py` | static checks, dry-run |
| `metrics.py` | `FoamJob`, `LocalJobMetric` |
| `default_config.py` | defaults + `get_config_docs()` harvest |
| `docs_concepts.py` | long-form concept entries |
| `archive.py` | trial archival |
| `analysis.py` | post-hoc analysis |
| `config_upgrade.py` | version migration |

## Orchestration

foamBO uses Ax's native `Orchestrator` (poll-based). Harvest latency is
bounded by `init_seconds_between_polls` in the scheduler options.

`FoamJobRunner.poll_trial` consumes a push queue at the top of each poll
cycle so remote runners (SLURM, SSH) can notify completion via HTTP POST:

- `POST /api/v1/trials/{idx}/push/status`   — writes `_state.trial_status_overrides[idx]`
- `POST /api/v1/trials/{idx}/push/metrics`  — appends to `_state.trial_pushed_metrics[idx]`
- `POST /api/v1/trials/{idx}/push/heartbeat` — liveness only
- `GET  /api/v1/events`                     — dashboard-facing event log

Env vars injected into every trial subprocess:
`FOAMBO_API_ENDPOINT`, `FOAMBO_TRIAL_INDEX`, `FOAMBO_SESSION_ID`.

Session-ID validation rejects pushes from stale jobs of prior crashed runs (HTTP 409).

## Key subsystems

- **Trial dependencies** — `TrialDependency` / `TrialSelector` / `TrialAction`. Phases: `immediate`, `pre_init`, `pre_mesh`, `pre_solve`, `post_solve`. Hook scripts exposed via `$FOAMBO_PRE_MESH` etc.
- **Dimensionality reduction** — `DimensionalityReductionOptions` inside `ConfigOrchestratorOptions`. Sobol-based param freezing after N trials.
- **Parameter groups** — `groups: [...]` tag on each param. Feeds group sensitivity, `matching_group` dep strategy, group-frozen sweeps, what-if in Predict tab.
- **Config builder UI** — `templates/config_builder.{html,js,css}`. Pico CSS + Alpine. Endpoints: `/config-builder`, `/api/v1/config/{schema,docs,validate,preflight}`.

## Docs pipeline

`default_config.py::get_config_docs()` merges:

1. Pydantic Field descriptions (`harvest_docs` on `_get_doc_models()`)
2. Manual YAML examples (`_examples` dict)
3. `docs_concepts.py::CONCEPTS` — long-form markdown
4. `load_tutorial_docs()` — `docs/*.md` summaries

Served at `GET /api/v1/config/docs` for the config builder tooltips + Concepts modal. TUI browses same data via `foamBO --docs`.

## Contributing

1. Add/modify pydantic model. Use `Field(description=..., examples=[...])` — both are harvested.
2. If it's a new root section, add to `FoamBOConfig` + `_get_root_models()` + `_get_doc_models()`.
3. For cross-cutting examples, drop an entry in `_examples` in `default_config.py`.
4. For concept-level docs, add to `CONCEPTS` in `docs_concepts.py`.
5. Run `foamBO --preflight <cfg>` on changed configs before commit.
