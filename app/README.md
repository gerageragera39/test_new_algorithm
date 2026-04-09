# `app/`

Backend application package for **YouTube Intel**.

## Responsibilities

- FastAPI API + SPA serving
- Topic Intelligence pipeline orchestration
- Appeal Analytics pipeline orchestration
- runtime settings, budget tracking, moderation helpers
- Celery async and scheduled execution

## Directory map

| Path | Responsibility |
|---|---|
| `api/` | FastAPI routes and dependencies |
| `core/` | configuration, exceptions, logging, utilities |
| `db/` | SQLAlchemy models, base, sessions |
| `schemas/` | Pydantic API/domain schemas |
| `services/` | pipeline logic, LLM/embedding integrations, reporting, moderation |
| `workers/` | Celery app and task entry points |
| `main.py` | FastAPI bootstrap |

## Runtime entry points

- API: `app/main.py`
- Topic pipeline: `app/services/pipeline/runner.py`
- Appeal pipeline: `app/services/appeal_analytics/runner.py`
- Celery tasks: `app/workers/tasks.py`

## Topic Intelligence stages

Stored in `runs.meta_json`:

1. `context`
2. `comments_fetch`
3. `preprocess`
4. `comments_persist`
5. `embeddings`
6. `clustering`
7. `episode_match` *(compatibility stage, currently skipped because transcription is removed from the active runtime flow)*
8. `labeling`
9. `clusters_persist`
10. `briefing`
11. `report_export`

## Appeal Analytics stages

Stored in `appeal_runs.meta_json`:

1. `load`
2. `classify`
3. `refine`
4. `toxic`
5. `persist`

## Notes

- runtime settings are persisted via `app/services/runtime_settings.py`;
- OpenAI usage is tracked in the database and protected by per-run budget / call safeguards;
- the pipeline package is intentionally decomposed into focused helpers (`PositionExtractor`, `ClusterEnricher`, `ReportBuilder`, `QualityMetrics`).
