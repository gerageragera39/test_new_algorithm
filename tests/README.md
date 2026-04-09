# `tests/`

Pytest suite for backend services, pipeline helpers, and API routes.

> Last verified against the current tracked test tree on 2026-04-07.

## Shared test infrastructure

| File | Purpose |
|---|---|
| `conftest.py` | shared fixtures such as `db_session` and `test_settings` |
| `fixtures/mock_youtube.json` | dry-run YouTube fixture |

## Coverage map

### Core services
- `test_preprocessing.py`
- `test_clustering.py`
- `test_labeling.py`
- `test_briefing.py`
- `test_exporter.py`
- `test_budget.py`
- `test_runtime_settings.py`
- `test_youtube_client_dry_run.py`

### Topic pipeline and helpers
- `test_pipeline_run_flow.py`
- `test_pipeline_upsert.py`
- `test_pipeline_moderation.py`
- `test_pipeline_positions.py`
- `test_pipeline_quality.py`
- `test_pipeline_watchdog.py`
- `test_pipeline_disagreement.py`
- `test_pipeline_llm_provider.py`
- `test_openai_provider_quota.py`

### Appeal analytics and toxic moderation
- `test_appeal_analytics.py`
- `test_toxic_moderation_prod.py`
- `test_youtube_ban_service_prod.py`
- `test_openai_payload_sanitization.py`

### API routes
- `test_routes_health.py`
- `test_routes_questions.py`
- `test_routes_report_detail.py`
- `test_routes_run_options.py`
- `test_routes_runtime_settings.py`

### Celery
- `test_celery_manual_mode.py`

## Running the suite

```bash
venv/bin/python -m pytest -q
```

Useful filters:

```bash
venv/bin/python -m pytest tests/test_pipeline_run_flow.py -q
venv/bin/python -m pytest tests -k appeal -q
venv/bin/python -m pytest tests/test_routes_run_options.py -q
```
