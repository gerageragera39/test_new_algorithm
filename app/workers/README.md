# app/workers/

Celery execution layer for asynchronous and scheduled pipeline runs.

## Files

| File | Purpose |
|---|---|
| `celery_app.py` | Celery app config (broker/backend, beat schedule, timezone) |
| `tasks.py` | Task entry points for Topic Intelligence and Appeal Analytics |

## Celery Runtime

- Broker: `CELERY_BROKER_URL`
- Result backend: `CELERY_RESULT_BACKEND`
- Timezone: `APP_TIMEZONE` (default `Europe/Berlin`)
- Serialization: JSON
- Windows mode: `solo` worker pool with concurrency `1`

## Registered Tasks

| Task Name | Function | Description |
|---|---|---|
| `app.workers.tasks.run_latest_task` | `run_latest_task` | Topic pipeline for latest playlist video |
| `app.workers.tasks.run_video_task` | `run_video_task` | Topic pipeline for explicit video URL |
| `app.workers.tasks.run_appeal_analytics_task` | `run_appeal_analytics_task` | Appeal pipeline for latest or explicit video |
| `app.workers.tasks.run_scheduled_latest_task` | `run_scheduled_latest_task` | Minute-based scheduler gate that may dispatch both pipelines |

## Scheduled Behavior

`run_scheduled_latest_task` executes every minute and checks runtime settings:

1. `beat_enabled` must be true.
2. Current Kyiv time must equal `beat_time_kyiv`.
3. The same day must not have been triggered already.

When all checks pass, it dispatches:

- `run_latest_task`
- `run_appeal_analytics_task`

## Runtime Settings Integration

Each task builds effective settings as:

1. base `Settings` from env,
2. runtime JSON state from `RuntimeSettingsStore`,
3. optional task-level overrides merged into pipeline settings.
