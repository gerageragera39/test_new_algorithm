# scripts/

Container startup scripts for backend services.

## Scripts

| Script | Purpose |
|---|---|
| `start_api.sh` | Runs DB migrations then starts FastAPI (`uvicorn`) |
| `start_worker.sh` | Starts Celery worker |
| `start_beat.sh` | Starts Celery Beat scheduler |
| `benchmark_topic_models.py` | Replays topic clustering on historical comments to compare embedding models |

## `start_api.sh`

Behavior:

1. retries `alembic upgrade head` (up to 12 attempts, 3s delay),
2. exits with non-zero status if migrations keep failing,
3. starts `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

## `start_worker.sh`

Starts worker capable of executing:

- Topic Intelligence tasks (`run_latest_task`, `run_video_task`)
- Appeal Analytics task (`run_appeal_analytics_task`)
- scheduled dispatcher (`run_scheduled_latest_task`)

## `start_beat.sh`

Starts minute-based beat loop. The scheduled task itself decides whether to dispatch runs based on runtime settings (`beat_enabled`, `beat_time_kyiv`, last-trigger date).

## `benchmark_topic_models.py`

Usage example:

```bash
PYTHONPATH=. python scripts/benchmark_topic_models.py \
  --models Qwen/Qwen3-Embedding-0.6B BAAI/bge-m3 intfloat/multilingual-e5-large \
  --limit-videos 8 \
  --min-comments 120
```

Artifacts are written to:

- `data/benchmarks/YYYY-MM-DD/topic_models_benchmark.json`
- `data/benchmarks/YYYY-MM-DD/topic_models_benchmark.md`
