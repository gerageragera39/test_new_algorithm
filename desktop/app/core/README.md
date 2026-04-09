# app/core/

Core infrastructure shared across the backend.

## Files

| File | Purpose |
|---|---|
| `config.py` | Pydantic `Settings` with env-backed configuration |
| `exceptions.py` | Domain exception hierarchy |
| `logging.py` | JSON logging setup via `dictConfig` |
| `utils.py` | Text/hash/time/file helper functions |

## `config.py` Highlights

`Settings` is loaded from `.env` and environment variables and includes:

- app runtime (`app_name`, timezone, log level),
- YouTube fetch controls,
- OpenAI endpoint and model settings,
- embedding, clustering, moderation, and position extraction knobs,
- report/diagnostic toggles,
- DB and Celery infrastructure settings.

Compatibility notes:

- transcript-related fields still exist for backward-compatible env parsing,
- active pipeline runtime currently runs comment-only context and skips transcript stage.

`get_settings()` is cached with `lru_cache(maxsize=1)` and also ensures required data directories exist.

## `exceptions.py`

Hierarchy:

- `AppError` (base)
- `ExternalServiceError`
- `BudgetExceededError`
- `InvalidConfigurationError`

## `logging.py`

`configure_logging(settings)` configures root logger with JSON output:

- timestamp,
- level,
- logger name,
- message.

## `utils.py`

Common helpers used by services:

- `utcnow()`
- `hash_text(text)`
- `normalize_text(text)` (URL stripping + whitespace collapse)
- `looks_like_noise(text)`
- `contains_cyrillic(text)`
- `write_json(path, payload)`
