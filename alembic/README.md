# alembic/

Database migration management for YouTubeAnalyzer.

## Files

| File/Dir | Purpose |
|---|---|
| `env.py` | Alembic runtime config (loads app settings and metadata) |
| `script.py.mako` | Migration template |
| `versions/` | Ordered migration scripts |

## Migration History

| Revision | File | Summary |
|---|---|---|
| `0001_initial_schema` | `0001_initial_schema.py` | Core tables (`videos`, `comments`, `runs`, `clusters`, `cluster_items`, `reports`, `budget_usage`, `embedding_cache`) |
| `0002_comment_moderation_fields` | `0002_comment_moderation_fields.py` | Adds comment moderation fields (`moderation_action`, `moderation_reason`, `moderation_source`, `moderation_score`) |
| `0003_appeal_analytics_tables` | `0003_appeal_analytics_tables.py` | Adds `appeal_runs`, `appeal_blocks`, `appeal_block_items` |

## How It Is Wired

`env.py` sets DB URL from application settings:

```python
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)
```

Migration target metadata is `Base.metadata` from `app.db.base`, with ORM models imported from `app.db.models`.

## Common Commands

```bash
alembic upgrade head
alembic revision --autogenerate -m "describe change"
alembic current
alembic history
alembic downgrade -1
```

## Startup Integration

`scripts/start_api.sh` runs `alembic upgrade head` before starting FastAPI.
