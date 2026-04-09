# app/db/

Database layer (SQLAlchemy ORM + session management).

## Files

| File | Purpose |
|---|---|
| `base.py` | Declarative `Base` and `TimestampMixin` |
| `models.py` | ORM table models |
| `session.py` | Engine and `SessionLocal` factory |

## ORM Models

| Model | Table | Description |
|---|---|---|
| `Video` | `videos` | YouTube video metadata |
| `Comment` | `comments` | Raw/processed comments with moderation fields |
| `Run` | `runs` | Topic pipeline execution state and stage metadata |
| `Cluster` | `clusters` | Topic cluster summary per run |
| `ClusterItem` | `cluster_items` | Comment-to-cluster linkage |
| `Report` | `reports` | Markdown/HTML/JSON report artifact |
| `AppealRun` | `appeal_runs` | Appeal pipeline execution state |
| `AppealBlock` | `appeal_blocks` | One persisted classification block in appeal run |
| `AppealBlockItem` | `appeal_block_items` | Comment-level entries inside appeal blocks |
| `BudgetUsage` | `budget_usage` | OpenAI usage/cost ledger |
| `EmbeddingCache` | `embedding_cache` | Embedding cache by provider/model/text hash |

## Session and Engine

`session.py` builds SQLAlchemy engine from `settings.database_url` with:

- `pool_pre_ping=True`
- `future=True`

`get_db()` yields and closes `SessionLocal()` for FastAPI dependencies.

## Migration Target

Alembic uses `Base.metadata` from `app.db.base` and imports `app.db.models` to register all tables.
