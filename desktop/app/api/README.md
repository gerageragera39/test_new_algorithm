# app/api/

FastAPI routing layer for backend HTTP and SPA routes.

## Files

| File | Purpose |
|---|---|
| `routes.py` | API endpoints, UI redirect/SPA handlers, response assembly |
| `deps.py` | Dependency providers for `Settings` and DB `Session` |

## Endpoint Groups

## Health

| Method | Path | Response | Notes |
|---|---|---|---|
| `GET` | `/health` | `HealthResponse` | Includes endpoint host/mode info for OpenAI base URL |

## Topic Intelligence Runs

| Method | Path | Response | Notes |
|---|---|---|---|
| `POST` | `/run/latest` | `RunResponse` | Async by default; `?sync=true` runs inline |
| `POST` | `/run/video` | `RunResponse` | Supports `video_url` in query or JSON body; `?sync=true` supported |

Optional run option for both endpoints:

- `skip_filtering` (query or JSON body): disables preprocessing filters for that run.

## Videos and Reports

| Method | Path | Response |
|---|---|---|
| `GET` | `/videos` | `list[VideoItemResponse]` |
| `GET` | `/videos/statuses` | `list[VideoStatusItemResponse]` |
| `GET` | `/videos/{video_id}` | `VideoDetailResponse` |
| `GET` | `/reports/latest` | `ReportResponse` |
| `GET` | `/reports/{video_id}` | `ReportResponse` |
| `GET` | `/reports/{video_id}/detail` | `ReportDetailResponse` |

`/videos/statuses` contains status fields for both pipelines:

- `run_status`, stage progress, report availability (Topic Intelligence),
- `appeal_run_status`, `has_appeal_report` (Appeal Analytics).

## Appeal Analytics

| Method | Path | Response | Notes |
|---|---|---|---|
| `POST` | `/appeal/run` | `RunResponse` | Latest video when URL omitted; `?sync=true` supported |
| `GET` | `/appeal/{video_id}` | `AppealAnalyticsResponse` | Returns latest completed appeal run for that video |
| `GET` | `/appeal/{video_id}/author/{author_name}` | `AuthorCommentsResponse` | Returns all comments by the specified author |

## Runtime and Budget

| Method | Path | Response |
|---|---|---|
| `GET` | `/budget` | `BudgetUsageResponse` |
| `GET` | `/settings/runtime` | `RuntimeSettingsResponse` |
| `PUT` | `/settings/runtime` | `RuntimeSettingsResponse` |

## UI Routing

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/` | Redirects to `/ui` |
| `GET` | `/ui` and `/ui/{path}` | Serves `frontend/dist/index.html` (SPA shell) |

If frontend build is missing, UI routes return `503` with build instructions.

## Dependency Injection

- `get_settings_dep()` -> cached `Settings` instance.
- `get_db_dep()` -> request-scoped SQLAlchemy `Session`.
