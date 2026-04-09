# app/schemas/

Pydantic schema layer for domain models and API payloads.

## Files

| File | Purpose |
|---|---|
| `domain.py` | Internal pipeline DTOs (topic pipeline and shared primitives) |
| `api.py` | FastAPI request/response payload schemas |

## `domain.py` Models

Core models:

- `VideoMeta`
- `RawComment`
- `ProcessedComment`
- `ClusterDraft`
- `ClusterLabelResult`
- `TopicPosition`
- `TopicSummary`
- `DailyBriefing`

Compatibility models:

- `EpisodeTopic`
- `EpisodeContext`

`EpisodeContext` remains in schema contracts, while active runtime uses comment-only context (`topics=[]`).

## `api.py` Models

### Core Pipeline + Reports

- `HealthResponse`
- `RunResponse`
- `VideoItemResponse`
- `VideoStatusItemResponse`
- `VideoDetailResponse`
- `ReportResponse`
- `ReportDetailResponse`
- `BudgetUsageResponse`
- `RuntimeSettingsResponse`
- `RuntimeSettingsUpdateRequest`

### Appeal Analytics

- `AppealBlockItemResponse`
- `AppealAuthorGroup`
- `AppealBlockResponse`
- `AppealAnalyticsResponse`
- `AuthorCommentsResponse`

All schemas use Pydantic v2 and are directly consumed by FastAPI route handlers.
