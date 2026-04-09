# API Requests Reference

Frontend/backend integration reference for the current FastAPI application.

> Last verified against `app/api/routes.py` and `app/schemas/api.py` on 2026-04-07.

## Base rules

- Base URL: `http://localhost:8000`
- No `/api` prefix is used.
- API auth is currently not required.
- JSON requests should use `Content-Type: application/json`.
- Typical error shape:

```json
{
  "detail": "Error message"
}
```

## Route map

### Core health, runs, reports

| Method | Endpoint | Notes |
|---|---|---|
| `GET` | `/health` | backend health + OpenAI endpoint mode |
| `POST` | `/run/latest` | trigger Topic Intelligence for the latest playlist video |
| `POST` | `/run/video` | trigger Topic Intelligence for a specific video URL |
| `GET` | `/videos` | list recent videos |
| `GET` | `/videos/statuses` | list videos with run progress |
| `GET` | `/videos/{video_id}` | get one video by YouTube ID |
| `GET` | `/reports/latest` | latest available topic report |
| `GET` | `/reports/{video_id}` | latest topic report for a video |
| `GET` | `/reports/{video_id}/detail` | enriched report with comments and positions |

### Appeal analytics and moderation

| Method | Endpoint | Notes |
|---|---|---|
| `POST` | `/appeal/run` | trigger Appeal Analytics |
| `GET` | `/appeal/{video_id}` | get latest completed appeal result |
| `GET` | `/appeal/{video_id}/author/{author_name}` | author drill-down |
| `GET` | `/appeal/{video_id}/toxic-review` | toxic comments queued for manual review |
| `POST` | `/appeal/ban-user` | manual moderation action |

### Runtime and settings

| Method | Endpoint | Notes |
|---|---|---|
| `GET` | `/settings/video-guests/{video_id}` | read guest names used by toxic targeting |
| `PUT` | `/settings/video-guests/{video_id}` | update guest names for a video |
| `GET` | `/budget` | budget / usage snapshot |
| `GET` | `/settings/runtime` | current mutable runtime settings |
| `PUT` | `/settings/runtime` | update mutable runtime settings |

---

## Common request patterns

### `POST /run/latest`

Optional body:

```json
{
  "skip_filtering": true
}
```

Behavior:
- `sync=false` by default;
- `skip_filtering` can be passed via query or body;
- query value wins over body value when both are present.

Response:

```json
{
  "task_id": "task-or-sync-id",
  "message": "Run triggered"
}
```

### `POST /run/video`

Required input: `video_url` (query or JSON body).

Example body:

```json
{
  "video_url": "https://www.youtube.com/watch?v=example",
  "skip_filtering": false
}
```

### `POST /appeal/run`

Example body:

```json
{
  "video_url": "https://www.youtube.com/watch?v=example",
  "guest_names": ["Guest One", "Guest Two"]
}
```

If `video_url` is omitted, the backend uses the latest video.

### `PUT /settings/runtime`

Supported mutable fields:

```json
{
  "beat_enabled": true,
  "beat_time_kyiv": "11:20",
  "author_name": "Channel Host",
  "openai_chat_model": "gpt-4o-mini",
  "embedding_mode": "openai",
  "cluster_max_count": 12,
  "max_comments_per_video": 2000,
  "youtube_include_replies": false,
  "moderation_enabled": true,
  "openai_enable_polish_call": true
}
```

---

## Response highlights

### Appeal analytics response

Persisted blocks currently include:

- `constructive_criticism`
- `constructive_question`
- `author_appeal`
- `toxic_auto_banned`
- `toxic_manual_review`

`skip` is filtered out and is not persisted as a block.

### Runtime settings response

Current shape includes:

- `beat_enabled`
- `beat_time_kyiv`
- `updated_at`
- `author_name`
- `openai_chat_model`
- `embedding_mode`
- `cluster_max_count`
- `max_comments_per_video`
- `youtube_include_replies`
- `moderation_enabled`
- `openai_enable_polish_call`

---

## UI-serving note

This file documents JSON API endpoints only.

The FastAPI app also serves:
- the SPA shell at `/ui`
- built frontend assets at `/static/app`
