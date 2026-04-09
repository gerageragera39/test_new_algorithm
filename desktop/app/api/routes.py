"""API and UI route definitions for the YouTube analyzer.

Registers all REST endpoints (health, runs, videos, reports, budget, settings)
and the SPA catch-all route under the /ui prefix.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_db_dep, get_settings_dep
from app.core.config import Settings
from app.db.models import (
    AppealBlock,
    AppealBlockItem,
    AppealRun,
    BannedUser,
    Cluster,
    ClusterItem,
    Comment,
    Report,
    Run,
    Video,
    VideoSettings,
)
from app.db.session import SessionLocal
from app.schemas.api import (
    AppealAnalyticsResponse,
    AppealAuthorGroup,
    AppealBlockItemResponse,
    AppealBlockResponse,
    AuthorCommentsResponse,
    BanUserRequest,
    BanUserResponse,
    BudgetUsageResponse,
    HealthResponse,
    QueueSnapshotResponse,
    ReportDetailResponse,
    ReportResponse,
    RunResponse,
    RuntimeSettingsResponse,
    RuntimeSettingsUpdateRequest,
    SetupRequest,
    SetupStatusResponse,
    SetupUpdateRequest,
    ToxicReviewItemResponse,
    ToxicReviewResponse,
    UnbanUserRequest,
    UnbanUserResponse,
    UpdateVideoGuestsRequest,
    VideoDetailResponse,
    VideoGuestsResponse,
    VideoItemResponse,
    VideoStatusItemResponse,
)
from app.schemas.domain import DailyBriefing, TopicPosition
from app.services.briefing import BriefingService
from app.services.budget import BudgetGovernor
from app.services.openai_endpoint import openai_base_url_host, openai_endpoint_mode
from app.services.pipeline import DailyRunService
from app.services.runtime_settings import RuntimeSettingsState, RuntimeSettingsStore
from app.services.toxic_training_service import ToxicTrainingService
from app.services.youtube_ban_service import YouTubeBanService
from desktop.bootstrap import get_setup_status, save_first_run_setup, update_setup
from desktop.paths import resource_root
from desktop.queue import get_task_queue

api_router = APIRouter()
ui_router = APIRouter()
_SPA_INDEX_FILE = resource_root() / "frontend" / "dist" / "index.html"

_OFFENSIVE_QUESTION_RE = re.compile(
    r"\b("
    r"бляд|бля|сука|сучк|хуй|хуе|пизд|ебан|ебат|мудак|дебил|идиот|мраз|твар|гандон|"
    r"fuck|fucking|shit|bitch|asshole|moron|idiot"
    r")\b",
    re.IGNORECASE,
)
_QUESTION_SIGNAL_RE = re.compile(
    r"\b(почему|зачем|как|какой|какая|какие|когда|где|что|кто|можно ли|есть ли|"
    r"why|how|what|when|where|who|is there|can you)\b",
    re.IGNORECASE,
)


def _report_to_response(report: Report, video: Video) -> ReportResponse:
    briefing = DailyBriefing.model_validate(report.structured_json)
    return ReportResponse(
        video_id=video.youtube_video_id,
        generated_at=report.created_at,
        markdown=report.content_markdown,
        html=report.content_html,
        briefing=briefing,
    )


def _build_run_status(run: Run | None) -> dict[str, Any]:
    if run is None:
        return {
            "run_status": None,
            "run_status_text": "-",
            "stage_current": 0,
            "stage_total": 0,
            "progress_pct": 0,
            "stage_label": None,
            "has_error": False,
        }
    meta = run.meta_json if isinstance(run.meta_json, dict) else {}
    stage_current = int(meta.get("stage_current", 0) or 0)
    stage_total = int(meta.get("stage_total", 0) or 0)
    stage_label = str(meta.get("stage_label", "")).strip() or None
    progress_pct = int(meta.get("progress_pct", 0) or 0)

    if run.status == "running" and stage_total > 0:
        status_text = f"running {stage_current}/{stage_total}"
        if stage_label:
            status_text = f"{status_text} - {stage_label}"
    elif run.status == "failed":
        error_tail = (run.error or "").strip()
        status_text = "failed"
        if error_tail:
            status_text = f"failed - {error_tail[:120]}"
    else:
        status_text = run.status

    return {
        "run_status": run.status,
        "run_status_text": status_text,
        "stage_current": stage_current,
        "stage_total": stage_total,
        "progress_pct": max(0, min(100, progress_pct)),
        "stage_label": stage_label,
        "has_error": run.status == "failed",
    }


def _pick_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _runtime_settings_to_response(state: RuntimeSettingsState) -> RuntimeSettingsResponse:
    return RuntimeSettingsResponse(
        beat_enabled=state.beat_enabled,
        beat_time_kyiv=state.beat_time_kyiv,
        updated_at=state.updated_at,
        author_name=state.author_name,
        openai_chat_model=state.openai_chat_model,
        embedding_mode=state.embedding_mode,
        local_embedding_model=state.local_embedding_model,
        cluster_max_count=state.cluster_max_count,
        max_comments_per_video=state.max_comments_per_video,
        youtube_include_replies=state.youtube_include_replies,
        openai_enable_polish_call=state.openai_enable_polish_call,
    )


def _setup_status_to_response() -> SetupStatusResponse:
    status = get_setup_status()
    return SetupStatusResponse(
        is_configured=status.is_configured,
        has_openai_api_key=status.has_openai_api_key,
        has_youtube_api_key=status.has_youtube_api_key,
        has_playlist_id=status.has_playlist_id,
        has_youtube_oauth_client_id=status.has_youtube_oauth_client_id,
        has_youtube_oauth_client_secret=status.has_youtube_oauth_client_secret,
        has_youtube_oauth_refresh_token=status.has_youtube_oauth_refresh_token,
        runtime_env_path=status.runtime_env_path,
    )


def _enqueue_latest_job(
    *,
    settings: Settings,
    runtime_overrides: dict[str, Any],
    skip_filtering: bool | None,
) -> dict[str, Any]:
    with SessionLocal() as job_db:
        effective_settings = settings.model_copy(update=runtime_overrides)
        service = DailyRunService(effective_settings, job_db)
        return service.run_latest(skip_filtering=skip_filtering)


def _enqueue_video_job(
    *,
    settings: Settings,
    runtime_overrides: dict[str, Any],
    video_url: str,
    skip_filtering: bool | None,
) -> dict[str, Any]:
    with SessionLocal() as job_db:
        effective_settings = settings.model_copy(update=runtime_overrides)
        service = DailyRunService(effective_settings, job_db)
        return service.run_video(video_url=video_url, skip_filtering=skip_filtering)


def _enqueue_appeal_job(
    *,
    settings: Settings,
    runtime_overrides: dict[str, Any],
    video_url: str | None,
    guest_names: list[str] | None,
) -> dict[str, Any]:
    from app.services.appeal_analytics import AppealAnalyticsService

    with SessionLocal() as job_db:
        effective_settings = settings.model_copy(update=runtime_overrides)
        service = AppealAnalyticsService(effective_settings, job_db)
        if video_url:
            return service.run_for_video_url(video_url=video_url, guest_names=guest_names)
        return service.run_for_latest(guest_names=guest_names)


def _coerce_optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
        raise HTTPException(status_code=400, detail=f"{field_name} must be a boolean.")
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise HTTPException(status_code=400, detail=f"{field_name} must be a boolean.")


def _extract_video_url_from_payload(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    return _pick_non_empty(payload.get("video_url"), payload.get("url"), payload.get("videoUrl"))


def _normalize_guest_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        raise HTTPException(
            status_code=400,
            detail="guest_names must be a list of strings or comma-separated string.",
        )

    normalized: list[str] = []
    for item in raw_items:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _extract_guest_names_from_payload(payload: dict[str, Any] | None) -> list[str] | None:
    if not isinstance(payload, dict):
        return None
    raw_value = payload.get("guest_names", payload.get("guestNames"))
    if raw_value is None:
        return None
    return _normalize_guest_names(raw_value)


async def _extract_video_url_from_request(request: Request) -> str:
    query_value = _pick_non_empty(
        request.query_params.get("video_url"), request.query_params.get("url")
    )
    if query_value:
        return query_value

    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    body = await request.body()
    if not body:
        return ""

    if content_type == "application/json":
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return ""
        return _extract_video_url_from_payload(payload if isinstance(payload, dict) else None)

    if content_type in {"application/x-www-form-urlencoded", "multipart/form-data"}:
        parsed = parse_qs(body.decode("utf-8", errors="ignore"), keep_blank_values=False)
        return _pick_non_empty(
            parsed.get("video_url", [""])[0],
            parsed.get("url", [""])[0],
            parsed.get("videoUrl", [""])[0],
        )

    return ""


def _load_video_statuses(db: Session) -> tuple[list[Video], dict[int, dict[str, Any]]]:
    videos = list(db.scalars(select(Video).order_by(Video.published_at.desc()).limit(100)))
    if not videos:
        return videos, {}

    video_ids = [v.id for v in videos]

    latest_run_ids = (
        select(func.max(Run.id)).where(Run.video_id.in_(video_ids)).group_by(Run.video_id)
    )
    latest_runs: dict[int, Run] = {
        r.video_id: r for r in db.scalars(select(Run).where(Run.id.in_(latest_run_ids)))
    }

    latest_report_ids = (
        select(func.max(Report.id)).where(Report.video_id.in_(video_ids)).group_by(Report.video_id)
    )
    latest_reports: dict[int, Report] = {
        r.video_id: r for r in db.scalars(select(Report).where(Report.id.in_(latest_report_ids)))
    }

    latest_appeal_run_ids = (
        select(func.max(AppealRun.id))
        .where(AppealRun.video_id.in_(video_ids))
        .group_by(AppealRun.video_id)
    )
    latest_appeal_runs: dict[int, AppealRun] = {
        r.video_id: r
        for r in db.scalars(select(AppealRun).where(AppealRun.id.in_(latest_appeal_run_ids)))
    }

    statuses: dict[int, dict[str, Any]] = {}
    for video in videos:
        latest_run = latest_runs.get(video.id)
        latest_report = latest_reports.get(video.id)
        latest_appeal_run = latest_appeal_runs.get(video.id)
        status = _build_run_status(latest_run)
        status["has_report"] = bool(latest_report)
        status["has_appeal_report"] = bool(
            latest_appeal_run and latest_appeal_run.status == "completed"
        )
        status["appeal_run_status"] = latest_appeal_run.status if latest_appeal_run else None
        statuses[video.id] = status
    return videos, statuses


def _load_spa_index_html() -> str:
    if not _SPA_INDEX_FILE.exists():
        msg = (
            "Frontend build is missing. Run "
            "`npm --prefix frontend install` and `npm --prefix frontend run build`."
        )
        raise HTTPException(status_code=503, detail=msg)
    return _SPA_INDEX_FILE.read_text(encoding="utf-8")


def _normalize_comment_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _is_question_comment_text(value: str) -> bool:
    normalized = _normalize_comment_text(value)
    return "?" in normalized or "¿" in normalized or "？" in normalized


def _split_topic_comment_buckets(comments: list[str]) -> tuple[list[str], list[str]]:
    main: list[str] = []
    questions: list[str] = []
    for text in comments:
        if _is_question_comment_text(text):
            questions.append(text)
        else:
            main.append(text)
    return main, questions


def _cluster_relevance_threshold(scores: list[float]) -> float:
    if not scores:
        return 0.0
    ordered = sorted(float(score) for score in scores)
    count = len(ordered)
    if count >= 80:
        quantile = 0.65
        baseline = 0.52
    elif count >= 40:
        quantile = 0.55
        baseline = 0.50
    elif count >= 20:
        quantile = 0.45
        baseline = 0.47
    else:
        quantile = 0.30
        baseline = 0.42
    idx = max(0, min(count - 1, int(round((count - 1) * quantile))))
    return max(baseline, ordered[idx])


def _filter_cluster_comments(items: list[tuple[bool, float, str]]) -> list[str]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: (item[0], item[1]), reverse=True)

    selected: list[str] = []
    seen: set[str] = set()
    for is_representative, score, text in ordered:
        _ = is_representative, score
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(text)
    return selected


def _is_offensive_question_text(value: str) -> bool:
    normalized = _normalize_comment_text(value).lower()
    if not normalized:
        return True
    return bool(_OFFENSIVE_QUESTION_RE.search(normalized))


def _question_quality_score(value: str, index: int) -> float:
    text = _normalize_comment_text(value)
    words = [word for word in text.split() if word]
    score = 0.0
    if "?" in text:
        score += 1.2
    if 6 <= len(words) <= 30:
        score += 1.0
    elif len(words) < 4:
        score -= 0.7
    if _QUESTION_SIGNAL_RE.search(text):
        score += 0.8
    if len(text) > 320:
        score -= 0.6
    if text.isupper() and len(text) > 15:
        score -= 0.6
    score -= index * 0.04
    return score


def _prioritize_question_comments(questions: list[str]) -> list[str]:
    seen: set[str] = set()
    candidates: list[tuple[float, int, str]] = []
    for idx, raw in enumerate(questions):
        text = _normalize_comment_text(raw)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        if _is_offensive_question_text(text):
            continue
        score = _question_quality_score(text, idx)
        candidates.append((score, idx, text))

    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    if not candidates:
        return []
    best = candidates[0][2]
    rest = [item[2] for item in candidates[1:]]
    return [best] + rest


def _load_topic_comments_for_report(
    db: Session,
    report: Report,
    briefing: DailyBriefing,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    topic_comments: dict[str, list[str]] = {}
    topic_question_comments: dict[str, list[str]] = {}
    clusters = list(
        db.scalars(
            select(Cluster)
            .where(Cluster.run_id == report.run_id)
            .order_by(Cluster.sort_order.asc())
        )
    )
    cluster_id_to_key = {cluster.id: cluster.cluster_key for cluster in clusters}
    cluster_ids = list(cluster_id_to_key.keys())

    if cluster_ids:
        rows = db.execute(
            select(
                ClusterItem.cluster_id,
                ClusterItem.is_representative,
                ClusterItem.score,
                Comment.text_raw,
                Comment.text_normalized,
            )
            .join(Comment, Comment.id == ClusterItem.comment_id)
            .where(ClusterItem.cluster_id.in_(cluster_ids))
        ).all()
        grouped: dict[int, list[tuple[bool, float, str]]] = {}
        for cluster_id, is_representative, score, text_raw, text_normalized in rows:
            text = _normalize_comment_text(text_raw) or _normalize_comment_text(text_normalized)
            if not text:
                continue
            grouped.setdefault(int(cluster_id), []).append(
                (bool(is_representative), float(score or 0.0), text)
            )

        for cluster_id, items in grouped.items():
            ordered = _filter_cluster_comments(items)
            cluster_key = cluster_id_to_key.get(cluster_id)
            if cluster_key:
                main, questions = _split_topic_comment_buckets(ordered)
                topic_comments[cluster_key] = main
                topic_question_comments[cluster_key] = _prioritize_question_comments(questions)

    for topic in briefing.top_topics:
        key = topic.cluster_key

        curated_main: list[str] = []
        curated_questions: list[str] = []
        seen: set[str] = set()
        for quote in topic.representative_quotes:
            text = _normalize_comment_text(quote)
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            if _is_question_comment_text(text):
                curated_questions.append(text)
            else:
                curated_main.append(text)
        for question_text in getattr(topic, "question_comments", []):
            text = _normalize_comment_text(question_text)
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            curated_questions.append(text)

        if curated_main or curated_questions:
            topic_comments[key] = curated_main
            topic_question_comments[key] = _prioritize_question_comments(curated_questions)
            continue

        if key in topic_comments:
            continue
        topic_comments[key] = []
        topic_question_comments[key] = []
    return topic_comments, topic_question_comments


def _load_topic_positions_for_report(briefing: DailyBriefing) -> dict[str, list[TopicPosition]]:
    topic_positions: dict[str, list[TopicPosition]] = {}
    for topic in briefing.top_topics:
        key = topic.cluster_key
        topic_positions[key] = list(topic.positions)
    return topic_positions


def _load_previous_briefings_for_report(
    db: Session,
    current_report: Report,
    *,
    limit: int = 6,
) -> list[DailyBriefing]:
    stmt = (
        select(Report)
        .where(Report.video_id != current_report.video_id)
        .order_by(Report.created_at.desc())
        .limit(limit)
    )
    previous_reports = db.scalars(stmt).all()
    briefings: list[DailyBriefing] = []
    for report in reversed(previous_reports):
        try:
            briefings.append(DailyBriefing.model_validate(report.structured_json))
        except Exception:
            continue
    return briefings


@api_router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings_dep)) -> HealthResponse:
    """Return application health status and configuration summary."""
    endpoint_host = openai_base_url_host(settings.openai_base_url)
    endpoint_mode = openai_endpoint_mode(settings.openai_base_url)
    return HealthResponse(
        status="ok",
        openai_endpoint_host=endpoint_host or None,
        openai_endpoint_mode=endpoint_mode,
        timestamp=datetime.now(UTC),
    )


@api_router.get("/app/setup/status", response_model=SetupStatusResponse)
def get_desktop_setup_status() -> SetupStatusResponse:
    """Return desktop first-run setup status."""
    return _setup_status_to_response()


@api_router.post("/app/setup", response_model=SetupStatusResponse)
def complete_desktop_setup(payload: SetupRequest) -> SetupStatusResponse:
    """Persist desktop setup secrets and optional moderation OAuth credentials."""
    save_first_run_setup(
        openai_api_key=payload.openai_api_key,
        youtube_api_key=payload.youtube_api_key,
        youtube_playlist_id=payload.youtube_playlist_id,
        youtube_oauth_client_id=payload.youtube_oauth_client_id,
        youtube_oauth_client_secret=payload.youtube_oauth_client_secret,
        youtube_oauth_refresh_token=payload.youtube_oauth_refresh_token,
    )
    return _setup_status_to_response()


@api_router.put("/app/setup", response_model=SetupStatusResponse)
def update_desktop_setup(payload: SetupUpdateRequest) -> SetupStatusResponse:
    """Update stored desktop secrets without requiring every field."""
    update_setup(
        openai_api_key=payload.openai_api_key,
        youtube_api_key=payload.youtube_api_key,
        youtube_playlist_id=payload.youtube_playlist_id,
        youtube_oauth_client_id=payload.youtube_oauth_client_id,
        youtube_oauth_client_secret=payload.youtube_oauth_client_secret,
        youtube_oauth_refresh_token=payload.youtube_oauth_refresh_token,
    )
    return _setup_status_to_response()


@api_router.get("/queue", response_model=QueueSnapshotResponse)
def queue_snapshot() -> QueueSnapshotResponse:
    """Return the local desktop queue snapshot."""
    return QueueSnapshotResponse(**get_task_queue().snapshot())


@api_router.post("/run/latest", response_model=RunResponse)
def run_latest(
    sync: bool = False,
    skip_filtering: str | bool | None = Query(default=None),
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> RunResponse:
    """Trigger an analysis run for the latest video, synchronously or via local queue."""
    runtime_store = RuntimeSettingsStore(settings)
    runtime_state = runtime_store.load()
    effective_settings = runtime_store.build_pipeline_settings(runtime_state)
    runtime_overrides = runtime_store.pipeline_overrides(runtime_state)
    query_skip_filtering = _coerce_optional_bool(
        skip_filtering,
        field_name="skip_filtering",
    )
    body_skip_filtering = _coerce_optional_bool(
        payload.get("skip_filtering") if isinstance(payload, dict) else None,
        field_name="skip_filtering",
    )
    resolved_skip_filtering = (
        query_skip_filtering if query_skip_filtering is not None else body_skip_filtering
    )

    if sync:
        service = DailyRunService(effective_settings, db)
        result = service.run_latest(
            skip_filtering=resolved_skip_filtering,
        )
        return RunResponse(
            task_id=f"sync-{result['run_id']}", message=f"Completed run for {result['video_id']}"
        )

    task = get_task_queue().enqueue(
        "run_latest",
        {
            "skip_filtering": resolved_skip_filtering,
            "runtime_overrides": runtime_overrides,
        },
        lambda: _enqueue_latest_job(
            settings=settings,
            runtime_overrides=runtime_overrides,
            skip_filtering=resolved_skip_filtering,
        ),
    )
    return RunResponse(task_id=task.id, message="Run triggered")


@api_router.post("/run/video", response_model=RunResponse)
def run_video(
    video_url: str | None = Query(default=None),
    sync: bool | None = Query(default=None),
    skip_filtering: str | bool | None = Query(default=None),
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> RunResponse:
    """Trigger an analysis run for a specific video URL."""
    runtime_store = RuntimeSettingsStore(settings)
    runtime_state = runtime_store.load()
    effective_settings = runtime_store.build_pipeline_settings(runtime_state)
    runtime_overrides = runtime_store.pipeline_overrides(runtime_state)
    body_url = _extract_video_url_from_payload(payload)
    resolved_video_url = _pick_non_empty(video_url, body_url)
    if not resolved_video_url:
        raise HTTPException(
            status_code=422,
            detail="video_url is required (query param or JSON body field: video_url).",
        )

    body_sync = payload.get("sync") if isinstance(payload, dict) else None
    resolved_sync = bool(sync if sync is not None else body_sync)

    query_skip_filtering = _coerce_optional_bool(
        skip_filtering,
        field_name="skip_filtering",
    )
    body_skip_filtering = _coerce_optional_bool(
        payload.get("skip_filtering") if isinstance(payload, dict) else None,
        field_name="skip_filtering",
    )
    resolved_skip_filtering = (
        query_skip_filtering if query_skip_filtering is not None else body_skip_filtering
    )

    if resolved_sync:
        service = DailyRunService(effective_settings, db)
        result = service.run_video(
            video_url=resolved_video_url,
            skip_filtering=resolved_skip_filtering,
        )
        return RunResponse(
            task_id=f"sync-{result['run_id']}", message=f"Completed run for {result['video_id']}"
        )

    task = get_task_queue().enqueue(
        "run_video",
        {
            "video_url": resolved_video_url,
            "skip_filtering": resolved_skip_filtering,
            "runtime_overrides": runtime_overrides,
        },
        lambda: _enqueue_video_job(
            settings=settings,
            runtime_overrides=runtime_overrides,
            video_url=resolved_video_url,
            skip_filtering=resolved_skip_filtering,
        ),
    )
    return RunResponse(task_id=task.id, message="Video run triggered")


@api_router.get("/videos", response_model=list[VideoItemResponse])
def list_videos(db: Session = Depends(get_db_dep)) -> list[VideoItemResponse]:
    """Return a list of the most recent videos with their run statuses."""
    videos = list(db.scalars(select(Video).order_by(Video.published_at.desc()).limit(100)))
    if not videos:
        return []

    video_ids = [v.id for v in videos]

    latest_run_ids = (
        select(func.max(Run.id)).where(Run.video_id.in_(video_ids)).group_by(Run.video_id)
    )
    latest_runs: dict[int, Run] = {
        r.video_id: r for r in db.scalars(select(Run).where(Run.id.in_(latest_run_ids)))
    }

    latest_report_ids = (
        select(func.max(Report.id)).where(Report.video_id.in_(video_ids)).group_by(Report.video_id)
    )
    video_ids_with_report: set[int] = {
        r.video_id for r in db.scalars(select(Report).where(Report.id.in_(latest_report_ids)))
    }

    response: list[VideoItemResponse] = []
    for video in videos:
        latest_run = latest_runs.get(video.id)
        response.append(
            VideoItemResponse(
                youtube_video_id=video.youtube_video_id,
                title=video.title,
                published_at=video.published_at,
                has_report=video.id in video_ids_with_report,
                latest_run_status=latest_run.status if latest_run else None,
            )
        )
    return response


@api_router.get("/videos/statuses", response_model=list[VideoStatusItemResponse])
def list_video_statuses(db: Session = Depends(get_db_dep)) -> list[VideoStatusItemResponse]:
    """Return videos with detailed run progress and stage information."""
    videos, statuses = _load_video_statuses(db)
    response: list[VideoStatusItemResponse] = []
    for video in videos:
        status = statuses.get(video.id, {})
        response.append(
            VideoStatusItemResponse(
                video_id=video.id,
                youtube_video_id=video.youtube_video_id,
                title=video.title,
                published_at=video.published_at,
                has_report=bool(status.get("has_report", False)),
                has_appeal_report=bool(status.get("has_appeal_report", False)),
                run_status=status.get("run_status"),
                run_status_text=str(status.get("run_status_text") or "-"),
                stage_current=int(status.get("stage_current", 0) or 0),
                stage_total=int(status.get("stage_total", 0) or 0),
                progress_pct=int(status.get("progress_pct", 0) or 0),
                stage_label=status.get("stage_label"),
                has_error=bool(status.get("has_error", False)),
                appeal_run_status=status.get("appeal_run_status"),
            )
        )
    return response


@api_router.get("/videos/{video_id}", response_model=VideoDetailResponse)
def get_video(video_id: str, db: Session = Depends(get_db_dep)) -> VideoDetailResponse:
    """Return detailed information for a single video by its YouTube ID."""
    video = db.scalar(select(Video).where(Video.youtube_video_id == video_id))
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    latest_run = db.scalar(
        select(Run).where(Run.video_id == video.id).order_by(Run.created_at.desc()).limit(1)
    )
    has_report = bool(
        db.scalar(
            select(Report.id)
            .where(Report.video_id == video.id)
            .order_by(Report.created_at.desc())
            .limit(1)
        )
    )
    return VideoDetailResponse(
        youtube_video_id=video.youtube_video_id,
        title=video.title,
        description=video.description,
        published_at=video.published_at,
        duration_seconds=video.duration_seconds,
        report_available=has_report,
        latest_run_status=latest_run.status if latest_run else None,
    )


@api_router.get("/reports/latest", response_model=ReportResponse)
def get_latest_report(db: Session = Depends(get_db_dep)) -> ReportResponse:
    """Return the most recently generated report across all videos."""
    report = db.scalar(select(Report).order_by(Report.created_at.desc()).limit(1))
    if not report:
        raise HTTPException(status_code=404, detail="No reports available")
    video = db.get(Video, report.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return _report_to_response(report, video)


@api_router.get("/reports/{video_id}", response_model=ReportResponse)
def get_report(video_id: str, db: Session = Depends(get_db_dep)) -> ReportResponse:
    """Return the latest report for a specific video."""
    video = db.scalar(select(Video).where(Video.youtube_video_id == video_id))
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    report = db.scalar(
        select(Report)
        .where(Report.video_id == video.id)
        .order_by(Report.created_at.desc())
        .limit(1)
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found for video")
    return _report_to_response(report, video)


@api_router.get("/reports/{video_id}/detail", response_model=ReportDetailResponse)
def get_report_detail(
    video_id: str,
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> ReportDetailResponse:
    """Return an enriched report with per-topic comments and position breakdowns."""
    video = db.scalar(select(Video).where(Video.youtube_video_id == video_id))
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    report = db.scalar(
        select(Report)
        .where(Report.video_id == video.id)
        .order_by(Report.created_at.desc())
        .limit(1)
    )
    if not report:
        raise HTTPException(status_code=404, detail="Report not found for video")
    briefing = DailyBriefing.model_validate(report.structured_json)
    topic_comments, topic_question_comments = _load_topic_comments_for_report(db, report, briefing)
    topic_positions = _load_topic_positions_for_report(briefing)
    previous_briefings = _load_previous_briefings_for_report(db, report)
    topic_trends = BriefingService(settings).build_topic_trend_series(briefing, previous_briefings)
    return ReportDetailResponse(
        report=_report_to_response(report, video),
        author_name=settings.author_name,
        topic_comments=topic_comments,
        topic_question_comments=topic_question_comments,
        topic_positions=topic_positions,
        topic_trends=topic_trends,
    )


@api_router.post("/appeal/run", response_model=RunResponse)
async def run_appeal_analytics(
    request: Request,
    video_url: str | None = Query(default=None),
    sync: bool | None = Query(default=None),
    payload: dict[str, Any] | None = Body(default=None),
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> RunResponse:
    """Trigger appeal analytics pipeline for a video URL or the latest video."""
    body_url = _extract_video_url_from_payload(payload)
    resolved_video_url = _pick_non_empty(video_url, body_url)
    resolved_guest_names = _extract_guest_names_from_payload(payload)

    body_sync = payload.get("sync") if isinstance(payload, dict) else None
    resolved_sync = bool(sync if sync is not None else body_sync)

    if resolved_sync:
        from app.services.appeal_analytics import AppealAnalyticsService

        service = AppealAnalyticsService(settings, db)
        if resolved_video_url:
            result = service.run_for_video_url(
                video_url=resolved_video_url,
                guest_names=resolved_guest_names,
            )
        else:
            result = service.run_for_latest(guest_names=resolved_guest_names)
        return RunResponse(
            task_id=f"sync-appeal-{result.get('appeal_run_id', 0)}",
            message=f"Completed appeal analytics for {result.get('video_id', '')}",
        )

    runtime_store = RuntimeSettingsStore(settings)
    runtime_state = runtime_store.load()
    runtime_overrides = runtime_store.pipeline_overrides(runtime_state)
    task = get_task_queue().enqueue(
        "run_appeal_analytics",
        {
            "video_url": resolved_video_url or None,
            "guest_names": resolved_guest_names,
            "runtime_overrides": runtime_overrides,
        },
        lambda: _enqueue_appeal_job(
            settings=settings,
            runtime_overrides=runtime_overrides,
            video_url=resolved_video_url or None,
            guest_names=resolved_guest_names,
        ),
    )
    return RunResponse(task_id=task.id, message="Appeal analytics triggered")


@api_router.get("/appeal/{video_id}", response_model=AppealAnalyticsResponse)
def get_appeal_analytics(
    video_id: str,
    db: Session = Depends(get_db_dep),
) -> AppealAnalyticsResponse:
    """Return appeal analytics results for a video."""
    block_order = {
        "constructive_question": 1,
        "constructive_criticism": 2,
        "author_appeal": 3,
        "toxic_auto_banned": 4,
        "toxic_manual_review": 5,
    }
    video = db.scalar(select(Video).where(Video.youtube_video_id == video_id))
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    appeal_run = db.scalar(
        select(AppealRun)
        .where(AppealRun.video_id == video.id, AppealRun.status == "completed")
        .order_by(AppealRun.created_at.desc())
        .limit(1)
    )
    if not appeal_run:
        raise HTTPException(status_code=404, detail="Appeal analytics not found for video")

    blocks_db = list(
        db.scalars(
            select(AppealBlock)
            .where(AppealBlock.appeal_run_id == appeal_run.id)
            .order_by(AppealBlock.sort_order.asc())
        )
    )
    # Filter out legacy spam blocks — spam classification was removed and
    # must not appear in API responses even for old DB records.
    blocks_db = [b for b in blocks_db if b.block_type != "spam"]

    # Batch-load all block items and comments in 2 queries instead of 2*N
    block_ids = [b.id for b in blocks_db]
    all_items_db = (
        list(db.scalars(select(AppealBlockItem).where(AppealBlockItem.block_id.in_(block_ids))))
        if block_ids
        else []
    )
    block_by_id = {block.id: block for block in blocks_db}
    all_comment_ids = [item.comment_id for item in all_items_db]
    comments_map: dict[int, Comment] = (
        {c.id: c for c in db.scalars(select(Comment).where(Comment.id.in_(all_comment_ids)))}
        if all_comment_ids
        else {}
    )
    auto_ban_comment_ids = [
        item.comment_id
        for item in all_items_db
        if block_by_id.get(item.block_id) is not None
        and block_by_id[item.block_id].block_type == "toxic_auto_banned"
    ]
    banned_by_comment_id: dict[int, BannedUser] = {}
    if auto_ban_comment_ids:
        banned_rows = list(
            db.scalars(
                select(BannedUser)
                .where(BannedUser.comment_id.in_(auto_ban_comment_ids))
                .where(BannedUser.unbanned_at.is_(None))
                .order_by(BannedUser.banned_at.desc())
            )
        )
        for banned_row in banned_rows:
            if banned_row.comment_id is not None:
                banned_by_comment_id.setdefault(banned_row.comment_id, banned_row)
    items_by_block: dict[int, list[AppealBlockItem]] = {}
    for item in all_items_db:
        items_by_block.setdefault(item.block_id, []).append(item)

    blocks: list[AppealBlockResponse] = []
    for block_db in blocks_db:
        # Skip legacy spam blocks — spam detection was removed; never expose it in output
        if block_db.block_type == "spam":
            continue
        items_db = items_by_block.get(block_db.id, [])
        # Build items with text and score
        api_items: list[AppealBlockItemResponse] = []
        for item in items_db:
            comment = comments_map.get(item.comment_id)
            detail = item.detail_json if isinstance(item.detail_json, dict) else {}
            raw_score = detail.get("score")
            score = int(raw_score) if raw_score is not None else None
            api_items.append(
                AppealBlockItemResponse(
                    comment_id=item.comment_id,
                    author_name=item.author_name,
                    text=_normalize_comment_text(comment.text_raw if comment else ""),
                    score=score,
                    author_channel_id=comment.author_channel_id if comment else None,
                )
            )

        # Sort criticism/question blocks by score descending
        if block_db.block_type in ("constructive_criticism", "constructive_question"):
            api_items.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)

        # Group by author for toxic block
        if block_db.block_type == "toxic_auto_banned":
            by_author: dict[str, list[AppealBlockItemResponse]] = {}
            author_meta: dict[str, dict[str, Any]] = {}
            for item in api_items:
                name = item.author_name or "Unknown"
                comment = comments_map.get(item.comment_id)
                group_key = comment.author_channel_id or f"name:{name}"
                by_author.setdefault(group_key, []).append(item)
                if group_key not in author_meta:
                    banned_user = banned_by_comment_id.get(item.comment_id)
                    author_meta[group_key] = {
                        "author_name": name,
                        "author_channel_id": comment.author_channel_id if comment else None,
                        "banned_user_id": banned_user.id if banned_user else None,
                        "youtube_banned": bool(banned_user.youtube_banned) if banned_user else False,
                    }
            authors = [
                AppealAuthorGroup(
                    author_name=meta["author_name"],
                    author_channel_id=meta["author_channel_id"],
                    banned_user_id=meta["banned_user_id"],
                    is_banned_active=meta["banned_user_id"] is not None,
                    youtube_banned=meta["youtube_banned"],
                    comment_count=len(items),
                    comments=items,
                )
                for key, items in sorted(by_author.items(), key=lambda x: -len(x[1]))
                for meta in [author_meta[key]]
            ]
            blocks.append(
                AppealBlockResponse(
                    block_type=block_db.block_type,
                    display_label=block_db.display_label,
                    sort_order=block_order.get(block_db.block_type, block_db.sort_order),
                    item_count=block_db.item_count,
                    authors=authors,
                )
            )
        else:
            blocks.append(
                AppealBlockResponse(
                    block_type=block_db.block_type,
                    display_label=block_db.display_label,
                    sort_order=block_order.get(block_db.block_type, block_db.sort_order),
                    item_count=block_db.item_count,
                    items=api_items,
                )
            )

    blocks.sort(key=lambda block: (block.sort_order, block.display_label.lower()))

    return AppealAnalyticsResponse(
        video_id=video.youtube_video_id,
        video_title=video.title,
        generated_at=appeal_run.created_at,
        total_comments=appeal_run.total_comments,
        classified_comments=appeal_run.processed_comments,
        blocks=blocks,
    )


@api_router.get("/appeal/{video_id}/author/{author_name}", response_model=AuthorCommentsResponse)
def get_author_comments(
    video_id: str,
    author_name: str,
    db: Session = Depends(get_db_dep),
) -> AuthorCommentsResponse:
    """Return all comments by a specific author under a video."""
    video = db.scalar(select(Video).where(Video.youtube_video_id == video_id))
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    comments = list(
        db.scalars(
            select(Comment).where(
                Comment.video_id == video.id,
                Comment.author_name == author_name,
            )
        )
    )

    return AuthorCommentsResponse(
        author_name=author_name,
        video_id=video.youtube_video_id,
        comments=[
            AppealBlockItemResponse(
                comment_id=c.id,
                author_name=c.author_name,
                text=_normalize_comment_text(c.text_raw),
            )
            for c in comments
        ],
    )


@api_router.get("/appeal/{video_id}/toxic-review", response_model=ToxicReviewResponse)
def get_toxic_review(
    video_id: str,
    db: Session = Depends(get_db_dep),
) -> ToxicReviewResponse:
    """Return toxic comments requiring manual review (medium confidence)."""
    video = db.scalar(select(Video).where(Video.youtube_video_id == video_id))
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Get latest COMPLETED appeal run (ignore running/failed runs)
    appeal_run = db.scalar(
        select(AppealRun)
        .where(
            AppealRun.video_id == video.id,
            AppealRun.status == "completed",
        )
        .order_by(AppealRun.created_at.desc())
    )
    if not appeal_run:
        raise HTTPException(status_code=404, detail="No completed appeal analytics found")

    # Get toxic_manual_review block
    review_block = db.scalar(
        select(AppealBlock).where(
            AppealBlock.appeal_run_id == appeal_run.id,
            AppealBlock.block_type == "toxic_manual_review",
        )
    )

    if not review_block:
        return ToxicReviewResponse(
            video_id=video.youtube_video_id,
            video_title=video.title,
            total_review_items=0,
            items=[],
        )

    # Get all items in this block with their comments
    # Exclude already banned users from review (prefer channel_id, fallback to username)
    banned_channel_ids = set(
        db.scalars(
            select(BannedUser.author_channel_id)
            .where(BannedUser.author_channel_id.isnot(None))
            .where(BannedUser.unbanned_at.is_(None))
            .distinct()
        )
    )

    banned_fallback_usernames = set(
        db.scalars(
            select(BannedUser.username)
            .where(BannedUser.author_channel_id.is_(None))
            .where(BannedUser.unbanned_at.is_(None))
            .distinct()
        )
    )

    items_query = (
        select(AppealBlockItem, Comment)
        .join(Comment, AppealBlockItem.comment_id == Comment.id)
        .where(AppealBlockItem.block_id == review_block.id)
        .order_by(AppealBlockItem.confidence_score.desc())
    )

    items = []
    for block_item, comment in db.execute(items_query):
        # Skip if user is already banned (check channel_id first, then username)
        if comment.author_channel_id and comment.author_channel_id in banned_channel_ids:
            continue
        if (
            comment.author_channel_id is None
            and comment.author_name
            and comment.author_name in banned_fallback_usernames
        ):
            continue

        items.append(
            ToxicReviewItemResponse(
                comment_id=comment.id,
                author_name=comment.author_name,
                text=_normalize_comment_text(comment.text_raw),
                confidence_score=block_item.confidence_score or 0.0,
                insult_target=block_item.insult_target,
            )
        )

    return ToxicReviewResponse(
        video_id=video.youtube_video_id,
        video_title=video.title,
        total_review_items=len(items),
        items=items,
    )


@api_router.post("/appeal/ban-user", response_model=BanUserResponse)
def ban_user(
    request: BanUserRequest,
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> BanUserResponse:
    """Ban a user manually from the admin review panel."""
    video = db.scalar(select(Video).where(Video.youtube_video_id == request.video_id))
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    comment = db.get(Comment, request.comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    # Get block item to extract confidence and target
    block_item = db.scalar(
        select(AppealBlockItem).where(AppealBlockItem.comment_id == request.comment_id)
    )
    confidence_score = block_item.confidence_score if block_item else 0.5
    insult_target = block_item.insult_target if block_item else "undefined"

    # Ban the user (use author info from DB, not from client request)
    ban_service = YouTubeBanService(settings, db)
    result = ban_service.ban_user(
        video_id=video.id,
        comment_id=request.comment_id,
        username=comment.author_name or "unknown",
        author_channel_id=comment.author_channel_id,
        ban_reason=request.ban_reason or f"Ручной бан админа: {insult_target}",
        confidence_score=confidence_score,
        insult_target=insult_target,
        banned_by_admin=True,
    )

    # Save to training dataset
    training_service = ToxicTrainingService(db)
    training_service.save_toxic_label(
        comment_id=request.comment_id,
        video_id=video.id,
        is_toxic=True,
        confidence_score=confidence_score,
        insult_target=insult_target,
        labeled_by="admin",
    )

    return BanUserResponse(
        status=result["status"],
        banned_user_id=result.get("banned_user_id"),
        youtube_banned=result.get("youtube_banned", False),
        youtube_error=result.get("youtube_error"),
        csv_saved=result.get("csv_saved", False),
    )


@api_router.post("/appeal/unban-user", response_model=UnbanUserResponse)
def unban_user(
    request: UnbanUserRequest,
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> UnbanUserResponse:
    """Lift a previously recorded ban and restore local access."""
    ban_service = YouTubeBanService(settings, db)
    result = ban_service.unban_user(
        banned_user_id=request.banned_user_id,
        unban_reason=request.unban_reason or "Разбанен админом",
        unbanned_by_admin=True,
    )
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Ban record not found")

    banned_user = db.get(BannedUser, request.banned_user_id)
    if banned_user and banned_user.comment_id is not None:
        training_service = ToxicTrainingService(db)
        training_service.save_toxic_label(
            comment_id=banned_user.comment_id,
            video_id=banned_user.video_id,
            is_toxic=False,
            confidence_score=max(0.0, float(banned_user.confidence_score)),
            insult_target=banned_user.insult_target,
            labeled_by="admin",
        )

    return UnbanUserResponse(
        status=result["status"],
        banned_user_id=result.get("banned_user_id"),
        youtube_unbanned=result.get("youtube_unbanned", False),
        youtube_error=result.get("youtube_error"),
    )
@api_router.get("/settings/video-guests/{video_id}", response_model=VideoGuestsResponse)
def get_video_guests(
    video_id: str,
    db: Session = Depends(get_db_dep),
) -> VideoGuestsResponse:
    """Get guest names for a specific video."""
    video = db.scalar(select(Video).where(Video.youtube_video_id == video_id))
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_settings = db.scalar(select(VideoSettings).where(VideoSettings.video_id == video.id))

    guest_names = []
    if video_settings and video_settings.guest_names:
        guest_names = [
            name.strip() for name in video_settings.guest_names.split(",") if name.strip()
        ]

    return VideoGuestsResponse(
        video_id=video.youtube_video_id,
        guest_names=guest_names,
    )


@api_router.put("/settings/video-guests/{video_id}", response_model=VideoGuestsResponse)
def update_video_guests(
    video_id: str,
    request: UpdateVideoGuestsRequest,
    db: Session = Depends(get_db_dep),
) -> VideoGuestsResponse:
    """Update guest names for a specific video."""
    video = db.scalar(select(Video).where(Video.youtube_video_id == video_id))
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Get or create video settings
    video_settings = db.scalar(select(VideoSettings).where(VideoSettings.video_id == video.id))

    guest_names_str = ", ".join(request.guest_names)

    if video_settings:
        video_settings.guest_names = guest_names_str
        db.add(video_settings)
    else:
        video_settings = VideoSettings(
            video_id=video.id,
            guest_names=guest_names_str,
        )
        db.add(video_settings)

    db.commit()

    return VideoGuestsResponse(
        video_id=video.youtube_video_id,
        guest_names=request.guest_names,
    )


@api_router.get("/budget", response_model=BudgetUsageResponse)
def get_budget_usage(
    db: Session = Depends(get_db_dep),
    settings: Settings = Depends(get_settings_dep),
) -> BudgetUsageResponse:
    """Return the current day's API budget usage snapshot."""
    governor = BudgetGovernor(settings, db)
    snapshot = governor.get_snapshot()
    return BudgetUsageResponse(
        usage_date=snapshot.usage_date.isoformat(),
        spent_usd=snapshot.spent_usd,
        tokens_used=snapshot.tokens_used,
        entries=snapshot.entries,
    )


@api_router.get("/settings/runtime", response_model=RuntimeSettingsResponse)
def get_runtime_settings(settings: Settings = Depends(get_settings_dep)) -> RuntimeSettingsResponse:
    """Return the current runtime settings state."""
    store = RuntimeSettingsStore(settings)
    return _runtime_settings_to_response(store.load())


@api_router.put("/settings/runtime", response_model=RuntimeSettingsResponse)
def update_runtime_settings(
    payload: RuntimeSettingsUpdateRequest,
    settings: Settings = Depends(get_settings_dep),
) -> RuntimeSettingsResponse:
    """Apply a partial update to runtime settings and return the new state."""
    store = RuntimeSettingsStore(settings)
    patch = payload.model_dump(exclude_none=True)
    state = store.save_patch(patch)
    return _runtime_settings_to_response(state)


@ui_router.get("/", include_in_schema=False)
def ui_root() -> RedirectResponse:
    """Redirect the root URL to the SPA UI."""
    return RedirectResponse(url="/ui", status_code=307)


@ui_router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
@ui_router.get("/ui/{full_path:path}", response_class=HTMLResponse, include_in_schema=False)
def ui_spa(full_path: str = "") -> HTMLResponse:
    """Serve the SPA frontend index.html for all UI routes."""
    _ = full_path
    return HTMLResponse(_load_spa_index_html())
