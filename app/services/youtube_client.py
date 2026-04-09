"""YouTube Data API v3 client for fetching video metadata and comments.

Provides playlist scanning, video detail retrieval, and paginated comment
fetching with rate limiting, retries, and support for dry-run fixtures.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from isodate import parse_duration
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError, InvalidConfigurationError
from app.core.utils import write_json
from app.schemas.domain import RawComment, VideoMeta

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _parse_rfc3339(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value).astimezone(UTC)


def extract_youtube_video_id(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        msg = "Video URL is empty."
        raise ExternalServiceError(msg)
    if _YOUTUBE_ID_RE.match(candidate):
        return candidate
    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")
    if "youtu.be" in host and path:
        short_id = path.split("/")[0]
        if _YOUTUBE_ID_RE.match(short_id):
            return short_id
    if "youtube.com" in host:
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id and _YOUTUBE_ID_RE.match(query_id):
            return query_id
        for prefix in ("shorts/", "live/"):
            if path.startswith(prefix):
                path_id = path[len(prefix) :].split("/")[0]
                if _YOUTUBE_ID_RE.match(path_id):
                    return path_id
    fallback_match = re.search(r"(?:[?&]v=|youtu\.be/|shorts/|live/)([A-Za-z0-9_-]{11})", candidate)
    if fallback_match:
        return fallback_match.group(1)
    msg = "Could not parse YouTube video ID from URL."
    raise ExternalServiceError(msg)


class YouTubeClient:
    """Client for the YouTube Data API v3 with rate limiting and retries.

    Handles playlist item enumeration, video metadata lookups, and paginated
    comment thread fetching with configurable ordering and reply inclusion.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = httpx.Client(timeout=settings.youtube_request_timeout_sec)
        self._min_interval_sec = 1.0 / max(0.1, settings.youtube_rate_limit_rps)
        self._last_request_ts = 0.0
        self._title_pattern = (
            re.compile(settings.youtube_title_regex, re.IGNORECASE)
            if settings.youtube_title_regex
            else None
        )

    def close(self) -> None:
        self._client.close()

    def _ensure_config(self) -> None:
        if self.settings.dry_run:
            return
        if not self.settings.youtube_api_key:
            msg = "YOUTUBE_API_KEY is required unless DRY_RUN=true."
            raise InvalidConfigurationError(msg)
        if not self.settings.youtube_playlist_id:
            msg = "YOUTUBE_PLAYLIST_ID is required unless DRY_RUN=true."
            raise InvalidConfigurationError(msg)

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._min_interval_sec:
            time.sleep(self._min_interval_sec - elapsed)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((httpx.HTTPError, ExternalServiceError)),
    )
    def _get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        self._rate_limit()
        url = f"{YOUTUBE_API_BASE}/{endpoint}"
        params = params.copy()
        params["key"] = self.settings.youtube_api_key
        response = self._client.get(url, params=params)
        self._last_request_ts = time.monotonic()
        if response.status_code >= 500:
            msg = f"YouTube API server error {response.status_code}"
            raise ExternalServiceError(msg)
        if response.status_code >= 400:
            msg = f"YouTube API request failed {response.status_code}: {response.text}"
            raise ExternalServiceError(msg)
        return response.json()

    def _load_dry_fixture(self) -> dict[str, Any]:
        path = Path(self.settings.dry_run_fixture_path)
        if not path.exists():
            msg = f"Dry-run fixture not found: {path}"
            raise InvalidConfigurationError(msg)
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_raw(self, video_id: str, name: str, payload: dict[str, Any]) -> None:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        path = self.settings.raw_dir / day / video_id / f"{name}.json"
        write_json(path, payload)

    def get_latest_video_from_playlist(self) -> VideoMeta:
        """Find and return the most recently published video from the configured playlist.

        Scans all playlist pages, fetches video details, applies title and
        duration filters, and returns the latest matching video.

        Returns:
            VideoMeta for the most recent qualifying video.

        Raises:
            ExternalServiceError: If no videos match the configured filters.
            InvalidConfigurationError: If required API keys are missing.
        """
        self._ensure_config()
        if self.settings.dry_run:
            fixture = self._load_dry_fixture()
            video_payload = fixture["latest_video"]
            return VideoMeta.model_validate(video_payload)

        collected: list[dict[str, Any]] = []
        next_page_token: str | None = None
        pages = 0
        while pages < self.settings.youtube_max_pages:
            params: dict[str, Any] = {
                "part": "snippet,contentDetails,status",
                "playlistId": self.settings.youtube_playlist_id,
                "maxResults": 50,
            }
            if next_page_token:
                params["pageToken"] = next_page_token
            payload = self._get_json("playlistItems", params=params)
            pages += 1
            self._save_raw("playlist_scan", f"playlist_items_page_{pages}", payload)
            collected.extend(payload.get("items", []))
            next_page_token = payload.get("nextPageToken")
            if not next_page_token:
                break

        if not collected:
            msg = "No playlist items found."
            raise ExternalServiceError(msg)

        video_ids: list[str] = []
        item_by_video_id: dict[str, dict[str, Any]] = {}
        for item in collected:
            video_id = item.get("contentDetails", {}).get("videoId")
            if not video_id:
                continue
            video_ids.append(video_id)
            item_by_video_id[video_id] = item

        details = self._get_video_details(video_ids)
        if not details:
            msg = "Could not fetch video details."
            raise ExternalServiceError(msg)

        candidates: list[VideoMeta] = []
        for detail in details:
            video_id = detail.get("id", "")
            snippet = detail.get("snippet", {})
            content_details = detail.get("contentDetails", {})
            if not video_id or video_id not in item_by_video_id:
                continue
            title = snippet.get("title", "").strip()
            if self._title_pattern and not self._title_pattern.search(title):
                continue
            duration_seconds: int | None = None
            if "duration" in content_details:
                try:
                    duration_seconds = int(
                        parse_duration(content_details["duration"]).total_seconds()
                    )
                except Exception:
                    duration_seconds = None
            if (
                self.settings.youtube_exclude_shorts
                and duration_seconds is not None
                and duration_seconds < self.settings.youtube_short_threshold_seconds
            ):
                continue
            published_at = _parse_rfc3339(snippet["publishedAt"])
            candidates.append(
                VideoMeta(
                    youtube_video_id=video_id,
                    playlist_id=self.settings.youtube_playlist_id,
                    title=title,
                    description=snippet.get("description"),
                    published_at=published_at,
                    duration_seconds=duration_seconds,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                )
            )

        if not candidates:
            msg = "No candidate videos after applying filters."
            raise ExternalServiceError(msg)

        latest = max(candidates, key=lambda item: item.published_at)
        return latest

    def get_video_meta_by_url(self, video_url: str) -> VideoMeta:
        """Fetch video metadata by parsing the video ID from a URL.

        Args:
            video_url: Full YouTube video URL or bare video ID.

        Returns:
            VideoMeta for the resolved video.

        Raises:
            ExternalServiceError: If the URL cannot be parsed or the video is not found.
        """
        video_id = extract_youtube_video_id(video_url)
        return self.get_video_meta_by_id(video_id)

    def get_video_meta_by_id(self, video_id: str) -> VideoMeta:
        if self.settings.dry_run:
            fixture = self._load_dry_fixture()
            base = fixture["latest_video"]
            title = base.get("title") or f"Dry run video {video_id}"
            published_raw = base.get("published_at") or datetime.now(UTC).isoformat()
            published_at = (
                _parse_rfc3339(published_raw)
                if isinstance(published_raw, str)
                else datetime.now(UTC)
            )
            return VideoMeta(
                youtube_video_id=video_id,
                playlist_id=self.settings.youtube_playlist_id or "direct_video",
                title=title,
                description=base.get("description"),
                published_at=published_at,
                duration_seconds=base.get("duration_seconds"),
                url=f"https://www.youtube.com/watch?v={video_id}",
            )

        self._ensure_config()
        details = self._get_video_details([video_id])
        if not details:
            msg = f"Could not fetch video details for {video_id}."
            raise ExternalServiceError(msg)
        detail = details[0]
        snippet = detail.get("snippet", {})
        content_details = detail.get("contentDetails", {})
        if "publishedAt" not in snippet:
            msg = f"Video details missing publishedAt for {video_id}."
            raise ExternalServiceError(msg)
        duration_seconds: int | None = None
        if "duration" in content_details:
            try:
                duration_seconds = int(parse_duration(content_details["duration"]).total_seconds())
            except Exception:
                duration_seconds = None
        return VideoMeta(
            youtube_video_id=video_id,
            playlist_id=self.settings.youtube_playlist_id or "direct_video",
            title=snippet.get("title", "").strip() or video_id,
            description=snippet.get("description"),
            published_at=_parse_rfc3339(snippet["publishedAt"]),
            duration_seconds=duration_seconds,
            url=f"https://www.youtube.com/watch?v={video_id}",
        )

    def _get_video_details(self, video_ids: list[str]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        chunk_size = 50
        for i in range(0, len(video_ids), chunk_size):
            chunk = video_ids[i : i + chunk_size]
            payload = self._get_json(
                "videos",
                {
                    "part": "snippet,contentDetails",
                    "id": ",".join(chunk),
                    "maxResults": len(chunk),
                },
            )
            self._save_raw("playlist_scan", f"video_details_chunk_{i // chunk_size + 1}", payload)
            items.extend(payload.get("items", []))
        return items

    def fetch_comments(
        self, video_id: str, include_replies: bool | None = None
    ) -> list[RawComment]:
        """Fetch comments for a video, optionally including replies.

        Uses hybrid time/relevance ordering when configured, merging both
        samples while respecting the maximum comment limit.

        Args:
            video_id: YouTube video ID to fetch comments for.
            include_replies: Whether to include reply comments; defaults to settings.

        Returns:
            List of RawComment objects up to the configured maximum.
        """
        if include_replies is None:
            include_replies = self.settings.youtube_include_replies or self.settings.include_replies
        if self.settings.dry_run:
            fixture = self._load_dry_fixture()
            comments = fixture["comments"]
            return [RawComment.model_validate(comment) for comment in comments]

        max_comments = max(1, int(self.settings.max_comments_per_video))
        time_comments = self._fetch_comments_with_order(
            video_id=video_id,
            order="time",
            include_replies=include_replies,
            max_comments=max_comments,
            max_pages=self.settings.youtube_max_pages,
        )

        relevance_comments: list[RawComment] = []
        if self.settings.youtube_mix_relevance_comments and max_comments >= 150:
            relevance_pages = max(
                1,
                min(
                    self.settings.youtube_max_pages,
                    int(round(self.settings.youtube_max_pages * 0.7)),
                ),
            )
            relevance_comments = self._fetch_comments_with_order(
                video_id=video_id,
                order="relevance",
                include_replies=include_replies,
                max_comments=max_comments,
                max_pages=relevance_pages,
            )
            merged = self._merge_comment_samples(
                time_comments=time_comments,
                relevance_comments=relevance_comments,
                max_comments=max_comments,
            )
            self.logger.info(
                "Fetched comments for %s with hybrid fetch: time=%s relevance=%s merged=%s (cap=%s).",
                video_id,
                len(time_comments),
                len(relevance_comments),
                len(merged),
                max_comments,
            )
            return merged

        return time_comments[:max_comments]

    def _fetch_comments_with_order(
        self,
        *,
        video_id: str,
        order: str,
        include_replies: bool,
        max_comments: int,
        max_pages: int,
    ) -> list[RawComment]:
        comments: list[RawComment] = []
        next_page_token: str | None = None
        pages = 0
        while pages < max_pages and len(comments) < max_comments:
            params: dict[str, Any] = {
                "part": "snippet,replies",
                "videoId": video_id,
                "maxResults": 100,
                "textFormat": "plainText",
                "order": order,
            }
            if next_page_token:
                params["pageToken"] = next_page_token
            payload = self._get_json("commentThreads", params=params)
            pages += 1
            self._save_raw(video_id, f"comments_{order}_page_{pages}", payload)
            for item in payload.get("items", []):
                top = item.get("snippet", {}).get("topLevelComment", {})
                top_snippet = top.get("snippet", {})
                top_id = top.get("id")
                if top_id and "publishedAt" in top_snippet:
                    author_channel_id = None
                    author_channel_data = top_snippet.get("authorChannelId")
                    if author_channel_data and isinstance(author_channel_data, dict):
                        author_channel_id = author_channel_data.get("value")

                    comments.append(
                        RawComment(
                            youtube_comment_id=top_id,
                            parent_comment_id=None,
                            author_name=top_snippet.get("authorDisplayName"),
                            author_channel_id=author_channel_id,
                            text_raw=top_snippet.get("textDisplay", ""),
                            like_count=int(top_snippet.get("likeCount", 0)),
                            reply_count=int(item.get("snippet", {}).get("totalReplyCount", 0)),
                            published_at=_parse_rfc3339(top_snippet["publishedAt"]),
                            is_top_level=True,
                        )
                    )
                if include_replies:
                    reply_items = item.get("replies", {}).get("comments", [])
                    for reply in reply_items:
                        snippet = reply.get("snippet", {})
                        reply_id = reply.get("id")
                        if not reply_id or "publishedAt" not in snippet:
                            continue

                        author_channel_id = None
                        author_channel_data = snippet.get("authorChannelId")
                        if author_channel_data and isinstance(author_channel_data, dict):
                            author_channel_id = author_channel_data.get("value")

                        comments.append(
                            RawComment(
                                youtube_comment_id=reply_id,
                                parent_comment_id=top_id,
                                author_name=snippet.get("authorDisplayName"),
                                author_channel_id=author_channel_id,
                                text_raw=snippet.get("textDisplay", ""),
                                like_count=int(snippet.get("likeCount", 0)),
                                reply_count=0,
                                published_at=_parse_rfc3339(snippet["publishedAt"]),
                                is_top_level=False,
                            )
                        )
                if len(comments) >= max_comments:
                    break
            if len(comments) >= max_comments:
                break
            next_page_token = payload.get("nextPageToken")
            if not next_page_token:
                break
        return comments[:max_comments]

    def _merge_comment_samples(
        self,
        *,
        time_comments: list[RawComment],
        relevance_comments: list[RawComment],
        max_comments: int,
    ) -> list[RawComment]:
        merged: list[RawComment] = []
        seen_ids: set[str] = set()

        for comment in time_comments:
            if comment.youtube_comment_id in seen_ids:
                continue
            seen_ids.add(comment.youtube_comment_id)
            merged.append(comment)
            if len(merged) >= max_comments:
                return merged

        for comment in relevance_comments:
            if comment.youtube_comment_id in seen_ids:
                continue
            seen_ids.add(comment.youtube_comment_id)
            merged.append(comment)
            if len(merged) >= max_comments:
                break
        return merged
