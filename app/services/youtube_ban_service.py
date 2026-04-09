"""YouTube Ban Service for moderating toxic commenters.

Handles:
1. Banning users via YouTube API (if OAuth2 credentials provided)
2. Fallback: saving to CSV if API unavailable
3. Recording bans in database for tracking

YouTube API Integration:
- Uses YouTube Data API v3 Comments.setModerationStatus endpoint
- Requires OAuth2 with scope: https://www.googleapis.com/auth/youtube.force-ssl
- Set YOUTUBE_OAUTH_CLIENT_ID, YOUTUBE_OAUTH_CLIENT_SECRET, YOUTUBE_OAUTH_REFRESH_TOKEN

IMPORTANT: YouTube bans are CHANNEL-LEVEL, not video-level.
- When you ban a user via YouTube API, they are banned from commenting on ALL videos
  of the channel that owns the video, not just the specific video.
- Our database stores the source video_id for analytics/audit purposes, but the
  actual YouTube ban affects the entire channel.
- This means: if you ban a user on video A, they cannot comment on video B either
  (if both videos belong to the same channel).

OAuth Setup:
1. Create OAuth2 credentials in Google Cloud Console
2. Obtain refresh token with youtube.force-ssl scope
3. Set environment variables (see .env.example)
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import requests  # type: ignore[import-untyped]
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.utils import utcnow
from app.db.models import BannedUser, Comment, Video

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_CSV_PATH = Path("data/banned_users.csv")
_CSV_HEADERS = [
    "banned_at",
    "video_id",
    "username",
    "comment_text",
    "ban_reason",
    "confidence_score",
    "insult_target",
]


class YouTubeBanService:
    """Service for banning YouTube users (via API or CSV fallback)."""

    def __init__(self, settings: Settings, db: Session) -> None:
        self.settings = settings
        self.db = db
        self._has_oauth = bool(
            settings.youtube_oauth_client_id
            and settings.youtube_oauth_client_secret
            and settings.youtube_oauth_refresh_token
        )
        self._access_token: str | None = None
        self._owned_channel_ids: set[str] | None = None
        self._video_owner_channel_cache: dict[str, str | None] = {}

    def ban_user(
        self,
        video_id: int,
        comment_id: int,
        username: str,
        author_channel_id: str | None,
        ban_reason: str,
        confidence_score: float,
        insult_target: str | None = None,
        banned_by_admin: bool = False,
    ) -> dict[str, Any]:
        """Ban a user and record the action.

        Args:
            video_id: Internal video ID
            comment_id: Internal comment ID
            username: YouTube username (display name) for display
            author_channel_id: YouTube channel ID (stable identifier)
            ban_reason: Human-readable reason
            confidence_score: LLM confidence (0.0-1.0)
            insult_target: Target of insult (author/guest/content/undefined)
            banned_by_admin: True if manual ban, False if auto

        Returns:
            dict with ban status and details
        """
        # Check if already banned channel-wide (prefer channel_id, fallback to
        # username only when neither side has a stable channel identifier).
        existing_query = select(BannedUser)
        existing_query = existing_query.where(BannedUser.unbanned_at.is_(None))
        if author_channel_id:
            existing_query = existing_query.where(BannedUser.author_channel_id == author_channel_id)
        else:
            existing_query = existing_query.where(
                BannedUser.author_channel_id.is_(None),
                BannedUser.username == username,
            )

        existing = self.db.scalar(existing_query)
        if existing:
            logger.info(
                "User %s (channel=%s) already banned channel-wide; source video=%d",
                username,
                author_channel_id or "unknown",
                video_id,
            )
            return {
                "status": "already_banned",
                "banned_user_id": existing.id,
                "youtube_banned": existing.youtube_banned,
            }

        # Get comment text
        comment = self.db.get(Comment, comment_id)
        comment_text = comment.text_raw if comment else ""
        youtube_comment_id = comment.youtube_comment_id if comment else None

        # Try YouTube API ban if OAuth configured and we have comment ID
        youtube_banned = False
        youtube_error = None

        if self._has_oauth and youtube_comment_id:
            video = self.db.get(Video, video_id)
            youtube_video_id = video.youtube_video_id if video else None
            youtube_banned, youtube_error = self._ban_via_youtube_api(
                youtube_comment_id,
                ban_reason,
                youtube_video_id=youtube_video_id,
            )
        elif not self._has_oauth:
            youtube_error = "YouTube OAuth not configured"
        else:
            youtube_error = "Comment ID not found"

        # Fallback: save to CSV
        if not youtube_banned:
            self._save_to_csv(
                video_id, username, comment_text, ban_reason, confidence_score, insult_target
            )

        # Record in database
        banned_user = BannedUser(
            video_id=video_id,
            comment_id=comment_id,
            username=username,
            author_channel_id=author_channel_id,
            ban_reason=ban_reason,
            confidence_score=confidence_score,
            insult_target=insult_target,
            banned_at=utcnow(),
            youtube_banned=youtube_banned,
            youtube_ban_error=youtube_error,
            banned_by_admin=banned_by_admin,
        )
        self.db.add(banned_user)
        self.db.commit()
        self.db.refresh(banned_user)

        logger.info(
            "Banned user %s (channel=%s, video_id=%d, confidence=%.2f, target=%s, youtube=%s, admin=%s)",
            username,
            author_channel_id or "unknown",
            video_id,
            confidence_score,
            insult_target or "unknown",
            youtube_banned,
            banned_by_admin,
        )

        return {
            "status": "banned",
            "banned_user_id": banned_user.id,
            "youtube_banned": youtube_banned,
            "youtube_error": youtube_error,
            "csv_saved": not youtube_banned,
        }

    def unban_user(
        self,
        *,
        banned_user_id: int,
        unban_reason: str | None = None,
        unbanned_by_admin: bool = True,
    ) -> dict[str, Any]:
        """Lift a previously recorded ban.

        Best-effort behavior:
        - locally mark the ban as inactive so the user is no longer blocked by
          our moderation workflows;
        - if OAuth is configured and the source comment is known, attempt to
          republish that comment via YouTube API.

        Note: YouTube's API documents comment republishing, but does not expose
        a separate explicit "unban author" endpoint. Therefore the API step is
        treated as best effort.
        """
        banned_user = self.db.get(BannedUser, banned_user_id)
        if banned_user is None:
            return {
                "status": "not_found",
                "banned_user_id": None,
                "youtube_unbanned": False,
                "youtube_error": "Ban record not found",
            }
        if banned_user.unbanned_at is not None:
            return {
                "status": "already_unbanned",
                "banned_user_id": banned_user.id,
                "youtube_unbanned": bool(banned_user.youtube_unbanned),
                "youtube_error": banned_user.youtube_unban_error,
            }

        youtube_unbanned = False
        youtube_error: str | None = None
        comment = self.db.get(Comment, banned_user.comment_id) if banned_user.comment_id else None
        youtube_comment_id = comment.youtube_comment_id if comment else None

        if banned_user.youtube_banned and self._has_oauth and youtube_comment_id:
            video = self.db.get(Video, banned_user.video_id)
            youtube_video_id = video.youtube_video_id if video else None
            youtube_unbanned, youtube_error = self._publish_comment_via_youtube_api(
                youtube_comment_id,
                youtube_video_id=youtube_video_id,
            )
        elif banned_user.youtube_banned and not self._has_oauth:
            youtube_error = "YouTube OAuth not configured for unban"
        elif banned_user.youtube_banned and not youtube_comment_id:
            youtube_error = "Source comment ID not found for unban"

        banned_user.unbanned_at = utcnow()
        banned_user.youtube_unbanned = youtube_unbanned
        banned_user.youtube_unban_error = youtube_error
        banned_user.unbanned_by_admin = unbanned_by_admin
        banned_user.unban_reason = unban_reason
        self.db.add(banned_user)
        self.db.commit()

        logger.info(
            "Unbanned user %s (channel=%s, banned_user_id=%d, youtube_unbanned=%s)",
            banned_user.username,
            banned_user.author_channel_id or "unknown",
            banned_user.id,
            youtube_unbanned,
        )
        return {
            "status": "unbanned",
            "banned_user_id": banned_user.id,
            "youtube_unbanned": youtube_unbanned,
            "youtube_error": youtube_error,
        }

    def _get_access_token(self) -> str | None:
        """Get or refresh OAuth2 access token."""
        if self._access_token:
            return self._access_token

        if not self._has_oauth:
            return None

        # Exchange refresh token for access token
        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": self.settings.youtube_oauth_client_id,
            "client_secret": self.settings.youtube_oauth_client_secret,
            "refresh_token": self.settings.youtube_oauth_refresh_token,
            "grant_type": "refresh_token",
        }

        try:
            response = requests.post(token_url, data=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            self._access_token = data.get("access_token")
            return self._access_token
        except Exception as exc:
            logger.error("Failed to refresh OAuth access token: %s", exc)
            return None

    def _ban_via_youtube_api(
        self,
        comment_id: str,
        reason: str,
        *,
        youtube_video_id: str | None = None,
        max_retries: int = 2,
    ) -> tuple[bool, str | None]:
        """Ban user via YouTube Data API v3 Comments.setModerationStatus.

        This calls the YouTube API to:
        1. Set comment moderation status to "rejected" (hides the comment)
        2. Ban the author from the channel (banAuthor=true)

        Note: This is a CHANNEL-LEVEL ban. The user is banned from commenting
        on all videos of the channel that owns the video, not just this one video.

        Required OAuth scope: https://www.googleapis.com/auth/youtube.force-ssl

        Args:
            comment_id: YouTube comment ID (not internal DB ID)
            reason: Ban reason for logging
            max_retries: Max retry attempts after 401 (default: 2)

        Returns:
            (success: bool, error: str | None)

        Official docs:
        https://developers.google.com/youtube/v3/docs/comments/setModerationStatus
        """
        for attempt in range(max_retries):
            access_token = self._get_access_token()
            if not access_token:
                return False, "Failed to get OAuth access token"

            if youtube_video_id:
                ownership_error = self._validate_channel_ownership(
                    access_token=access_token,
                    youtube_video_id=youtube_video_id,
                )
                if ownership_error:
                    return False, ownership_error

            # YouTube Data API v3 Comments.setModerationStatus endpoint
            url = "https://www.googleapis.com/youtube/v3/comments/setModerationStatus"
            params = {
                "id": comment_id,
                "moderationStatus": "rejected",  # Options: heldForReview, published, rejected
                "banAuthor": "true",  # Must be lowercase string for Google APIs
            }
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            }

            try:
                response = requests.post(
                    f"{url}?{urlencode(params)}",
                    headers=headers,
                    timeout=10,
                )

                if response.status_code == 204:
                    # Success - no content returned
                    logger.info(
                        "Successfully banned user via YouTube API: comment_id=%s (attempt %d/%d)",
                        comment_id,
                        attempt + 1,
                        max_retries,
                    )
                    return True, None

                elif response.status_code == 401:
                    # Token expired or invalid - clear cache and retry
                    self._access_token = None
                    logger.warning(
                        "YouTube API auth failed (attempt %d/%d) - retrying with fresh token",
                        attempt + 1,
                        max_retries,
                    )
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        logger.error("YouTube API auth failed after %d attempts", max_retries)
                        return False, "OAuth token invalid or expired after retries"

                elif response.status_code == 403:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("error", {}).get("message", "Forbidden")
                    logger.error(
                        "YouTube API ban forbidden (check OAuth scopes and channel ownership): %s",
                        error_msg,
                    )
                    return False, f"API forbidden: {error_msg}"

                elif response.status_code == 404:
                    logger.error("YouTube API: comment not found: comment_id=%s", comment_id)
                    return False, "Comment not found on YouTube"

                else:
                    # Unexpected status code
                    try:
                        error_data = response.json() if response.content else {}
                        error_root = error_data.get("error", {})
                        error_msg = error_root.get("message", response.text[:200])
                        error_reason = ""
                        if isinstance(error_root.get("errors"), list) and error_root["errors"]:
                            error_reason = str(error_root["errors"][0].get("reason", "") or "")
                        if error_reason:
                            error_msg = f"{error_msg} (reason={error_reason})"
                    except Exception:
                        error_msg = response.text[:200]

                    logger.error(
                        "YouTube API ban failed with status %d: %s", response.status_code, error_msg
                    )
                    return False, f"API error {response.status_code}: {error_msg}"

            except requests.exceptions.Timeout:
                logger.error("YouTube API ban request timeout: comment_id=%s", comment_id)
                return False, "API request timeout"
            except requests.exceptions.RequestException as exc:
                logger.error("YouTube API ban request failed: %s", exc)
                return False, f"API request failed: {exc}"
            except Exception as exc:
                logger.error("Unexpected error during YouTube ban: %s", exc)
                return False, f"Unexpected error: {exc}"

        # Should not reach here
        return False, "Max retries exceeded"

    def _publish_comment_via_youtube_api(
        self,
        comment_id: str,
        *,
        youtube_video_id: str | None = None,
        max_retries: int = 2,
    ) -> tuple[bool, str | None]:
        """Republish a moderated comment via YouTube Data API."""
        for attempt in range(max_retries):
            access_token = self._get_access_token()
            if not access_token:
                return False, "Failed to get OAuth access token"

            if youtube_video_id:
                ownership_error = self._validate_channel_ownership(
                    access_token=access_token,
                    youtube_video_id=youtube_video_id,
                )
                if ownership_error:
                    return False, ownership_error

            url = "https://www.googleapis.com/youtube/v3/comments/setModerationStatus"
            params = {
                "id": comment_id,
                "moderationStatus": "published",
            }
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            }

            try:
                response = requests.post(
                    f"{url}?{urlencode(params)}",
                    headers=headers,
                    timeout=10,
                )
                if response.status_code == 204:
                    logger.info(
                        "Successfully republished comment via YouTube API: comment_id=%s (attempt %d/%d)",
                        comment_id,
                        attempt + 1,
                        max_retries,
                    )
                    return True, None
                if response.status_code == 401:
                    self._access_token = None
                    if attempt < max_retries - 1:
                        continue
                    return False, "OAuth token invalid or expired after retries"
                try:
                    error_data = response.json() if response.content else {}
                    error_root = error_data.get("error", {})
                    error_msg = error_root.get("message", response.text[:200])
                    if isinstance(error_root.get("errors"), list) and error_root["errors"]:
                        reason = str(error_root["errors"][0].get("reason", "") or "")
                        if reason:
                            error_msg = f"{error_msg} (reason={reason})"
                except Exception:
                    error_msg = response.text[:200]
                return False, f"API error {response.status_code}: {error_msg}"
            except requests.exceptions.Timeout:
                return False, "API request timeout"
            except requests.exceptions.RequestException as exc:
                return False, f"API request failed: {exc}"
            except Exception as exc:
                return False, f"Unexpected error: {exc}"

        return False, "Max retries exceeded"

    def _restore_comment_via_youtube_api(
        self,
        comment_id: str,
        *,
        youtube_video_id: str | None = None,
        max_retries: int = 2,
    ) -> tuple[bool, str | None]:
        """Best-effort restore of a previously hidden comment on YouTube."""
        for attempt in range(max_retries):
            access_token = self._get_access_token()
            if not access_token:
                return False, "Failed to get OAuth access token"

            if youtube_video_id:
                ownership_error = self._validate_channel_ownership(
                    access_token=access_token,
                    youtube_video_id=youtube_video_id,
                )
                if ownership_error:
                    return False, ownership_error

            url = "https://www.googleapis.com/youtube/v3/comments/setModerationStatus"
            params = {
                "id": comment_id,
                "moderationStatus": "published",
            }
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            }

            try:
                response = requests.post(
                    f"{url}?{urlencode(params)}",
                    headers=headers,
                    timeout=10,
                )
                if response.status_code == 204:
                    logger.info(
                        "Successfully restored YouTube comment visibility: comment_id=%s (attempt %d/%d)",
                        comment_id,
                        attempt + 1,
                        max_retries,
                    )
                    return True, None
                if response.status_code == 401:
                    self._access_token = None
                    if attempt < max_retries - 1:
                        continue
                    return False, "OAuth token invalid or expired after retries"
                if response.status_code == 403:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("error", {}).get("message", "Forbidden")
                    return False, f"API forbidden: {error_msg}"
                if response.status_code == 404:
                    return False, "Comment not found on YouTube"
                try:
                    error_data = response.json() if response.content else {}
                    error_root = error_data.get("error", {})
                    error_msg = error_root.get("message", response.text[:200])
                except Exception:
                    error_msg = response.text[:200]
                return False, f"API error {response.status_code}: {error_msg}"
            except requests.exceptions.Timeout:
                return False, "API request timeout"
            except requests.exceptions.RequestException as exc:
                return False, f"API request failed: {exc}"
            except Exception as exc:
                return False, f"Unexpected error: {exc}"

        return False, "Max retries exceeded"

    def _validate_channel_ownership(
        self,
        *,
        access_token: str,
        youtube_video_id: str | None,
    ) -> str | None:
        """Ensure OAuth token belongs to the owner of the target video/channel.

        YouTube requires comments.setModerationStatus to be authorized by the
        owner of the channel or video associated with the comment. When the
        refresh token belongs to a different channel, Google often returns
        a vague 400 processingFailure instead of a clearer permissions error.
        """
        if not youtube_video_id:
            return (
                "Cannot verify YouTube ownership: source video is missing from the local database"
            )

        owned_channel_ids = self._get_authenticated_channel_ids(access_token)
        if not owned_channel_ids:
            return "Cannot verify YouTube ownership: failed to read channels for the configured OAuth token"

        owner_channel_id = self._get_video_owner_channel_id(
            access_token=access_token,
            youtube_video_id=youtube_video_id,
        )
        if not owner_channel_id:
            return f"Cannot verify YouTube ownership: failed to resolve owner channel for video {youtube_video_id}"

        if owner_channel_id in owned_channel_ids:
            return None

        owned_text = ", ".join(sorted(owned_channel_ids))
        message = (
            "OAuth token does not own the target YouTube channel: "
            f"video {youtube_video_id} belongs to channel {owner_channel_id}, "
            f"but the configured token is authorized for channel(s) {owned_text}. "
            "comments.setModerationStatus requires authorization from the owner of the channel/video. "
            "Re-authorize YOUTUBE_OAUTH_REFRESH_TOKEN while signed into the target YouTube channel (or its Brand Account)."
        )
        logger.error(message)
        return message

    def _get_authenticated_channel_ids(self, access_token: str) -> set[str]:
        """Return channel IDs available to the current OAuth token."""
        if self._owned_channel_ids is not None:
            return self._owned_channel_ids

        url = "https://www.googleapis.com/youtube/v3/channels"
        params = {
            "part": "id",
            "mine": "true",
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.error(
                    "Failed to fetch OAuth-owned YouTube channels: status=%s body=%s",
                    response.status_code,
                    response.text[:300],
                )
                return set()
            payload = response.json()
            channel_ids = {
                str(item.get("id")) for item in payload.get("items", []) if item.get("id")
            }
            self._owned_channel_ids = channel_ids
            return channel_ids
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch OAuth-owned YouTube channels: %s", exc)
            return set()

    def _get_video_owner_channel_id(
        self,
        *,
        access_token: str,
        youtube_video_id: str,
    ) -> str | None:
        """Resolve the owning channel ID for a YouTube video."""
        if youtube_video_id in self._video_owner_channel_cache:
            return self._video_owner_channel_cache[youtube_video_id]

        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "snippet",
            "id": youtube_video_id,
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.error(
                    "Failed to fetch owner channel for video %s: status=%s body=%s",
                    youtube_video_id,
                    response.status_code,
                    response.text[:300],
                )
                self._video_owner_channel_cache[youtube_video_id] = None
                return None
            payload = response.json()
            items = payload.get("items", [])
            channel_id = None
            if items:
                channel_id = items[0].get("snippet", {}).get("channelId")
            self._video_owner_channel_cache[youtube_video_id] = channel_id
            return channel_id
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch owner channel for video %s: %s", youtube_video_id, exc)
            self._video_owner_channel_cache[youtube_video_id] = None
            return None

    def _save_to_csv(
        self,
        video_id: int,
        username: str,
        comment_text: str,
        ban_reason: str,
        confidence_score: float,
        insult_target: str | None,
    ) -> None:
        """Save ban record to CSV file."""
        _CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to write header
        file_exists = _CSV_PATH.exists()

        with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(_CSV_HEADERS)

            writer.writerow(
                [
                    datetime.now().isoformat(),
                    video_id,
                    username,
                    comment_text[:200],  # Truncate for CSV readability
                    ban_reason,
                    f"{confidence_score:.2f}",
                    insult_target or "",
                ]
            )

        logger.info("Saved ban to CSV: %s", username)

    def get_banned_users(self, video_id: int | None = None) -> list[BannedUser]:
        """Get banned users.

        When ``video_id`` is omitted, returns the full channel-wide audit list.
        """
        query = select(BannedUser)
        query = query.where(BannedUser.unbanned_at.is_(None))
        if video_id is not None:
            query = query.where(BannedUser.video_id == video_id)
        query = query.order_by(BannedUser.banned_at.desc())
        return list(self.db.scalars(query))

    def is_user_banned(
        self,
        *,
        username: str,
        author_channel_id: str | None = None,
    ) -> bool:
        """Check if a user is already banned channel-wide."""
        query = select(BannedUser.id)
        query = query.where(BannedUser.unbanned_at.is_(None))
        if author_channel_id:
            query = query.where(BannedUser.author_channel_id == author_channel_id)
        else:
            query = query.where(
                BannedUser.author_channel_id.is_(None),
                BannedUser.username == username,
            )
        return self.db.scalar(query) is not None
