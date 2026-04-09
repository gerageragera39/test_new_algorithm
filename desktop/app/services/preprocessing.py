"""Comment preprocessing pipeline for filtering, deduplication, and moderation.

Transforms raw YouTube comments into weighted, normalized, and moderated
ProcessedComment objects ready for downstream clustering and analysis.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langdetect import DetectorFactory, LangDetectException, detect

from app.core.config import Settings
from app.core.utils import hash_text, looks_like_noise, normalize_text
from app.schemas.domain import ProcessedComment, RawComment, VideoMeta

DetectorFactory.seed = 0

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_URL_ONLY_RE = re.compile(r"^(https?://\S+|www\.\S+)$", re.IGNORECASE)
_REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}", re.UNICODE)
_SPAM_HINT_RE = re.compile(
    r"(t\.me/|wa\.me/|telegram|телеграм|промокод|promo code|подпиш(ись|итесь)|заработа|реферал|referral|мой канал)",
    re.IGNORECASE,
)
_PROFANITY_RE = re.compile(
    r"\b("
    r"\u0431\u043b\u044f\u0434|\u0431\u043b\u044f|\u0441\u0443\u043a\u0430|\u0441\u0443\u0447\u043a|"
    r"\u0445\u0443\u0439|\u0445\u0443\u0435|\u043f\u0438\u0437\u0434|\u0435\u0431\u0430\u043d|\u0435\u0431\u0430\u0442|"
    r"\u043c\u0443\u0434\u0430\u043a|\u0434\u0435\u0431\u0438\u043b|\u0438\u0434\u0438\u043e\u0442|\u0442\u0432\u0430\u0440\u044c|\u0443\u0431\u043b\u044e\u0434"
    r"|fuck|fucking|shit|bitch|asshole|moron|idiot"
    r")\b",
    re.IGNORECASE,
)
_CLAIM_SIGNAL_RE = re.compile(
    r"\b("
    r"\u043f\u043e\u0442\u043e\u043c\u0443|\u043f\u043e\u044d\u0442\u043e\u043c\u0443|\u0441\u0447\u0438\u0442\u0430\u044e|\u0434\u0443\u043c\u0430\u044e|\u043f\u043e\u0434\u0434\u0435\u0440\u0436\u0438\u0432\u0430\u044e|"
    r"\u043f\u0440\u043e\u0442\u0438\u0432|\u043d\u0435 \u0441\u043e\u0433\u043b\u0430\u0441|\u043b\u043e\u0436|\u043f\u0440\u0430\u0432|\u0432\u0438\u043d\u043e\u0432|"
    r"\u043d\u0443\u0436\u043d\u043e|\u0434\u043e\u043b\u0436\u043d|\u043d\u0430\u0434\u043e"
    r")\b",
    re.IGNORECASE,
)
_TOPIC_STOPWORDS = {
    "\u044d\u0442\u043e",
    "\u043a\u0430\u043a",
    "\u0447\u0442\u043e",
    "\u0434\u043b\u044f",
    "\u0438\u043b\u0438",
    "\u0435\u0441\u043b\u0438",
    "\u043f\u0440\u043e",
    "\u0432\u0441\u0435",
    "\u043f\u0440\u043e\u0441\u0442\u043e",
    "\u043e\u0447\u0435\u043d\u044c",
    "this",
    "that",
    "with",
    "from",
    "have",
    "about",
}


@dataclass
class PreprocessResult:
    processed: list[ProcessedComment]
    all_comments: list[ProcessedComment]
    filtered_count: int
    total_count: int
    dropped_count: int = 0
    flagged_count: int = 0
    kept_count: int = 0
    dropped_by_reason: dict[str, int] = field(default_factory=dict)
    flagged_by_reason: dict[str, int] = field(default_factory=dict)
    llm_moderation_stats: dict[str, int | bool | str] = field(default_factory=dict)
    borderline_comment_ids: list[str] = field(default_factory=list)


class CommentPreprocessor:
    """Filters, normalizes, deduplicates, and moderates raw YouTube comments.

    Applies rule-based moderation, spam detection, profanity filtering, and
    weight scoring to produce a clean set of comments for analysis.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess(self, comments: list[RawComment], video: VideoMeta) -> PreprocessResult:
        """Run the full preprocessing pipeline on a list of raw comments.

        Args:
            comments: Raw comments fetched from YouTube.
            video: Metadata of the video the comments belong to.

        Returns:
            PreprocessResult containing processed, filtered, and flagged comments
            along with aggregated statistics.
        """
        canonical_index_by_hash: dict[str, int] = {}
        duplicate_count_by_hash: dict[str, int] = {}
        processed: list[ProcessedComment] = []
        all_comments: list[ProcessedComment] = []
        borderline_comment_ids: list[str] = []

        filter_enabled = self.settings.preprocessing_filter_enabled

        for comment in comments:
            raw_text = comment.text_raw.strip()
            if not raw_text:
                all_comments.append(self._filtered_comment(comment, video, "empty"))
                continue

            normalized = normalize_text(raw_text)

            words = _WORD_RE.findall(normalized)

            if filter_enabled:
                if len(words) < self.settings.comment_min_words:
                    all_comments.append(
                        self._filtered_comment(comment, video, "too_short", normalized)
                    )
                    continue
                if looks_like_noise(normalized) or _URL_ONLY_RE.match(raw_text.strip()):
                    all_comments.append(self._filtered_comment(comment, video, "noise", normalized))
                    continue
                if self._is_low_signal_comment(normalized, words):
                    all_comments.append(
                        self._filtered_comment(comment, video, "low_signal", normalized)
                    )
                    continue

            moderation_action = "keep"
            moderation_reason: str | None = None
            moderation_source = "rule"
            moderation_score: float | None = None
            borderline_review = False

            if filter_enabled:
                moderation = self._rule_based_moderation(
                    raw_text=raw_text, normalized=normalized, words=words, video=video
                )
                moderation_action = moderation["action"]
                moderation_reason = moderation["reason"]
                moderation_source = moderation["source"]
                moderation_score = moderation["score"]
                borderline_review = moderation["borderline_review"]
                if moderation_action == "drop":
                    all_comments.append(
                        self._filtered_comment(
                            comment,
                            video,
                            moderation_reason or "moderation_drop",
                            normalized,
                            moderation_source=moderation_source,
                            moderation_score=moderation_score,
                        )
                    )
                    continue

            text_hash = hash_text(normalized.lower())
            if text_hash in canonical_index_by_hash:
                duplicate_count_by_hash[text_hash] = duplicate_count_by_hash.get(text_hash, 1) + 1
                all_comments.append(self._filtered_comment(comment, video, "duplicate", normalized))
                continue
            canonical_index_by_hash[text_hash] = len(processed)
            duplicate_count_by_hash[text_hash] = 1

            language = self._detect_language(normalized)
            weight = self._calculate_weight(comment, normalized, video)
            if moderation_action == "flag":
                weight = min(
                    self.settings.comment_weight_max,
                    weight * self.settings.moderation_flagged_weight_multiplier,
                )

            processed_comment = ProcessedComment(
                youtube_comment_id=comment.youtube_comment_id,
                parent_comment_id=comment.parent_comment_id,
                author_name=comment.author_name,
                text_raw=comment.text_raw,
                text_normalized=normalized,
                text_hash=text_hash,
                language=language,
                like_count=comment.like_count,
                reply_count=comment.reply_count,
                published_at=comment.published_at,
                weight=weight,
                is_top_level=comment.is_top_level,
                is_filtered=False,
                filter_reason=None,
                moderation_action=moderation_action,
                moderation_reason=moderation_reason,
                moderation_source=moderation_source,
                moderation_score=moderation_score,
            )
            processed.append(processed_comment)
            all_comments.append(processed_comment)
            if borderline_review:
                borderline_comment_ids.append(comment.youtube_comment_id)

        for item in processed:
            duplicate_count = duplicate_count_by_hash.get(item.text_hash, 1)
            if duplicate_count <= 1:
                continue
            # Preserve repeated-comment volume as a soft signal instead of dropping it entirely.
            volume_boost = min(1.25, 0.18 * min(7, duplicate_count - 1))
            item.weight = min(self.settings.comment_weight_max, item.weight + volume_boost)

        dropped_by_reason = self._reason_counter(
            [comment for comment in all_comments if comment.is_filtered],
            attr_name="filter_reason",
        )
        flagged_by_reason = self._reason_counter(
            [comment for comment in processed if comment.moderation_action == "flag"],
            attr_name="moderation_reason",
        )
        dropped_count = sum(dropped_by_reason.values())
        flagged_count = sum(flagged_by_reason.values())
        kept_count = max(0, len(processed) - flagged_count)

        return PreprocessResult(
            processed=processed,
            all_comments=all_comments,
            filtered_count=dropped_count,
            total_count=len(comments),
            dropped_count=dropped_count,
            flagged_count=flagged_count,
            kept_count=kept_count,
            dropped_by_reason=dropped_by_reason,
            flagged_by_reason=flagged_by_reason,
            llm_moderation_stats={
                "reviewed_count": 0,
                "keep_count": 0,
                "flag_count": 0,
                "drop_count": 0,
                "fail_count": 0,
                "disabled": False,
                "disabled_reason": "",
            },
            borderline_comment_ids=borderline_comment_ids,
        )

    def _detect_language(self, text: str) -> str | None:
        try:
            return detect(text)
        except LangDetectException:
            return None

    def _calculate_weight(self, comment: RawComment, text: str, video: VideoMeta) -> float:
        like_component = min(comment.like_count, 100) * 0.03
        reply_component = min(comment.reply_count, 30) * 0.08
        len_component = min(len(_WORD_RE.findall(text)) / 45.0, 1.0)

        reference = video.published_at
        age_days = max(0.0, (datetime.now(UTC) - comment.published_at).total_seconds() / 86400.0)
        if comment.published_at < reference:
            age_days += 1.0
        recency_component = max(0.0, 1.0 - math.log1p(age_days) / 4.0)

        weight = 1.0 + like_component + reply_component + len_component + recency_component
        return min(weight, self.settings.comment_weight_max)

    def _is_low_signal_comment(self, text: str, words: list[str]) -> bool:
        if not self.settings.preprocessing_low_signal_filter_enabled:
            return False
        normalized = " ".join(text.split())
        if not normalized:
            return True
        if _REPEATED_CHAR_RE.search(normalized.lower()):
            return True

        alpha_chars = sum(1 for char in normalized if char.isalpha())
        if alpha_chars < self.settings.preprocessing_low_signal_min_alpha_chars:
            return True

        no_space_chars = [char for char in normalized if not char.isspace()]
        if not no_space_chars:
            return True
        symbol_count = sum(1 for char in no_space_chars if not char.isalnum())
        symbol_ratio = symbol_count / len(no_space_chars)
        if (
            symbol_ratio > self.settings.preprocessing_low_signal_max_symbol_ratio
            and len(words) <= 10
        ):
            return True

        if len(words) >= 6:
            lowered = [word.lower() for word in words]
            diversity = len(set(lowered)) / max(1, len(lowered))
            if diversity < self.settings.preprocessing_low_signal_min_lexical_diversity:
                return True
            counts = Counter(lowered)
            if counts and (max(counts.values()) / len(lowered)) >= 0.7:
                return True
        return False

    def _filtered_comment(
        self,
        comment: RawComment,
        video: VideoMeta,
        reason: str,
        normalized_override: str | None = None,
        *,
        moderation_source: str = "rule",
        moderation_score: float | None = None,
    ) -> ProcessedComment:
        normalized = (
            normalized_override
            if normalized_override is not None
            else normalize_text(comment.text_raw)
        )
        text_hash = hash_text((normalized or comment.text_raw).lower())
        return ProcessedComment(
            youtube_comment_id=comment.youtube_comment_id,
            parent_comment_id=comment.parent_comment_id,
            author_name=comment.author_name,
            text_raw=comment.text_raw,
            text_normalized=normalized,
            text_hash=text_hash,
            language=self._detect_language(normalized) if normalized else None,
            like_count=comment.like_count,
            reply_count=comment.reply_count,
            published_at=comment.published_at,
            weight=min(0.5, self._calculate_weight(comment, normalized or comment.text_raw, video)),
            is_top_level=comment.is_top_level,
            is_filtered=True,
            filter_reason=reason,
            moderation_action="drop",
            moderation_reason=reason,
            moderation_source=moderation_source,
            moderation_score=moderation_score,
        )

    def _rule_based_moderation(
        self,
        *,
        raw_text: str,
        normalized: str,
        words: list[str],
        video: VideoMeta,
    ) -> dict[str, Any]:
        if not self.settings.moderation_enabled:
            return {
                "action": "keep",
                "reason": None,
                "source": "rule",
                "score": None,
                "borderline_review": False,
            }

        if self._is_spam_link(raw_text=raw_text, normalized=normalized):
            return {
                "action": "drop",
                "reason": "spam_link",
                "source": "rule",
                "score": 0.98,
                "borderline_review": False,
            }

        profane = self._contains_profanity(normalized)
        has_position = self._has_position_signal(text=normalized, video=video)
        borderline_score = self._estimate_borderline_score(
            text=normalized,
            words=words,
            has_position=has_position,
            profane=profane,
        )
        score_min = min(
            self.settings.moderation_borderline_min_score,
            self.settings.moderation_borderline_max_score,
        )
        score_max = max(
            self.settings.moderation_borderline_min_score,
            self.settings.moderation_borderline_max_score,
        )

        if profane:
            if self.settings.moderation_toxicity_policy == "hard_delete":
                return {
                    "action": "drop",
                    "reason": "profanity",
                    "source": "rule",
                    "score": 0.94,
                    "borderline_review": False,
                }
            if has_position and self.settings.moderation_toxicity_policy == "keep_flag":
                return {
                    "action": "flag",
                    "reason": "toxic_with_position",
                    "source": "rule",
                    "score": 0.72,
                    "borderline_review": False,
                }
            if len(words) <= self.settings.moderation_profanity_only_max_words or not has_position:
                return {
                    "action": "drop",
                    "reason": "profanity_only",
                    "source": "rule",
                    "score": 0.90,
                    "borderline_review": False,
                }
            return {
                "action": "flag",
                "reason": "toxic_with_position",
                "source": "rule",
                "score": 0.68,
                "borderline_review": False,
            }

        if (
            self.settings.moderation_enable_llm_borderline
            and self.settings.moderation_llm_scope in {"borderline", "all"}
            and 3 <= len(words) <= 6
            and score_min <= borderline_score <= score_max
        ):
            return {
                "action": "keep",
                "reason": "borderline_for_llm",
                "source": "fallback",
                "score": borderline_score,
                "borderline_review": True,
            }

        return {
            "action": "keep",
            "reason": None,
            "source": "rule",
            "score": borderline_score,
            "borderline_review": False,
        }

    def _is_spam_link(self, *, raw_text: str, normalized: str) -> bool:
        raw = raw_text.lower()
        normalized_lower = normalized.lower()
        url_hits = len(_URL_RE.findall(raw_text))
        if url_hits >= 2:
            return True
        if _SPAM_HINT_RE.search(raw) or _SPAM_HINT_RE.search(normalized_lower):
            return True
        return bool(url_hits >= 1 and ("promo" in raw or "\u043f\u0440\u043e\u043c\u043e" in raw))

    def _contains_profanity(self, text: str) -> bool:
        return bool(_PROFANITY_RE.search(text.lower()))

    def _video_topic_tokens(self, video: VideoMeta) -> set[str]:
        text = " ".join(part for part in [video.title, video.description or ""] if part).lower()
        tokens = {token for token in _WORD_RE.findall(text) if len(token) >= 4}
        return {token for token in tokens if token not in _TOPIC_STOPWORDS}

    def _has_position_signal(self, *, text: str, video: VideoMeta) -> bool:
        text_lower = text.lower()
        if _CLAIM_SIGNAL_RE.search(text_lower):
            return True
        comment_tokens = {token for token in _WORD_RE.findall(text_lower) if len(token) >= 4}
        comment_tokens = {token for token in comment_tokens if token not in _TOPIC_STOPWORDS}
        if not comment_tokens:
            return False
        overlap = len(comment_tokens.intersection(self._video_topic_tokens(video)))
        return overlap >= max(0, int(self.settings.moderation_topic_token_min_overlap))

    def _estimate_borderline_score(
        self, *, text: str, words: list[str], has_position: bool, profane: bool
    ) -> float:
        score = 0.0
        normalized = " ".join(text.split()).lower()
        if 3 <= len(words) <= 6:
            score += 0.35
        elif len(words) <= 2:
            score += 0.12
        else:
            score += 0.20
        if has_position:
            score -= 0.22
        if profane:
            score += 0.18
        if "?" in normalized:
            score += 0.07
        if _REPEATED_CHAR_RE.search(normalized):
            score += 0.10
        unique_ratio = len({word.lower() for word in words}) / max(1, len(words))
        score += max(0.0, 0.3 - unique_ratio) * 0.5
        return max(0.0, min(1.0, score))

    def _reason_counter(
        self, comments: list[ProcessedComment], *, attr_name: str
    ) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for comment in comments:
            reason = str(getattr(comment, attr_name, "") or "").strip()
            if not reason:
                reason = "unspecified"
            counter[reason] += 1
        return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
