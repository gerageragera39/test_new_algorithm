"""Daily briefing generation from clustered YouTube comment topics.

Assembles executive summaries, action items, misunderstandings, risk signals,
and trend comparisons into a structured DailyBriefing report for the content author.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np

from app.core.config import Settings
from app.schemas.domain import (
    ActionItem,
    DailyBriefing,
    TopicSummary,
    TopicTrendPoint,
    TopicTrendSeries,
    VideoMeta,
)

_NON_PRACTICAL_ACTION_RE = re.compile(
    r"\b(правительств|министер|парламент|генштаб|армия|фронт|военн|реформ|"
    r"ввести механизм|создать механизм|общественные слушания)\b",
    re.IGNORECASE,
)
_QUESTION_SIGNAL_RE = re.compile(
    r"^(почему|зачем|как|когда|где|кто|что|сколько|разве|неужели|ли)\b",
    re.IGNORECASE,
)
_TOPIC_TOKEN_RE = re.compile(r"[a-zа-яё0-9]{3,}", re.IGNORECASE)
_TOXIC_SIGNAL_RE = re.compile(
    r"\b(ложь|ненавиж|пропаганд|предател|обман|манипул|туп|идиот|мраз|бред)\b",
    re.IGNORECASE,
)
_FACTUAL_SIGNAL_RE = re.compile(
    r"\b(факт|источник|доказатель|неверн|ошибк|искажен|неправд|неточн)\b",
    re.IGNORECASE,
)
_CONFUSION_SIGNAL_RE = re.compile(
    r"\b(не понял|непонятн|как так|почему|объясни|противореч|неясн)\b",
    re.IGNORECASE,
)
_TOPIC_STOPWORDS = {
    "это",
    "этот",
    "эта",
    "эти",
    "для",
    "или",
    "как",
    "что",
    "где",
    "когда",
    "если",
    "потому",
    "тема",
    "темы",
    "автор",
    "видео",
    "выпуск",
    "комментарий",
    "комментарии",
    "topic",
    "video",
}
_GENERIC_ACTION_RE = re.compile(
    r"\b(faq|вопрос аудитории|отдельный блок faq|дай faq|ответь на вопрос|"
    r"просто ответь|разбери тему|сделай блок по теме)\b",
    re.IGNORECASE,
)


def _cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    a_vec = np.array(a, dtype=np.float32)
    b_vec = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom == 0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


class BriefingService:
    """Builds a daily briefing report from analyzed topic summaries."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)

    def build(
        self,
        *,
        video: VideoMeta,
        mode: str,
        topics: list[TopicSummary],
        previous_topics: list[TopicSummary] | None = None,
        disagreement_comments: list[str] | None = None,
    ) -> DailyBriefing:
        top_topics = sorted(topics, key=lambda t: t.weighted_share, reverse=True)
        executive = self._build_executive_summary(top_topics)
        disagreements = self._collect_disagreements(disagreement_comments or [])
        action_items = self._build_action_items(top_topics)
        actions = [item.action for item in action_items]
        misunderstandings = self._collect_misunderstandings(top_topics)
        risks = self._collect_risks(top_topics)
        trends = self._build_trend(top_topics, previous_topics or [])
        return DailyBriefing(
            video_id=video.youtube_video_id,
            video_title=video.title,
            published_at=video.published_at,
            mode=mode,
            executive_summary=executive,
            top_topics=top_topics,
            actions_for_tomorrow=actions,
            action_items=action_items,
            misunderstandings_and_controversies=misunderstandings,
            audience_requests_and_questions=[],
            risks_and_toxicity=risks,
            representative_quotes=[],
            author_disagreement_comments=disagreements,
            trend_vs_previous=trends,
            metadata={
                "topic_count": len(top_topics),
                "top_topic_labels": [topic.label for topic in top_topics[:3]],
                "author_disagreement_count": len(disagreements),
                "actions_count": len(actions),
                "misunderstandings_count": len(misunderstandings),
                "requests_count": 0,
                "risks_count": len(risks),
                "trend_matching_method": "centroid+lexical",
            },
        )

    def build_topic_trend_series(
        self,
        current_briefing: DailyBriefing,
        previous_briefings: Sequence[DailyBriefing],
    ) -> list[TopicTrendSeries]:
        history = sorted(previous_briefings, key=lambda item: item.published_at)
        series_list: list[TopicTrendSeries] = []
        recent_previous_topics = history[-1].top_topics if history else []

        for topic in current_briefing.top_topics[: min(6, len(current_briefing.top_topics))]:
            points: list[TopicTrendPoint] = []
            for briefing in history:
                score, matched = self._match_topic(topic, briefing.top_topics)
                if matched is None:
                    points.append(
                        TopicTrendPoint(
                            video_id=briefing.video_id,
                            video_title=briefing.video_title,
                            published_at=briefing.published_at,
                            share_pct=0.0,
                            weighted_share=0.0,
                            matched_topic_label="",
                            similarity=0.0,
                            is_current=False,
                        )
                    )
                    continue
                points.append(
                    TopicTrendPoint(
                        video_id=briefing.video_id,
                        video_title=briefing.video_title,
                        published_at=briefing.published_at,
                        share_pct=matched.share_pct,
                        weighted_share=matched.weighted_share,
                        matched_topic_label=matched.label,
                        similarity=round(score, 3),
                        is_current=False,
                    )
                )

            points.append(
                TopicTrendPoint(
                    video_id=current_briefing.video_id,
                    video_title=current_briefing.video_title,
                    published_at=current_briefing.published_at,
                    share_pct=topic.share_pct,
                    weighted_share=topic.weighted_share,
                    matched_topic_label=topic.label,
                    similarity=1.0,
                    is_current=True,
                )
            )
            summary = self._build_single_trend_summary(topic, recent_previous_topics)
            series_list.append(
                TopicTrendSeries(
                    cluster_key=topic.cluster_key,
                    topic_label=topic.label,
                    summary=summary,
                    points=points,
                )
            )
        return series_list

    def _build_action_items(self, topics: Iterable[TopicSummary]) -> list[ActionItem]:
        action_items: list[ActionItem] = []

        for topic in topics:
            label = self._clean_label(topic.label)
            complaints = int(topic.intent_distribution.get("complaint", 0) or 0)
            questions = int(topic.intent_distribution.get("question", 0) or 0)
            priority = self._calculate_action_priority(topic, complaints, questions)
            question = self._pick_key_question(topic)
            action_text = self._pick_topic_action(topic)

            if action_text and _NON_PRACTICAL_ACTION_RE.search(action_text.lower()):
                continue
            action_items.append(
                ActionItem(
                    topic_cluster_key=topic.cluster_key,
                    topic_label=label,
                    share_pct=topic.share_pct,
                    priority=priority,
                    action=action_text,
                    key_criticism="",
                    key_question=question,
                )
            )

        action_items.sort(key=lambda item: item.priority, reverse=True)
        if action_items:
            return action_items

        return [
            ActionItem(
                topic_cluster_key="overview",
                topic_label="Общий обзор",
                priority=1,
                action="",
            )
        ]

    def _clean_label(self, label: str | None) -> str:
        return " ".join((label or "").split()).strip() or "текущая тема"

    def _calculate_action_priority(
        self, topic: TopicSummary, complaints: int, questions: int
    ) -> int:
        priority = int(topic.weighted_share * 100)
        if topic.sentiment == "negative":
            priority += 50
        elif topic.sentiment == "mixed":
            priority += 25

        toxic_count = sum(
            1
            for quote in topic.representative_quotes
            if _TOXIC_SIGNAL_RE.search(quote.lower())
        )
        priority += toxic_count * 15

        factual_count = sum(
            1
            for quote in topic.representative_quotes
            if _FACTUAL_SIGNAL_RE.search(quote.lower())
        )
        priority += factual_count * 20

        if complaints >= max(2, questions):
            priority += 30

        return priority

    def _pick_topic_action(self, topic: TopicSummary) -> str:
        for raw in topic.author_actions:
            prepared = self._prepare_full_snippet(raw, max_chars=260)
            if not prepared:
                continue
            if _GENERIC_ACTION_RE.search(prepared):
                continue
            if _NON_PRACTICAL_ACTION_RE.search(prepared):
                continue
            return prepared
        return ""

    def _pick_key_question(self, topic: TopicSummary) -> str:
        scored: list[tuple[int, str]] = []
        for raw in list(topic.question_comments) + list(topic.representative_quotes):
            prepared = self._prepare_full_snippet(raw)
            if not prepared:
                continue
            score = self._score_question_candidate(prepared)
            if score <= 0:
                continue
            scored.append((score, prepared))
        if not scored:
            return ""
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _score_question_candidate(self, text: str) -> int:
        score = 0
        lowered = text.lower()
        if "?" in text:
            score += 5
        if _QUESTION_SIGNAL_RE.match(lowered) or _QUESTION_SIGNAL_RE.search(lowered):
            score += 4
        length = len(text)
        if 35 <= length <= 220:
            score += 3
        elif 20 <= length <= 320:
            score += 1
        if any(token in lowered for token in ("почему", "как", "что будет", "что делать", "зачем")):
            score += 2
        return score

    def _prepare_full_snippet(self, text: str | None, *, max_chars: int = 800) -> str:
        if not text:
            return ""
        prepared = " ".join(str(text).split()).strip().strip("«»\"'")
        if not prepared:
            return ""
        if len(prepared) > max_chars:
            prepared = prepared[: max_chars - 1].rstrip(" ,.;:") + "…"
        return prepared

    def _build_executive_summary(self, topics: list[TopicSummary]) -> str:
        if not topics:
            return "Существенных тематических кластеров не найдено. Аудитория распределена по разрозненным реакциям."
        dominant = topics[0]
        negative_share = sum(topic.share_pct for topic in topics if topic.sentiment == "negative")
        mood = "напряженный" if negative_share >= 35 else "умеренный"
        return (
            f"Главный фокус аудитории: «{dominant.label}» ({dominant.share_pct:.1f}% комментариев, "
            f"{dominant.weighted_share:.1f}% с учетом веса). "
            f"Общий тон дискуссии: {mood}. "
            "Перед следующим выпуском важно закрыть претензии, которые чаще всего повторяются в топ-темах."
        )

    def _collect_misunderstandings(self, topics: list[TopicSummary]) -> list[str]:
        items_with_confidence: list[tuple[float, str]] = []

        for topic in topics:
            total_quotes = len(topic.representative_quotes)
            if total_quotes < 2:
                continue

            factual_hits = [
                self._prepare_full_snippet(quote)
                for quote in topic.representative_quotes
                if _FACTUAL_SIGNAL_RE.search(quote.lower())
            ]
            confusion_hits = [
                self._prepare_full_snippet(quote)
                for quote in topic.representative_quotes
                if _CONFUSION_SIGNAL_RE.search(quote.lower())
            ]
            factual_ratio = len(factual_hits) / max(1, total_quotes)
            confusion_ratio = len(confusion_hits) / max(1, total_quotes)
            confidence = factual_ratio * 0.6 + confusion_ratio * 0.4

            if confidence < 0.2 or max(len(factual_hits), len(confusion_hits)) < 2:
                continue

            example = factual_hits[0] if factual_hits else confusion_hits[0]
            if factual_ratio >= confusion_ratio:
                message = (
                    f"Тема «{topic.label}»: аудитория спорит с фактурой "
                    f"({len(factual_hits)}/{total_quotes} сильных сигналов). Пример: «{example}»."
                )
            else:
                message = (
                    f"Тема «{topic.label}»: аудитория не понимает ход аргументации "
                    f"({len(confusion_hits)}/{total_quotes} сильных сигналов). Пример: «{example}»."
                )
            items_with_confidence.append((confidence, message))

        items_with_confidence.sort(key=lambda item: item[0], reverse=True)
        return [message for _, message in items_with_confidence[:4]]

    def _collect_risks(self, topics: list[TopicSummary]) -> list[str]:
        risks_with_severity: list[tuple[float, str]] = []

        for topic in topics:
            total_quotes = len(topic.representative_quotes)
            if total_quotes == 0:
                continue

            toxic_hits = [
                self._prepare_full_snippet(quote)
                for quote in topic.representative_quotes
                if _TOXIC_SIGNAL_RE.search(quote.lower())
            ]
            toxic_ratio = len(toxic_hits) / max(1, total_quotes)
            severity = toxic_ratio * 0.75
            if topic.sentiment == "negative":
                severity += 0.25

            if severity < 0.28 or len(toxic_hits) < 2:
                continue

            example = toxic_hits[0]
            if severity >= 0.55:
                message = (
                    f"Тема «{topic.label}»: высокий риск агрессивной эскалации "
                    f"({len(toxic_hits)}/{total_quotes} репрезентативных цитат). Пример: «{example}»."
                )
            else:
                message = (
                    f"Тема «{topic.label}»: заметен конфликтный тон "
                    f"({len(toxic_hits)}/{total_quotes} цитат с агрессивной формулировкой). Пример: «{example}»."
                )
            risks_with_severity.append((severity, message))

        risks_with_severity.sort(key=lambda item: item[0], reverse=True)
        return [message for _, message in risks_with_severity[:4]]

    def _collect_disagreements(self, comments: list[str]) -> list[str]:
        if not comments:
            return []
        items: list[str] = []
        seen: set[str] = set()
        for comment in comments:
            cleaned = " ".join(comment.split()).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(cleaned)
        return items

    def _build_trend(self, current: list[TopicSummary], previous: list[TopicSummary]) -> list[str]:
        if not previous:
            return []
        return [self._build_single_trend_summary(topic, previous) for topic in current[:6]]

    def _build_single_trend_summary(
        self, topic: TopicSummary, previous_topics: Sequence[TopicSummary]
    ) -> str:
        score, matched = self._match_topic(topic, previous_topics)
        if matched is None:
            return (
                f"Тема «{topic.label}»: в предыдущем отчете не найдено достаточно похожей темы, "
                "поэтому динамику лучше считать новой линией наблюдения."
            )

        delta = topic.share_pct - matched.share_pct
        if abs(delta) < 1.0:
            change = (
                f"держится почти на том же уровне: {topic.share_pct:.1f}% сейчас против "
                f"{matched.share_pct:.1f}% раньше"
            )
        elif delta > 0:
            change = (
                f"выросла на {abs(delta):.1f} процентного пункта: {matched.share_pct:.1f}% → "
                f"{topic.share_pct:.1f}%"
            )
        else:
            change = (
                f"снизилась на {abs(delta):.1f} процентного пункта: {matched.share_pct:.1f}% → "
                f"{topic.share_pct:.1f}%"
            )
        return (
            f"Тема «{topic.label}» ближе всего к прошлой теме «{matched.label}» "
            f"(сходство {score:.2f}) и {change}."
        )

    def _match_topic(
        self, current_topic: TopicSummary, candidates: Sequence[TopicSummary]
    ) -> tuple[float, TopicSummary | None]:
        best_score = 0.0
        best_topic: TopicSummary | None = None
        for candidate in candidates:
            score = self._topic_match_score(current_topic, candidate)
            if score > best_score:
                best_score = score
                best_topic = candidate
        threshold = max(0.55, float(self.settings.trend_similarity_threshold) - 0.12)
        if best_topic is None or best_score < threshold:
            return best_score, None
        return best_score, best_topic

    def _topic_match_score(self, left: TopicSummary, right: TopicSummary) -> float:
        left_label = self._normalize_text(left.label)
        right_label = self._normalize_text(right.label)
        if left_label and left_label == right_label:
            return 1.0

        centroid_score = _cosine_similarity(left.centroid, right.centroid)
        signature_score = self._keyword_jaccard(
            self._topic_signature_tokens(left),
            self._topic_signature_tokens(right),
        )
        label_score = self._keyword_jaccard(
            self._topic_signature_tokens(left, include_quotes=False),
            self._topic_signature_tokens(right, include_quotes=False),
        )
        return max(
            centroid_score * 0.75 + signature_score * 0.25,
            centroid_score * 0.6 + label_score * 0.4,
            signature_score * 0.7 + label_score * 0.3,
        )

    def _topic_signature_tokens(
        self,
        topic: TopicSummary,
        *,
        include_quotes: bool = True,
    ) -> set[str]:
        parts = [topic.label, topic.description]
        if include_quotes:
            parts.extend(topic.representative_quotes[:2])
            parts.extend(topic.question_comments[:1])
        tokens = {
            token.lower()
            for part in parts
            for token in _TOPIC_TOKEN_RE.findall(part or "")
            if token.lower() not in _TOPIC_STOPWORDS
        }
        return tokens

    def _keyword_jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left.union(right)
        if not union:
            return 0.0
        return len(left.intersection(right)) / len(union)

    def _normalize_text(self, value: str | None) -> str:
        return " ".join((value or "").lower().split()).strip()

    def summarize_emotions(self, topics: list[TopicSummary]) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for topic in topics:
            counter.update(topic.emotion_tags)
        return dict(counter)
