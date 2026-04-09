"""LLM-powered cluster labeling and editorial analysis for comment topics.

Provides multiple LLM provider implementations (OpenAI, Ollama, LM Studio,
OpenRouter) and a rule-based fallback for analyzing comment clusters,
generating topic labels, sentiment analysis, and actionable editorial guidance.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from app.core.config import Settings
from app.core.exceptions import BudgetExceededError
from app.schemas.domain import ClusterDraft, ClusterLabelResult, ProcessedComment
from app.services.budget import BudgetGovernor
from app.services.openai_compat import (
    build_completion_token_kwargs,
    build_temperature_kwargs,
    extract_cached_input_tokens,
)
from app.services.openai_endpoint import ensure_openai_endpoint_allowed

_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)
_GENERIC_LABEL_RE = re.compile(
    r"\b(comment|comments|video|news|topic|analysis|reaction|комментар|видео|новост|реакц|мнение)\b",
    re.IGNORECASE,
)
_WEAK_ACTION_RE = re.compile(
    r"^(like|dislike|comment|share|subscribe|liking|sharing|commenting|reaction|engagement|"
    r"лайк|дизлайк|коммент|комментарий|комментарии|подпишись|подписка|репост|шэр)$",
    re.IGNORECASE,
)
_NON_REALISTIC_ACTION_RE = re.compile(
    r"\b("
    r"правительств|президент|парламент|министер|департамент|закон|реформ|госпрограмм|"
    r"генштаб|армия|армии|армией|фронт|батальон|бригада|военн|командир|командован|"
    r"ввести механизм|создать механизм|система мониторинга|общественные слушания|"
    r"audit of military strategy|military strategy|commander|government|ministry|policy reform"
    r")\b",
    re.IGNORECASE,
)
_PRACTICAL_ACTION_HINT_RE = re.compile(
    r"\b("
    r"выпуск|ролик|видео|интро|подводк|описани|таймкод|глав|цитат|источник|ссылк|"
    r"уточн|объясн|ответ|вопрос|faq|закреп|комментар|опрос|тезис|формулировк|"
    r"превью|заголов|структур|блок|раздел|сегмент|плашк|корректировк"
    r")\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "это",
    "как",
    "что",
    "для",
    "или",
    "если",
    "про",
    "все",
    "сегодня",
    "завтра",
    "когда",
    "очень",
    "тоже",
    "почему",
    "просто",
    "чтобы",
    "который",
    "которые",
    "потому",
    "будет",
    "нужно",
    "можно",
    "канал",
    "видео",
    "news",
    "video",
    "comments",
}
_POSITIVE = {"спасибо", "класс", "отлично", "поддерживаю", "супер", "хорошо"}
_NEGATIVE = {"ложь", "проблема", "плохо", "ужас", "бред", "ошибка", "скандал", "негатив"}
_INTENT_KEYS = ("question", "request", "complaint", "praise", "suggestion", "joke", "other")
_DEFAULT_AUTHOR_NAME = "автор канала"
_UNSAFE_JSON_TEXT_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF]")


@dataclass
class ClusterContext:
    """Contextual data for a single comment cluster awaiting analysis.

    Bundles the cluster draft, representative comments, episode topics,
    and previous labels needed by LLM providers to generate analysis.
    """

    cluster: ClusterDraft
    representative_comments: list[ProcessedComment]
    all_comments: list[ProcessedComment]
    episode_topics: list[str] = field(default_factory=list)
    matched_episode_topic: str | None = None
    previous_topic_labels: list[str] = field(default_factory=list)


class LLMProvider(ABC):
    """Abstract base class for LLM providers that analyze comment clusters.

    Subclasses must implement availability checking and cluster analysis
    to produce structured ClusterLabelResult objects.
    """

    provider_name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Provider availability probe."""

    @abstractmethod
    def analyze_cluster(self, ctx: ClusterContext) -> ClusterLabelResult:
        """Analyze one topic cluster."""


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _enforce_label_limit(label: str) -> str:
    return _compact(label)


def _extract_json(text: str) -> dict[str, Any] | None:
    raw_text = (text or "").strip()
    if not raw_text:
        return None

    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    if raw_text.startswith("```"):
        fenced = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE | re.DOTALL
        ).strip()
        if fenced:
            try:
                payload = json.loads(fenced)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                raw_text = fenced

    start = -1
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(raw_text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = raw_text[start : idx + 1]
                try:
                    payload = json.loads(candidate)
                    if isinstance(payload, dict):
                        return payload
                except json.JSONDecodeError:
                    pass
    return None


def _sanitize_openai_text(text: str) -> str:
    """Remove code points that can make the JSON request body invalid.

    Some raw YouTube comments occasionally contain lone surrogates or control
    characters. Those characters can break JSON serialization at the HTTP layer
    and trigger OpenAI's "could not parse the JSON body" 400 response.
    """
    if not text:
        return ""
    normalized = str(text).encode("utf-8", errors="replace").decode("utf-8")
    return _UNSAFE_JSON_TEXT_RE.sub(" ", normalized)


def _normalize_description(raw_description: Any) -> str:
    text = _compact(str(raw_description or ""))
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        parsed: Any = None
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
        if isinstance(parsed, list):
            items = [_compact(str(item)) for item in parsed if _compact(str(item))]
            if items:
                return ". ".join(items[:3])
    return text


def _topic_supported_by_comments(topic: str | None, comments: list[ProcessedComment]) -> bool:
    if not topic:
        return False
    topic_tokens = [token.lower() for token in _WORD_RE.findall(topic) if len(token) >= 4]
    topic_tokens = [token for token in topic_tokens if token not in _STOPWORDS]
    if not topic_tokens:
        return False
    corpus = " ".join(comment.text_normalized.lower() for comment in comments)
    hits = sum(1 for token in topic_tokens if token in corpus)
    return hits >= max(1, min(2, len(topic_tokens) // 2))


def _comment_token_set(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(text.lower())
        if len(token) >= 4 and token.lower() not in _STOPWORDS
    }


def _build_prompt_comment_lines(ctx: ClusterContext, max_items: int | None = None) -> list[str]:
    seen: set[str] = set()
    representative_lines: list[str] = []
    for comment in ctx.representative_comments:
        text = _compact(comment.text_raw or comment.text_normalized)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        representative_lines.append(text)
        if max_items is not None and len(representative_lines) >= max_items:
            return representative_lines

    anchor_tokens: set[str] = set()
    for line in representative_lines:
        anchor_tokens.update(_comment_token_set(line))

    prioritized: list[tuple[tuple[float, float, float], str]] = []
    fallback: list[tuple[tuple[float, float, float], str]] = []
    for comment in sorted(ctx.all_comments, key=lambda item: item.weight, reverse=True):
        text = _compact(comment.text_raw or comment.text_normalized)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens = _comment_token_set(text)
        overlap = float(len(tokens.intersection(anchor_tokens))) if anchor_tokens else 0.0
        coverage = overlap / max(1.0, float(len(tokens))) if tokens else 0.0
        rank = (overlap, coverage, float(comment.weight))

        if not anchor_tokens:
            prioritized.append((rank, text))
            continue
        if overlap >= 2.0 or (overlap >= 1.0 and coverage >= 0.14):
            prioritized.append((rank, text))
        else:
            fallback.append((rank, text))

    prioritized.sort(key=lambda item: item[0], reverse=True)
    fallback.sort(key=lambda item: item[0], reverse=True)

    lines: list[str] = list(representative_lines)
    for _, text in prioritized:
        lines.append(text)
        if max_items is not None and len(lines) >= max_items:
            return lines
    for _, text in fallback:
        lines.append(text)
        if max_items is not None and len(lines) >= max_items:
            break
    return lines


def _quote_limit_for_context(ctx: ClusterContext) -> int:
    return max(8, min(40, len(ctx.all_comments)))


def _build_default_intents(comments: list[ProcessedComment]) -> dict[str, int]:
    intents = dict.fromkeys(_INTENT_KEYS, 0)
    for comment in comments:
        text = comment.text_normalized.lower()
        if "?" in text:
            intents["question"] += 1
        elif any(word in text for word in ("прошу", "пожалуйста", "сделайте", "расскажите")):
            intents["request"] += 1
        elif any(word in text for word in ("плохо", "ошибка", "неправда", "бред", "ложь")):
            intents["complaint"] += 1
        elif any(word in text for word in ("спасибо", "класс", "отлично", "лучший")):
            intents["praise"] += 1
        elif any(word in text for word in ("можно", "стоит", "предлагаю", "лучше")):
            intents["suggestion"] += 1
        elif any(word in text for word in ("ха", "лол", "шутка", "мем")):
            intents["joke"] += 1
        else:
            intents["other"] += 1
    return intents


_LOW_VALUE_KEYWORD_TOKENS = {
    "more",
    "most",
    "very",
    "just",
    "only",
    "than",
    "then",
    "this",
    "that",
    "they",
    "them",
    "their",
    "about",
    "\u0431\u043e\u043b\u0435\u0435",
    "\u043c\u0435\u043d\u0435\u0435",
    "\u0442\u0435\u043f\u0435\u0440\u044c",
    "\u043f\u0440\u043e\u0441\u0442\u043e",
    "\u0442\u043e\u043b\u044c\u043a\u043e",
    "\u0435\u0433\u043e",
    "\u0435\u0435",
    "\u0438\u0445",
    "\u043e\u043d\u0438",
    "\u043e\u043d\u0430",
    "\u043e\u043d\u043e",
    "\u0432\u0430\u0448",
    "\u0432\u0430\u0448\u0438",
    "\u043d\u0430\u0448",
    "\u043d\u0430\u0448\u0438",
}


def _extract_salient_keywords(
    comments: list[ProcessedComment], *, max_keywords: int = 5
) -> list[str]:
    token_tf: Counter[str] = Counter()
    token_df: Counter[str] = Counter()
    token_weight: dict[str, float] = {}
    docs = 0

    for comment in comments:
        text = _compact(comment.text_raw or comment.text_normalized)
        if not text:
            continue
        docs += 1
        seen: set[str] = set()
        for token in _WORD_RE.findall(text.lower()):
            if len(token) < 4:
                continue
            if token in _STOPWORDS or token in _LOW_VALUE_KEYWORD_TOKENS:
                continue
            if token.isdigit():
                continue
            token_tf[token] += 1
            token_weight[token] = token_weight.get(token, 0.0) + max(0.1, float(comment.weight))
            seen.add(token)
        for token in seen:
            token_df[token] += 1

    if docs <= 0:
        return []

    min_tf = 2 if docs >= 6 else 1
    scored: list[tuple[float, str]] = []
    for token, tf in token_tf.items():
        if tf < min_tf:
            continue
        df_ratio = token_df.get(token, 1) / float(max(1, docs))
        if df_ratio > 0.72:
            continue
        weight_bonus = token_weight.get(token, 0.0) / float(max(1, docs))
        score = (tf * (1.6 - df_ratio)) + (0.35 * weight_bonus)
        scored.append((score, token))

    scored.sort(key=lambda item: item[0], reverse=True)
    keywords: list[str] = []
    for _, token in scored:
        if token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= max(1, max_keywords):
            break
    return keywords


def _build_keyword_appendix(comments: list[ProcessedComment]) -> str:
    keywords = _extract_salient_keywords(comments, max_keywords=5)
    if not keywords:
        return (
            "\u0412 \u043a\u043e\u043c\u043c\u0435\u043d\u0442\u0430\u0440\u0438\u044f\u0445 \u043d\u0435\u0442 "
            "\u0443\u0441\u0442\u043e\u0439\u0447\u0438\u0432\u044b\u0445 \u0441\u043c\u044b\u0441\u043b\u043e\u0432\u044b\u0445 "
            "\u043c\u0430\u0440\u043a\u0435\u0440\u043e\u0432."
        )
    return (
        "\u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u043c\u0430\u0440\u043a\u0435\u0440\u044b "
        "\u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u044f: "
        f"{', '.join(keywords)}."
    )


def _sanitize_label(
    raw_label: Any,
    fallback_label: str,
    ctx: ClusterContext,
    *,
    allow_matched_topic: bool,
) -> str:
    # Handle various types: str, list of strings, dict with nested label.
    if isinstance(raw_label, str):
        label = _enforce_label_limit(raw_label)
    elif isinstance(raw_label, list) and raw_label:
        label = _enforce_label_limit(str(raw_label[0]))
    elif isinstance(raw_label, dict):
        label = _enforce_label_limit(str(raw_label.get("text", raw_label.get("title", ""))))
    else:
        label = ""
    if len(label) < 2:
        if allow_matched_topic and ctx.matched_episode_topic:
            label = _enforce_label_limit(ctx.matched_episode_topic)
        else:
            label = fallback_label
    return label


def _is_day_one_author_action(action: str) -> bool:
    normalized = _compact(action).lower()
    if not normalized:
        return False
    if len(normalized.split()) < 4:
        return False
    if _NON_REALISTIC_ACTION_RE.search(normalized):
        return False
    return bool(_PRACTICAL_ACTION_HINT_RE.search(normalized))


def _sanitize_actions(
    raw_actions: Any,
    fallback_actions: list[str],
    *,
    max_items: int = 50,
) -> list[str]:
    actions: list[str] = []
    if isinstance(raw_actions, list):
        for item in raw_actions:
            action = _compact(str(item))
            if not action:
                continue
            if _WEAK_ACTION_RE.match(action):
                continue
            if not _is_day_one_author_action(action):
                continue
            actions.append(action)
    if len(actions) < 2:
        actions = list(fallback_actions)
    seen: set[str] = set()
    deduped: list[str] = []
    for action in actions:
        key = action.lower()
        if key in seen:
            continue
        seen.add(key)
        if not _is_day_one_author_action(action):
            continue
        deduped.append(action.strip())
        if len(deduped) >= max_items:
            break
    if deduped:
        return deduped
    fallback: list[str] = []
    for action in fallback_actions:
        if _is_day_one_author_action(action):
            fallback.append(action.strip())
        if len(fallback) >= max_items:
            break
    return fallback


def _normalize_intents(raw_intents: Any, fallback_intents: dict[str, int]) -> dict[str, int]:
    if not isinstance(raw_intents, dict):
        return dict(fallback_intents)
    normalized = dict.fromkeys(_INTENT_KEYS, 0)
    for key, value in raw_intents.items():
        if key not in normalized:
            continue
        if isinstance(value, (int, float)):
            normalized[key] = max(0, int(value))
    if sum(normalized.values()) == 0:
        return dict(fallback_intents)
    return normalized


def _sanitize_quotes(
    raw_quotes: Any, fallback_quotes: list[str], *, max_items: int = 8
) -> list[str]:
    quotes: list[str] = []
    if isinstance(raw_quotes, list):
        for item in raw_quotes:
            quote = _compact(str(item))
            if quote:
                quotes.append(quote)
    if not quotes:
        quotes = list(fallback_quotes)
    deduped: list[str] = []
    seen: set[str] = set()
    for quote in quotes:
        key = quote.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(quote)
        if len(deduped) >= max_items:
            break
    return deduped


def _build_strict_cluster_prompt(
    *,
    ctx: ClusterContext,
    comment_lines: list[str],
    author_name: str,
) -> str:
    comments = "\n".join(f"- {line}" for line in comment_lines) or "- n/a"
    episode_topics = "\n".join(f"- {topic}" for topic in ctx.episode_topics[:6]) or "- n/a"
    matched = ctx.matched_episode_topic or "n/a"
    return (
        "Analyze one topic cluster from YouTube comments.\n"
        "Return strict JSON object with keys:\n"
        "label, description, author_actions, sentiment, emotion_tags, intent_distribution, representative_quotes.\n"
        "Host identity:\n"
        f"- The host is {author_name}.\n"
        "Action rules (critical):\n"
        "- author_actions: list of 3-5 concrete steps for the next episode only.\n"
        "- Each action must be realistic in one working day and executable personally by the host.\n"
        "- Allowed: clarify thesis, add sources, structure blocks, FAQ, pinned comment, timestamps.\n"
        "- Forbidden: state policy, military command changes, legal/government reforms, organization-wide systems.\n"
        "- Do NOT output platform verbs like like/comment/share/subscribe.\n"
        "Topic rules:\n"
        "- If cluster clearly maps to matched episode topic, keep it aligned.\n"
        "- If comments introduce a better/new angle than transcript topics, create a new category from comments.\n"
        "- label: specific and concise, no generic labels.\n"
        "- description: 2-4 concise sentences tied to comments.\n"
        "- sentiment must be positive|neutral|negative.\n"
        "- intent_distribution keys: question,request,complaint,praise,suggestion,joke,other.\n"
        "- representative_quotes: 6-12 short exact comments from sample.\n"
        "- Output language: Russian.\n"
        f"Matched episode topic: {matched}\n"
        f"Episode topics:\n{episode_topics}\n"
        f"Comment sample ({len(comment_lines)}):\n{comments}\n"
    )


def _merge_short_description_with_fallback(
    description: str, fallback_description: str, *, min_len: int = 80
) -> str:
    base = _compact(description)
    fallback = _compact(fallback_description)
    if len(base) >= min_len:
        return base
    if not fallback:
        return base
    if not base:
        return fallback
    if fallback.lower() in base.lower():
        return base
    if not base.endswith((".", "!", "?")):
        base = f"{base}."
    return f"{base} {fallback}"


def _unwrap_nested_response(data: dict[str, Any]) -> dict[str, Any]:
    """Unwrap GPT-5 responses that may nest the actual payload inside a wrapper key."""
    if not data:
        return data
    # If 'label' is already at top level, no unwrapping needed.
    if "label" in data:
        return data
    # GPT-5 sometimes wraps the response in a single key like "result", "response",
    # "analysis", "cluster", "topic", "data", or "output".
    for wrapper_key in ("result", "response", "analysis", "cluster", "topic", "data", "output"):
        nested = data.get(wrapper_key)
        if isinstance(nested, dict) and "label" in nested:
            return nested
    # If there's exactly one key and its value is a dict with 'label', unwrap it.
    if len(data) == 1:
        only_value = next(iter(data.values()))
        if isinstance(only_value, dict) and "label" in only_value:
            return only_value
    return data


def _normalize_llm_result(
    data: dict[str, Any],
    ctx: ClusterContext,
    *,
    author_name: str = _DEFAULT_AUTHOR_NAME,
    max_actions: int = 50,
    max_quotes: int = 8,
) -> ClusterLabelResult:
    data = _unwrap_nested_response(data)
    fallback = NoLLMFallbackProvider().analyze_cluster(ctx)
    topic_is_relevant = _topic_supported_by_comments(ctx.matched_episode_topic, ctx.all_comments)
    label = _sanitize_label(
        data.get("label"),
        fallback.label,
        ctx,
        allow_matched_topic=topic_is_relevant,
    )

    description = _normalize_description(data.get("description", ""))
    description = _merge_short_description_with_fallback(
        description, fallback.description, min_len=80
    )
    if (
        topic_is_relevant
        and ctx.matched_episode_topic
        and ctx.matched_episode_topic.lower() not in description.lower()
    ):
        description = f"{description} Связь с темой выпуска: {ctx.matched_episode_topic}."
    if len(description) > 520:
        description = description[:520].rstrip()

    actions = _sanitize_actions(
        data.get("author_actions"),
        fallback.author_actions,
        max_items=max_actions,
    )
    if not actions:
        actions = [
            f"{author_name}: начните выпуск с 30-секундного уточнения спорного тезиса и одной проверяемой ссылки.",
            f"{author_name}: добавьте в описание 2-3 источника по теме кластера и закрепите комментарий с фактами.",
        ][: max(1, max_actions)]

    sentiment = str(data.get("sentiment", fallback.sentiment)).lower()
    if sentiment not in {"positive", "neutral", "negative"}:
        sentiment = fallback.sentiment

    emotion_tags = data.get("emotion_tags", fallback.emotion_tags)
    if not isinstance(emotion_tags, list):
        emotion_tags = fallback.emotion_tags
    emotion_values = [_compact(str(item)) for item in emotion_tags if _compact(str(item))]
    if not emotion_values:
        emotion_values = fallback.emotion_tags

    intents = _normalize_intents(data.get("intent_distribution"), fallback.intent_distribution)
    quotes = _sanitize_quotes(
        data.get("representative_quotes"),
        fallback.representative_quotes,
        max_items=max_quotes,
    )
    return ClusterLabelResult(
        label=label,
        description=description,
        author_actions=actions,
        sentiment=sentiment,
        emotion_tags=emotion_values[:2],
        intent_distribution=intents,
        representative_quotes=quotes,
    )


class NoLLMFallbackProvider(LLMProvider):
    """Rule-based fallback provider that operates without any LLM backend.

    Uses keyword frequency, simple sentiment heuristics, and intent detection
    to produce cluster labels when no LLM provider is available.
    """

    provider_name = "fallback"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self.author_name = (
            settings.author_name.strip()
            if settings and settings.author_name and settings.author_name.strip()
            else _DEFAULT_AUTHOR_NAME
        )
        self.actions_limit = 50
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_available(self) -> bool:
        return True

    def analyze_cluster(self, ctx: ClusterContext) -> ClusterLabelResult:
        """Analyze a cluster using keyword heuristics without an LLM.

        Args:
            ctx: Cluster context with comments and episode topic data.

        Returns:
            A ClusterLabelResult with heuristically derived labels and actions.
        """
        texts = [comment.text_normalized for comment in ctx.all_comments]
        lowered = " ".join(texts).lower()
        tokens = [token for token in _WORD_RE.findall(lowered) if token not in _STOPWORDS]
        keywords = _extract_salient_keywords(ctx.all_comments, max_keywords=5)
        topic_is_relevant = _topic_supported_by_comments(
            ctx.matched_episode_topic, ctx.all_comments
        )
        if topic_is_relevant and ctx.matched_episode_topic:
            label = _enforce_label_limit(ctx.matched_episode_topic)
        else:
            if keywords:
                # Use " и " instead of " / " to avoid sanitizer rejection
                label = _enforce_label_limit(" и ".join(keywords[:2]))
            else:
                label = _enforce_label_limit("Новая тема комментариев")

        pos_hits = sum(1 for token in tokens if token in _POSITIVE)
        neg_hits = sum(1 for token in tokens if token in _NEGATIVE)
        if neg_hits > pos_hits + 1:
            sentiment = "negative"
            emotions = ["тревога", "раздражение"]
        elif pos_hits > neg_hits + 1:
            sentiment = "positive"
            emotions = ["поддержка"]
        else:
            sentiment = "neutral"
            emotions = []

        intents = _build_default_intents(ctx.all_comments)
        request_total = intents.get("request", 0) + intents.get("question", 0)
        complaint_total = intents.get("complaint", 0)

        description_parts: list[str] = [_build_keyword_appendix(ctx.all_comments)]
        if topic_is_relevant and ctx.matched_episode_topic:
            description_parts.append(
                f"Комментарии связаны с сегментом выпуска: {ctx.matched_episode_topic}."
            )
        else:
            description_parts.append(
                "Тема сформировалась из комментариев и не совпала напрямую с темами транскрипта."
            )
        description_parts.append(
            f"Перед следующим выпуском {self.author_name} важно закрыть тему коротко и с проверяемыми формулировками."
        )
        description = " ".join(description_parts)

        actions: list[str] = []
        if topic_is_relevant and ctx.matched_episode_topic:
            actions.append(
                f"{self.author_name}: дайте короткий блок по теме «{ctx.matched_episode_topic}» с четкой позицией и 1-2 источниками."
            )
        if complaint_total >= 2:
            actions.append(
                f"{self.author_name}: заранее выпишите спорный тезис и один фактчек-источник, который озвучите в начале."
            )
        if request_total >= 2:
            actions.append(
                f"{self.author_name}: добавьте мини-FAQ из 3 вопросов аудитории и коротких ответов в конце выпуска."
            )
        actions.append(
            f"{self.author_name}: в финале сформулируйте 2 конкретных вывода и один следующий шаг для аудитории."
        )

        seen: set[str] = set()
        deduped_actions: list[str] = []
        for action in actions:
            key = action.lower()
            if key in seen:
                continue
            seen.add(key)
            if _is_day_one_author_action(action):
                deduped_actions.append(action)
            if len(deduped_actions) >= self.actions_limit:
                break
        quote_limit = _quote_limit_for_context(ctx)
        quote_candidates = _build_prompt_comment_lines(
            ctx, max_items=max(quote_limit, len(ctx.representative_comments))
        )
        quotes = _sanitize_quotes(quote_candidates, quote_candidates, max_items=quote_limit)
        return ClusterLabelResult(
            label=label,
            description=description,
            author_actions=deduped_actions[: self.actions_limit],
            sentiment=sentiment,
            emotion_tags=emotions[:2],
            intent_distribution=intents,
            representative_quotes=quotes,
        )


class OpenAIChatProvider(LLMProvider):
    """LLM provider that uses the OpenAI Chat Completions API.

    Enforces per-run call limits, per-task quotas, and budget constraints
    before each API call, and records usage for cost tracking.
    """

    provider_name = "openai"

    def __init__(self, settings: Settings, budget: BudgetGovernor) -> None:
        self.settings = settings
        self.budget = budget
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url_host, self.endpoint_mode = ensure_openai_endpoint_allowed(settings)
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.calls_in_run = 0
        self.calls_by_task: Counter[str] = Counter()
        self.blocked_calls_by_reason: Counter[str] = Counter()

    def is_available(self) -> bool:
        return bool(self.settings.openai_api_key)

    def _task_limit(self, task: str) -> int | None:
        task_name = str(task or "").strip().lower()
        if task_name == "moderation_borderline":
            return max(0, int(self.settings.openai_max_moderation_calls_per_run))
        if task_name == "position_naming":
            return max(0, int(self.settings.openai_max_position_naming_calls_per_run))
        return None

    def _task_blocked_message(self, task: str) -> str:
        task_name = str(task or "").strip().lower()
        if task_name == "moderation_borderline":
            return "OpenAI moderation calls reserved for topic labeling."
        return f"OpenAI task quota reached for task={task_name or 'unknown'}."

    def _assert_call_allowed(self, task: str) -> None:
        """Enforce per-run and per-task call limits for OpenAI requests."""
        task_name = str(task or "").strip().lower()
        max_calls = max(1, int(self.settings.openai_max_calls_per_run))
        if self.calls_in_run >= max_calls:
            self.blocked_calls_by_reason["max_calls"] += 1
            raise BudgetExceededError("OpenAI per-run call limit reached.")

        if task_name == "moderation_borderline":
            reserve = max(0, int(self.settings.openai_calls_reserved_for_labeling))
            remaining_calls = max_calls - int(self.calls_in_run)
            if remaining_calls <= reserve:
                self.blocked_calls_by_reason["reserved_for_labeling"] += 1
                raise BudgetExceededError("OpenAI moderation calls reserved for topic labeling.")

        task_limit = self._task_limit(task_name)
        if task_limit is not None and int(self.calls_by_task.get(task_name, 0)) >= task_limit:
            self.blocked_calls_by_reason["task_quota"] += 1
            raise BudgetExceededError(self._task_blocked_message(task_name))

    def get_call_stats(self) -> dict[str, int]:
        blocked_task_quota = int(self.blocked_calls_by_reason.get("task_quota", 0)) + int(
            self.blocked_calls_by_reason.get("max_calls", 0)
        )
        return {
            "openai_calls_total": int(self.calls_in_run),
            "openai_calls_moderation": int(self.calls_by_task.get("moderation_borderline", 0)),
            "openai_calls_cluster_labeling": int(self.calls_by_task.get("cluster_labeling", 0)),
            "openai_calls_cluster_title": int(self.calls_by_task.get("cluster_title_naming", 0)),
            "openai_calls_position_naming": int(self.calls_by_task.get("position_naming", 0)),
            "openai_calls_blocked_reserved_for_labeling": int(
                self.blocked_calls_by_reason.get("reserved_for_labeling", 0)
            ),
            "openai_calls_blocked_task_quota": blocked_task_quota,
        }

    def request_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        task: str,
        temperature: float,
        estimated_out_tokens: int,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        self._assert_call_allowed(task)

        system_prompt = _sanitize_openai_text(system_prompt)
        prompt = _sanitize_openai_text(prompt)

        max_tokens = max(128, int(max_output_tokens or self.settings.openai_max_output_tokens))
        estimated_out = max(64, min(int(estimated_out_tokens), max_tokens))
        if self.settings.openai_hard_budget_enforced:
            # Reserve worst-case completion and a conservative prompt upper bound before making the API call.
            budgeted_out = max_tokens
            estimated_in = self.budget.estimate_tokens_upper_bound(
                [system_prompt, prompt],
                overhead_tokens=64,
            )
        else:
            budgeted_out = estimated_out
            estimated_in = self.budget.estimate_tokens([system_prompt, prompt])
        estimated_cost = self.budget.estimate_chat_cost(
            self.settings.openai_chat_model, estimated_in, budgeted_out
        )
        self.budget.assert_can_spend(
            estimated_cost=estimated_cost, estimated_tokens=estimated_in + budgeted_out
        )

        response = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            **build_temperature_kwargs(self.settings.openai_chat_model, temperature),
            **build_completion_token_kwargs(self.settings.openai_chat_model, max_tokens),
        )
        self.calls_in_run += 1
        self.calls_by_task[str(task or "").strip().lower()] += 1
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        finish_reason = choice.finish_reason

        # Diagnostic: log all message attributes for GPT-5 debugging.
        if not content or content.strip() in ("", "{}"):
            msg_attrs = {
                attr: repr(getattr(message, attr, None))[:200]
                for attr in (
                    "content",
                    "parsed",
                    "refusal",
                    "role",
                    "tool_calls",
                    "function_call",
                    "audio",
                )
                if hasattr(message, attr)
            }
            self.logger.warning(
                "LLM empty content for task=%s model=%s finish_reason=%s "
                "completion_tokens=%s message_attrs=%s",
                task,
                self.settings.openai_chat_model,
                finish_reason,
                response.usage.completion_tokens if response.usage else "?",
                msg_attrs,
            )

        # GPT-5 models may return structured output via .parsed instead of .content.
        if not content or content.strip() in ("", "{}"):
            parsed_attr = getattr(message, "parsed", None)
            if parsed_attr is not None:
                if isinstance(parsed_attr, dict):
                    content = json.dumps(parsed_attr, ensure_ascii=False)
                elif isinstance(parsed_attr, str) and parsed_attr.strip():
                    content = parsed_attr
                else:
                    content = str(parsed_attr)
                if content and content.strip() not in ("", "{}", "None"):
                    self.logger.info(
                        "LLM response for task=%s: recovered from message.parsed",
                        task,
                    )

        # Try model_dump() for SDK-parsed structured outputs.
        if not content or content.strip() in ("", "{}", "None"):
            try:
                dumped = message.model_dump(exclude_none=True)
                # Some SDKs put the response in unexpected fields.
                for key in ("content", "parsed", "text"):
                    val = dumped.get(key)
                    if isinstance(val, dict) and val:
                        content = json.dumps(val, ensure_ascii=False)
                        self.logger.info(
                            "LLM response for task=%s: recovered from model_dump[%s]",
                            task,
                            key,
                        )
                        break
                    if isinstance(val, str) and val.strip() not in ("", "{}", "None"):
                        content = val
                        break
            except Exception:
                pass

        if not content or content.strip() in ("", "None"):
            content = "{}"
        parsed = _extract_json(content) or {}
        self.logger.info(
            "LLM response for task=%s: keys=%s label=%s (raw_len=%d)",
            task,
            list(parsed.keys())[:8] if parsed else "EMPTY",
            str(parsed.get("label", ""))[:80] if parsed else "N/A",
            len(content),
        )
        prompt_tokens = response.usage.prompt_tokens if response.usage else estimated_in
        completion_tokens = response.usage.completion_tokens if response.usage else estimated_out
        cached_input_tokens = extract_cached_input_tokens(response.usage)
        final_cost = self.budget.estimate_chat_cost(
            self.settings.openai_chat_model,
            prompt_tokens,
            completion_tokens,
            cached_input_tokens=cached_input_tokens,
        )
        self.budget.record_usage(
            model=self.settings.openai_chat_model,
            provider="openai_chat",
            tokens_input=prompt_tokens,
            tokens_output=completion_tokens,
            estimated_cost_usd=final_cost,
            meta={"task": task, "cached_input_tokens": cached_input_tokens},
        )
        return parsed

    def analyze_cluster(self, ctx: ClusterContext) -> ClusterLabelResult:
        """Analyze a cluster via the OpenAI Chat Completions API.

        Args:
            ctx: Cluster context with comments and episode topic data.

        Returns:
            A ClusterLabelResult parsed and normalized from the LLM response.

        Raises:
            BudgetExceededError: If budget or call limits are exhausted.
        """
        prompt = self._build_prompt(ctx)
        parsed = self.request_json(
            prompt=prompt,
            system_prompt=(
                f"You produce practical editorial guidance for {self.settings.author_name or _DEFAULT_AUTHOR_NAME}. "
                "Use only provided context and suggest actions feasible in one day."
            ),
            task="cluster_labeling",
            temperature=0.15,
            estimated_out_tokens=800,
            max_output_tokens=min(self.settings.openai_max_output_tokens, 2048),
        )
        return self._from_llm_json(parsed, ctx)

    def _build_prompt(self, ctx: ClusterContext) -> str:
        comment_lines = _build_prompt_comment_lines(
            ctx,
            max_items=max(8, min(40, len(ctx.all_comments))),
        )
        return _build_strict_cluster_prompt(
            ctx=ctx,
            comment_lines=comment_lines,
            author_name=self.settings.author_name or _DEFAULT_AUTHOR_NAME,
        )

    def _from_llm_json(self, data: dict[str, Any], ctx: ClusterContext) -> ClusterLabelResult:
        return _normalize_llm_result(
            data,
            ctx,
            author_name=self.settings.author_name or _DEFAULT_AUTHOR_NAME,
            max_quotes=_quote_limit_for_context(ctx),
        )
