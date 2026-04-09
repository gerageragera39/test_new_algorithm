"""Enhanced toxic comment classifier with target detection and confidence scoring.

Classifies toxic comments by:
1. Target: author/guest/content/undefined/third_party
2. Confidence: 0.0-1.0 score for auto-ban vs manual review decisions

Design:
- author: Direct insults to channel author ("ты идиот", "автор дебил")
- guest: Insults to video guests from settings ("Солонин несёт бред")
- content: Insults to video/channel content ("видео говно", "канал позор")
- undefined: Insults to unspecified persons ("этот дед", "старый жид", "эти евреи")
- third_party: Insults to politicians/public figures ("Путин идиот") → NOT toxic for our purposes

Confidence levels:
- HIGH (>= 0.85): candidate for auto-ban if target is author/guest/content
- MEDIUM/LOW: retained for manual review unless the target is third_party
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.core.exceptions import BudgetExceededError, ExternalServiceError

if TYPE_CHECKING:
    from app.db.models import Comment
    from app.services.labeling import LLMProvider

logger = logging.getLogger(__name__)

_BATCH_SIZE = 20
RequestLLMJsonFn = Any


# Offensive language pre-filter (reused from toxic_detector)
_TOXIC_RE = re.compile(
    r"(?:"
    r"\bбляд\w*\b|\bбля\b|\bсука\w*\b|\bсуч\w+\b|"
    r"\bхуй\w*\b|\bхуе\w*\b|\bхуё\w*\b|\bпизд\w*\b|"
    r"\bебан\w*\b|\bебат\w*\b|\bёбан\w*\b|\bеб[аа]л\w*\b|"
    r"\bмудак\w*\b|\bмудил\w*\b|\bдолбо[её]\w*\b|"
    r"\bдебил\w*\b|\bидиот\w*\b|\bмраз\w*\b|\bтвар\w*\b|"
    r"\bублюд\w*\b|\bгандон\w*\b|\bгнид\w*\b|\bчмо\w*\b|"
    r"\bлох\w*\b|\bшлюх\w*\b|\bшалав\w*\b|\bдаун\w*\b|"
    r"\bкретин\w*\b|\bнедоум\w*\b|\bотброс\w*\b|\bподонок\w*\b|\bподонк\w*\b|"
    r"\bурод\w*\b|\bпадл\w*\b|\bмразот\w*\b|"
    r"\bжид\w*\b|\bхач\w*\b|\bчурк\w*\b|\bнигер\w*\b|\bчерножоп\w*\b|"
    r"\bfuck\w*\b|\bshit\w*\b|\bbitch\w*\b|\basshole\w*\b|"
    r"\bmoron\w*\b|\bretard\w*\b|\bcunt\w*\b|\bdumbass\w*\b"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_DEROGATORY_PHRASE_RE = re.compile(
    r"(?:"
    r"что за бред|закрой рот|заткни\w*|пошёл на\w*|пошел на\w*|"
    r"иди на\w*|катись|вали отсюда|рот закрой|"
    r"больной на голову|тупой|тупая|тупое|клоун|клоуны|позор|позорище|"
    r"несёт чушь|несет чушь|несёт бред|несет бред|городит чушь"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _has_offensive_language(text: str) -> bool:
    """Pre-filter: check if text contains any offensive language."""
    return bool(_TOXIC_RE.search(text) or _DEROGATORY_PHRASE_RE.search(text))


def _build_name_patterns(names: list[str]) -> list[re.Pattern[str]]:
    """Build regex patterns for names (author, guests)."""
    patterns: list[re.Pattern[str]] = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        parts = name.split()
        variants: list[str] = []
        for part in parts:
            if len(part) < 2:
                continue
            stem = part.rstrip("аеёийоуыьюя")
            if len(stem) < 2:
                stem = part[: max(2, len(part) - 1)]
            variants.append(re.escape(stem) + r"\w*")
        if variants:
            patterns.append(
                re.compile(r"\b(?:" + "|".join(variants) + r")\b", re.IGNORECASE | re.UNICODE)
            )
    return patterns


def _build_toxic_target_prompt(
    numbered_comments: list[tuple[int, str]],
    author_name: str,
    guest_names: list[str],
) -> str:
    """Build LLM prompt for target detection and confidence scoring."""
    lines = [f"{num}. {text[:400]}" for num, text in numbered_comments]
    comments_text = "\n".join(lines)

    guests_str = ", ".join(f"«{g}»" for g in guest_names) if guest_names else "нет"

    return (
        f"Ты модератор YouTube-канала автора «{author_name}» (политический русскоязычный канал).\n"
        f"Гости в видео: {guests_str}\n\n"
        f"Ниже — комментарии, которые предыдущий этап уже пометил как потенциально токсичные.\n"
        f"Определи для КАЖДОГО:\n"
        f"1. **target** — кому адресовано оскорбление:\n"
        f"   - 'author' — оскорбление автора канала «{author_name}» лично (по имени, «ты»/«вы», «автор», «ведущий»)\n"
        f"   - 'guest' — оскорбление гостя из списка выше (по имени или явной ссылке на гостя)\n"
        f"   - 'content' — оскорбление видео/канала/контента («видео говно», «канал позор»)\n"
        f"   - 'undefined' — оскорбление неопределённого человека или группы («этот дед», «старый жид», «эти евреи»)\n"
        f"   - 'third_party' — оскорбление известной третьей стороны (политиков, общественных деятелей, армий)\n\n"
        f"2. **confidence** — уверенность 0.0-1.0 в том, что это ДЕЙСТВИТЕЛЬНО оскорбление, за которое стоит банить.\n"
        f"   - 0.9-1.0: Явное прямое оскорбление автора/гостя/контента\n"
        f"   - 0.7-0.9: Чёткое оскорбление, но с небольшой неоднозначностью\n"
        f"   - 0.5-0.7: Грубость или оскорбление неопределённого лица\n"
        f"   - 0.3-0.5: Резкая критика без явного оскорбления\n"
        f"   - 0.0-0.3: Эмоциональная лексика, но не оскорбление\n\n"
        f"ВАЖНО:\n"
        f"- target='author'/'guest'/'content' → высокая уверенность (>= 0.85)\n"
        f"- target='undefined' → средняя уверенность (0.5-0.85)\n"
        f"- target='third_party' → низкая уверенность (< 0.5)\n\n"
        f"Комментарии:\n{comments_text}\n\n"
        f"Верни JSON:\n"
        f'{{"results": {{\n'
        f'  "1": {{"target": "author", "confidence": 0.95}},\n'
        f'  "2": {{"target": "third_party", "confidence": 0.2}},\n'
        f"  ...\n"
        f"}}}}\n"
        f"Включи ВСЕ номера из списка."
    )


def _parse_toxic_target_response(
    raw: dict | list | None,
    valid_nums: set[int],
) -> dict[int, dict[str, Any]]:
    """Parse LLM response into {comment_num: {target, confidence}}."""
    if not isinstance(raw, dict):
        return {}

    results = raw.get("results")
    if not isinstance(results, dict):
        results = raw

    parsed: dict[int, dict[str, Any]] = {}
    for key, value in results.items():
        try:
            num = int(key)
        except (ValueError, TypeError):
            continue
        if num not in valid_nums:
            continue
        if not isinstance(value, dict):
            continue

        target = value.get("target", "undefined")
        if target not in ("author", "guest", "content", "undefined", "third_party"):
            target = "undefined"

        try:
            confidence = float(value.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.0

        parsed[num] = {"target": target, "confidence": confidence}

    return parsed


def classify_toxic_with_targets(
    comments: list[Comment],
    author_name: str,
    guest_names: list[str],
    llm_provider: LLMProvider,
    request_llm_json: RequestLLMJsonFn,
) -> dict[int, dict[str, Any]]:
    """Classify toxic comments with target and confidence.

    Production policy:
    - NO hard prefilter (maximizes recall for manual review)
    - LLM classifies ALL toxic candidates from unified classifier
    - Low-confidence cases still returned (routing decision in runner)

    Returns:
        dict[comment_id, {"target": str, "confidence": float, "text": str}]
    """
    from app.services.labeling import NoLLMFallbackProvider

    if not comments:
        return {}

    # Use LLM for target detection and confidence scoring (no prefilter)
    if isinstance(llm_provider, NoLLMFallbackProvider):
        return _classify_toxic_heuristic(comments, author_name, guest_names)

    return _classify_toxic_llm(comments, author_name, guest_names, llm_provider, request_llm_json)


def _classify_toxic_llm(
    offensive_comments: list[Comment],
    author_name: str,
    guest_names: list[str],
    llm_provider: LLMProvider,
    request_llm_json: RequestLLMJsonFn,
) -> dict[int, dict[str, Any]]:
    """Use LLM to classify toxic comments with target and confidence."""
    result: dict[int, dict[str, Any]] = {}

    for batch_start in range(0, len(offensive_comments), _BATCH_SIZE):
        batch = offensive_comments[batch_start : batch_start + _BATCH_SIZE]
        numbered: list[tuple[int, str]] = []
        batch_id_map: dict[int, int] = {}

        for i, comment in enumerate(batch, start=1):
            text = (comment.text_raw or "").strip()
            numbered.append((i, text))
            batch_id_map[i] = comment.id

        prompt = _build_toxic_target_prompt(numbered, author_name, guest_names)
        valid_nums = set(batch_id_map.keys())

        try:
            data = request_llm_json(
                llm_provider,
                prompt,
                task="toxic_target_classification",
                estimated_out_tokens=max(100, len(batch) * 15),
                max_output_tokens=max(300, len(batch) * 25),
                system_prompt=(
                    f"Ты модератор канала «{author_name}». "
                    "Определи цель оскорбления и уверенность для каждого комментария. "
                    "Отвечай строго JSON."
                ),
            )
        except (BudgetExceededError, ExternalServiceError) as exc:
            logger.warning("Toxic target LLM call failed: %s — using heuristic", exc)
            heuristic_result = _classify_toxic_heuristic(batch, author_name, guest_names)
            result.update(heuristic_result)
            continue
        except Exception as exc:
            logger.warning("Toxic target unexpected error: %s — using heuristic", exc)
            heuristic_result = _classify_toxic_heuristic(batch, author_name, guest_names)
            result.update(heuristic_result)
            continue

        parsed = _parse_toxic_target_response(data, valid_nums)
        for num, classification in parsed.items():
            cid = batch_id_map.get(num)
            if cid is not None:
                comment = next((c for c in batch if c.id == cid), None)
                if comment:
                    result[cid] = {
                        "target": classification["target"],
                        "confidence": classification["confidence"],
                        "text": comment.text_raw or "",
                    }

    return result


def _classify_toxic_heuristic(
    comments: list[Comment],
    author_name: str,
    guest_names: list[str],
) -> dict[int, dict[str, Any]]:
    """Heuristic fallback for toxic classification.

    Production policy: checks for offensive language but also classifies
    borderline cases with lower confidence (maximizes recall).
    """
    author_re = _build_name_patterns([author_name]) if author_name else []
    guest_res = _build_name_patterns(guest_names)

    _AUTHOR_ADDRESS_RE = re.compile(
        r"\b(вы|вас|вам|ваш|ваша|ваше|ваши|ты|тебя|тебе|тобой|твой|твоя|автор\w*|ведущ\w*)\b",
        re.IGNORECASE,
    )
    _CONTENT_RE = re.compile(
        r"\b(видео|канал|выпуск|эфир|передач\w+|програм\w+)\b",
        re.IGNORECASE,
    )
    _UNDEFINED_RE = re.compile(
        r"\b(этот|эта|эти|те|тот|та|дед|жид\w*|еврей\w*|цыган\w*|хач\w*|чурк\w*)\b",
        re.IGNORECASE,
    )

    result: dict[int, dict[str, Any]] = {}

    for comment in comments:
        text = (comment.text_raw or "").strip()
        if not text:
            continue

        # Check if has offensive language
        has_offensive = _has_offensive_language(text)

        # Check author
        has_author_ref = bool(_AUTHOR_ADDRESS_RE.search(text))
        has_author_name = any(pattern.search(text) for pattern in author_re)

        # Check guests
        has_guest_ref = any(pattern.search(text) for pattern in guest_res)

        # Check content
        has_content_ref = bool(_CONTENT_RE.search(text))

        # Check undefined
        has_undefined_ref = bool(_UNDEFINED_RE.search(text))

        # Prioritize target detection
        if has_author_ref or has_author_name:
            target = "author"
            confidence = 0.90 if has_offensive else 0.55
        elif has_guest_ref:
            target = "guest"
            confidence = 0.90 if has_offensive else 0.55
        elif has_content_ref:
            target = "content"
            confidence = 0.85 if has_offensive else 0.50
        elif has_undefined_ref:
            target = "undefined"
            confidence = 0.60 if has_offensive else 0.40
        else:
            target = "third_party"
            confidence = 0.30

        # Only include if has some signal (offensive OR targeting)
        if has_offensive or has_author_ref or has_author_name or has_guest_ref or has_content_ref:
            result[comment.id] = {
                "target": target,
                "confidence": confidence,
                "text": text,
            }

    return result
