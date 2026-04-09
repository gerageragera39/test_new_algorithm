"""Detect comments that are toxic/offensive SPECIFICALLY toward the video author.

On political channels, commenters frequently insult third parties (politicians,
public figures).  This module must distinguish:
  - "Путин — идиот" → NOT author-directed toxic (skip)
  - "Автор — идиот" / "ты дебил" / "Солонин несёт бред" → author-directed toxic

Strategy:
  1. Regex pre-filter: keep only comments that contain ANY offensive language.
  2. LLM classification: from the pre-filtered set, ask the LLM which ones
     are directed at the video author (not at third parties).
  3. Regex fallback (free mode): check if the comment also references the
     author by name or second-person pronouns.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.services.labeling import LLMProvider, NoLLMFallbackProvider

if TYPE_CHECKING:
    from app.db.models import Comment

# Broad offensive language detector (used as pre-filter only)
_TOXIC_RE = re.compile(
    r"(?:"
    # Russian mat core stems
    r"\bбляд\w*\b|\bбля\b|\bсука\w*\b|\bсуч\w+\b|"
    r"\bхуй\w*\b|\bхуе\w*\b|\bхуё\w*\b|\bпизд\w*\b|"
    r"\bебан\w*\b|\bебат\w*\b|\bёбан\w*\b|\bеб[аа]л\w*\b|"
    r"\bмудак\w*\b|\bмудил\w*\b|\bдолбо[её]\w*\b|"
    # Insults
    r"\bдебил\w*\b|\bидиот\w*\b|\bмраз\w*\b|\bтвар\w*\b|"
    r"\bублюд\w*\b|\bгандон\w*\b|\bгнид\w*\b|\bчмо\w*\b|"
    r"\bлох\w*\b|\bшлюх\w*\b|\bшалав\w*\b|\bдаун\w*\b|"
    r"\bкретин\w*\b|\bнедоум\w*\b|\bотброс\w*\b|\bподонок\w*\b|\bподонк\w*\b|"
    r"\bурод\w*\b|\bпадл\w*\b|\bмразот\w*\b|"
    # Ethnic slurs
    r"\bжид\w*\b|\bхач\w*\b|\bчурк\w*\b|\bнигер\w*\b|\bчерножоп\w*\b|"
    # English profanity
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

# Author-address signals for regex fallback
_AUTHOR_ADDRESS_RE = re.compile(
    r"\b(вы|вас|вам|ваш|ваша|ваше|ваши|ты|тебя|тебе|тобой|твой|твоя|автор\w*)\b",
    re.IGNORECASE,
)

_BATCH_SIZE = 40
RequestLLMJsonFn = Any


def _has_offensive_language(text: str) -> bool:
    return bool(_TOXIC_RE.search(text) or _DEROGATORY_PHRASE_RE.search(text))


def _build_name_pattern(author_name: str) -> re.Pattern[str] | None:
    name = author_name.strip()
    if not name:
        return None
    parts = name.split()
    variants: list[str] = []
    for part in parts:
        if len(part) < 2:
            continue
        stem = part.rstrip("аеёийоуыьюя")
        if len(stem) < 2:
            stem = part[: max(2, len(part) - 1)]
        variants.append(re.escape(stem) + r"\w*")
    if not variants:
        return None
    return re.compile(r"\b(?:" + "|".join(variants) + r")\b", re.IGNORECASE | re.UNICODE)


def _build_toxic_author_prompt(
    numbered_comments: list[tuple[int, str]],
    author_name: str,
) -> str:
    lines = [f"{num}. {text[:400]}" for num, text in numbered_comments]
    comments_text = "\n".join(lines)

    return (
        f"Ты анализируешь комментарии под видео автора YouTube-канала по имени «{author_name}».\n"
        f"Канал — политический. Комментаторы часто обсуждают других людей "
        f"(политиков, общественных деятелей) и используют оскорбления в их адрес.\n\n"
        f"Все комментарии ниже содержат грубую лексику или оскорбления.\n"
        f"Твоя задача — определить, какие из них направлены ЛИЧНО на автора канала «{author_name}».\n\n"
        f"Комментарий направлен на автора если:\n"
        f"- Оскорбляет автора напрямую (по имени, «ты», «вы», «автор»)\n"
        f'- Оскорбляет позицию/мнение/контент автора ("несёшь бред", "чушь несёте")\n'
        f"- Использует сарказм/иронию с целью унизить именно автора\n"
        f"- Обращается к автору с грубой руганью\n\n"
        f"Комментарий НЕ направлен на автора если:\n"
        f"- Оскорбляет политиков, общественных деятелей, других комментаторов\n"
        f"- Использует мат как эмоциональное усиление, но обсуждает третьих лиц\n"
        f"- Ругает ситуацию/страну/правительство, а не автора лично\n\n"
        f"Комментарии:\n{comments_text}\n\n"
        f'Верни JSON-объект: {{"indices": [номера комментариев направленных на автора]}}\n'
        f'Если таких нет, верни: {{"indices": []}}'
    )


def _parse_llm_indices(raw: dict | list | None) -> list[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw if isinstance(x, (int, float))]
    if isinstance(raw, dict):
        for key in ("indices", "numbers", "result", "comments"):
            val = raw.get(key)
            if isinstance(val, list):
                return [int(x) for x in val if isinstance(x, (int, float))]
    return []


def classify_toxic(
    comments: list[Comment],
    author_name: str,
    llm_provider: LLMProvider | None = None,
    request_llm_json: RequestLLMJsonFn | None = None,
) -> list[int]:
    """Return IDs of comments that are toxic TOWARD THE AUTHOR specifically.

    Step 1: Pre-filter comments with any offensive language.
    Step 2: Use LLM (or regex fallback) to keep only author-directed ones.
    """
    # Step 1: Pre-filter — find all comments with offensive language
    offensive_comments: list[Comment] = []
    for comment in comments:
        text = (comment.text_raw or "").strip()
        if text and _has_offensive_language(text):
            offensive_comments.append(comment)

    if not offensive_comments:
        return []

    # Step 2: Determine which are directed at the author
    if (
        llm_provider is not None
        and request_llm_json is not None
        and not isinstance(llm_provider, NoLLMFallbackProvider)
    ):
        return _classify_toxic_llm(offensive_comments, author_name, llm_provider, request_llm_json)

    return _classify_toxic_heuristic(offensive_comments, author_name)


def _classify_toxic_llm(
    offensive_comments: list[Comment],
    author_name: str,
    llm_provider: LLMProvider,
    request_llm_json: RequestLLMJsonFn,
) -> list[int]:
    """Use LLM to determine which offensive comments target the author."""
    import logging

    from app.core.exceptions import BudgetExceededError, ExternalServiceError

    logger = logging.getLogger(__name__)
    result_ids: list[int] = []
    id_map: dict[int, int] = {}

    for batch_start in range(0, len(offensive_comments), _BATCH_SIZE):
        batch = offensive_comments[batch_start : batch_start + _BATCH_SIZE]
        numbered = []
        for i, comment in enumerate(batch, start=1):
            numbered.append((i, (comment.text_raw or "").strip()))
            id_map[batch_start + i] = comment.id

        prompt = _build_toxic_author_prompt(numbered, author_name)
        try:
            data = request_llm_json(
                llm_provider,
                prompt,
                task="appeal_toxic",
                estimated_out_tokens=200,
                max_output_tokens=400,
                system_prompt=(
                    f"Ты модератор комментариев YouTube-канала «{author_name}». "
                    "Определяй только оскорбления направленные на автора канала. "
                    "Отвечай строго JSON."
                ),
            )
        except (BudgetExceededError, ExternalServiceError) as exc:
            logger.warning("Toxic LLM classification failed: %s", exc)
            # Fallback for this batch
            batch_ids = _classify_toxic_heuristic(batch, author_name)
            result_ids.extend(batch_ids)
            continue
        except Exception as exc:
            logger.warning("Unexpected toxic LLM error: %s", exc)
            batch_ids = _classify_toxic_heuristic(batch, author_name)
            result_ids.extend(batch_ids)
            continue

        indices = _parse_llm_indices(data)
        for idx in indices:
            cid = id_map.get(batch_start + idx)
            if cid is not None:
                result_ids.append(cid)

    return result_ids


def _classify_toxic_heuristic(comments: list[Comment], author_name: str) -> list[int]:
    """Regex fallback: offensive + author-address signals."""
    name_re = _build_name_pattern(author_name)
    result: list[int] = []
    for comment in comments:
        text = (comment.text_raw or "").strip()
        if not text:
            continue
        has_author_ref = bool(_AUTHOR_ADDRESS_RE.search(text))
        has_name_ref = name_re is not None and bool(name_re.search(text))
        if has_author_ref or has_name_ref:
            result.append(comment.id)
    return result
