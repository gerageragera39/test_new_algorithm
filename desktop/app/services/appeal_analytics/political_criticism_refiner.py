"""Second-pass filter: retain only constructive POLITICAL criticism of the author's position.

After unified classification produces ``criticism`` candidates, this module runs
a focused LLM pass to distinguish genuine political-analytical criticism from
off-topic complaints, personal remarks, and unsupported emotional reactions.

KEPT (keep=true):
  - Constructive disagreement with the author's political position, assessments, or arguments.
  - Points to a factual error, logical inconsistency, bias, or one-sided framing by the author.
  - Criticises the author's choice of sources, methodology, or analytical approach.
  - Backed by a specific argument, counter-example, or factual reference.

DROPPED (keep=false):
  - Comments about the author's appearance, voice, manner of speech, age, or personal traits.
  - Pure emotional reactions or label-only statements without any argument ("не согласен").
  - Off-topic discussion or commentary about third parties with no link to the author's position.
  - Statements that merely repeat the author's words without critique.

On LLM failure or unavailability all candidates are retained (fail-open = no data loss).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from app.core.exceptions import BudgetExceededError, ExternalServiceError

if TYPE_CHECKING:
    from app.db.models import Comment
    from app.services.labeling import LLMProvider

logger = logging.getLogger(__name__)

_BATCH_SIZE = 20
RequestLLMJsonFn = Any


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _build_political_criticism_prompt(
    numbered_comments: list[tuple[int, str]],
    author_name: str,
) -> str:
    lines = [f"{num}. {text}" for num, text in numbered_comments]
    comments_text = "\n".join(lines)

    return (
        f"Ты аналитик комментариев YouTube-канала «{author_name}» (политический русскоязычный канал).\n"
        f"Тебе дан список комментариев, предварительно отобранных как «критика» в адрес автора.\n"
        f"Твоя задача: решить, является ли каждый комментарий КОНСТРУКТИВНОЙ КРИТИКОЙ "
        f"ПОЛИТИЧЕСКОЙ ПОЗИЦИИ автора «{author_name}».\n\n"
        f"ОСТАВЛЯЙ (true) если комментарий:\n"
        f"  1. Содержит аргументированное несогласие с политической позицией, оценками или "
        f"аргументацией автора в видео;\n"
        f"  2. Указывает на фактическую ошибку, логическое противоречие, предвзятость или "
        f"однобокость в подаче материала автором;\n"
        f"  3. Критикует выбор источников, фреймирование темы или аналитический подход автора;\n"
        f"  4. Содержит конкретный аргумент, контрпример или ссылку на факты — "
        f"а не просто несогласие.\n\n"
        f"ОТСЕИВАЙ (false) если комментарий:\n"
        f"  1. Критикует внешность, голос, манеру речи, возраст или личные черты автора;\n"
        f"  2. Содержит только эмоцию или ярлык без аргумента "
        f"(«не согласен», «неправда», «чушь»);\n"
        f"  3. Обсуждает действия третьих лиц (политиков, правительств) без связи с позицией автора;\n"
        f"  4. Является оффтопом или темой, не связанной с содержанием и позицией автора;\n"
        f"  5. Просто повторяет слова автора без критической оценки.\n\n"
        f"ВАЖНО: При сомнении — оставляй (true). Лучше пропустить пограничный случай, "
        f"чем потерять настоящую критику.\n\n"
        f"Комментарии:\n{comments_text}\n\n"
        f"Верни строго JSON:\n"
        f'{{"results": {{"1": true, "2": false, "3": true, ...}}}}\n'
        f"true = оставить, false = отсеять. Включи ВСЕ номера из списка."
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_political_criticism_response(
    raw: dict | list | None,
    valid_nums: set[int],
) -> dict[int, bool]:
    """Parse LLM response into {comment_num: keep_bool}.

    Defaults to True (keep) for any missing or unparseable entries.
    """
    if not isinstance(raw, dict):
        return dict.fromkeys(valid_nums, True)

    results = raw.get("results")
    if not isinstance(results, dict):
        results = raw

    parsed: dict[int, bool] = {}
    for key, value in results.items():
        try:
            num = int(key)
        except (ValueError, TypeError):
            continue
        if num not in valid_nums:
            continue
        if isinstance(value, bool):
            parsed[num] = value
        elif isinstance(value, int):
            parsed[num] = bool(value)
        elif isinstance(value, str):
            parsed[num] = value.strip().lower() in ("true", "yes", "1", "keep")
        else:
            parsed[num] = True  # unknown type → keep

    # Default any missing nums to True (keep)
    for n in valid_nums:
        if n not in parsed:
            parsed[n] = True

    return parsed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def filter_political_criticism(
    comments: list[Comment],
    llm_provider: LLMProvider,
    request_llm_json: RequestLLMJsonFn,
    author_name: str = "",
) -> list[int]:
    """Filter criticism candidates to retain only constructive political criticism.

    Returns a list of comment IDs that pass the filter.
    If LLM is unavailable (NoLLMFallbackProvider or call failure), returns all input IDs
    unchanged so no data is silently lost.
    """
    from app.services.labeling import NoLLMFallbackProvider

    if not comments:
        return []

    if isinstance(llm_provider, NoLLMFallbackProvider):
        logger.debug(
            "Political criticism filter: no LLM, retaining all %d candidates", len(comments)
        )
        return [c.id for c in comments]

    kept_ids: list[int] = []

    for batch_start in range(0, len(comments), _BATCH_SIZE):
        batch = comments[batch_start : batch_start + _BATCH_SIZE]
        numbered: list[tuple[int, str]] = []
        batch_id_map: dict[int, int] = {}

        for i, comment in enumerate(batch, start=1):
            text = (comment.text_raw or "").strip()
            numbered.append((i, text))
            batch_id_map[i] = comment.id

        prompt = _build_political_criticism_prompt(numbered, author_name)
        valid_nums = set(batch_id_map.keys())

        try:
            data = request_llm_json(
                llm_provider,
                prompt,
                task="political_criticism_filter",
                estimated_out_tokens=max(60, len(batch) * 8),
                max_output_tokens=max(200, len(batch) * 15),
                system_prompt=(
                    "Ты строгий фильтр политической конструктивной критики. "
                    "Отвечай строго JSON. Для каждого номера: true=оставить, false=отсеять."
                ),
            )
        except (BudgetExceededError, ExternalServiceError) as exc:
            logger.warning("Political criticism filter LLM call failed: %s — retaining batch", exc)
            kept_ids.extend(batch_id_map.values())
            continue
        except Exception as exc:
            logger.warning("Political criticism filter unexpected error: %s — retaining batch", exc)
            kept_ids.extend(batch_id_map.values())
            continue

        parsed = _parse_political_criticism_response(data, valid_nums)
        batch_kept = 0
        batch_dropped = 0
        for num, keep in parsed.items():
            cid = batch_id_map.get(num)
            if cid is not None:
                if keep:
                    kept_ids.append(cid)
                    batch_kept += 1
                else:
                    batch_dropped += 1

        logger.debug(
            "Political criticism filter batch: kept %d, dropped %d",
            batch_kept,
            batch_dropped,
        )

    return kept_ids
