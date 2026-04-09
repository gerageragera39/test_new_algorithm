"""High-precision second-pass filter for toxic comment candidates.

Raises the precision of the ``toxic`` block to guard against false positives
in a system where toxic classification leads to automated moderation actions (auto-ban).

KEPT (confidence >= CONFIDENCE_THRESHOLD):
  - Contains explicit insult, degradation, or profanity DIRECTED AT the channel author.
  - Clear personal attacks aimed at the author by name, "ты/вы", "автор", "ведущий".
  - Unambiguous derogatory characterisations: "клоун", "идиот", "позор" addressed to the author.
  - Threats or wishes of harm directed at the author.

DROPPED (confidence < CONFIDENCE_THRESHOLD):
  - Sharp but non-offensive criticism of the author's position.
  - Sarcasm or satire without an explicit personal insult.
  - Profanity used as emotional colour when discussing events/politics, not aimed at the author.
  - Insults aimed at politicians, governments, armies, or third parties — not the author.
  - Any case where it is unclear whether the hostility is directed at the author specifically.

Principle: on any ambiguity → NOT toxic.  Better to miss a borderline case than to
incorrectly ban a viewer who expressed strong (but non-insulting) disagreement.

On LLM failure all candidates are retained unchanged (trust the upstream heuristic/LLM classifier).
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
_CONFIDENCE_THRESHOLD = 0.85  # minimum confidence to keep a comment as toxic
_AUTO_BAN_VERIFICATION_THRESHOLD = 0.90
RequestLLMJsonFn = Any


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _build_toxic_precision_prompt(
    numbered_comments: list[tuple[int, str]],
    author_name: str,
    guest_names: list[str] | None = None,
) -> str:
    lines = [f"{num}. {text}" for num, text in numbered_comments]
    comments_text = "\n".join(lines)
    guests_text = ", ".join(f"«{name}»" for name in guest_names or [] if name.strip()) or "не указаны"

    return (
        f"Ты специалист по модерации YouTube-канала «{author_name}» (политический русскоязычный канал).\n"
        f"Дополнительные гости текущего видео: {guests_text}.\n"
        f"Тебе дан список комментариев, предварительно помеченных как кандидаты на АВТОБАН.\n"
        f"Для каждого оцени, насколько ты уверен, что там есть ПРЯМОЕ оскорбление/унижение "
        f"или явная брань, направленные КОНКРЕТНО на автора «{author_name}», гостя из списка или сам контент канала.\n\n"
        f"Высокая уверенность (>= 0.85) ТОЛЬКО если:\n"
        f"  - Прямое личное оскорбление автора/гостя: по имени, «ты»/«вы», «автор», «ведущий»;\n"
        f"  - Явная брань/ругательства, адресованные автору, гостю или контенту канала (не обсуждению событий);\n"
        f"  - Унизительные характеристики автора/гостя/канала: «клоун», «идиот», «позор», «лжец» и т.п.;\n"
        f"  - Угрозы, оскорбительные пожелания, издёвки над личностью автора/гостя или над самим каналом.\n\n"
        f"Низкая уверенность (< 0.85) если:\n"
        f"  - Резкая, но не оскорбительная критика позиции или контента автора;\n"
        f"  - Сарказм, ирония без прямого личного оскорбления;\n"
        f"  - Мат как эмоциональное усиление при обсуждении событий — не в адрес автора;\n"
        f"  - Оскорбления политиков, правительств, армий, других комментаторов или иных третьих лиц;\n"
        f"  - Неоднозначно: непонятно, кому адресовано — автору/гостю/каналу или третьим лицам;\n"
        f"  - Это цитата, пересказ чужих слов, спор между комментаторами или насмешка не над каналом.\n\n"
        f"ПРИНЦИП: при любом сомнении давай низкую уверенность (< 0.85). "
        f"Лучше пропустить пограничный случай, чем ошибочно заблокировать.\n\n"
        f"Комментарии:\n{comments_text}\n\n"
        f"Верни строго JSON:\n"
        f'{{"results": {{"1": 0.95, "2": 0.30, "3": 0.88, ...}}}}\n'
        f"Значение: уверенность 0.0-1.0. Включи ВСЕ номера из списка."
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def _parse_toxic_precision_response(
    raw: dict | list | None,
    valid_nums: set[int],
) -> dict[int, float]:
    """Parse LLM response into {comment_num: confidence_score}.

    Missing or unparseable entries default to 0.0 (not confident → drop).
    """
    if not isinstance(raw, dict):
        return {}

    results = raw.get("results")
    if not isinstance(results, dict):
        results = raw

    parsed: dict[int, float] = {}
    for key, value in results.items():
        try:
            num = int(key)
        except (ValueError, TypeError):
            continue
        if num not in valid_nums:
            continue
        try:
            confidence = float(value)
            parsed[num] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            parsed[num] = 0.0  # unparseable → not confident → drop

    return parsed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def filter_toxic_precision(
    comments: list[Comment],
    llm_provider: LLMProvider,
    request_llm_json: RequestLLMJsonFn,
    author_name: str = "",
    guest_names: list[str] | None = None,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> list[int]:
    """Filter toxic candidates to only high-confidence direct insults at the author.

    Returns IDs of comments with confidence >= confidence_threshold.
    If LLM is unavailable (NoLLMFallbackProvider or call failure), returns all input IDs
    unchanged (trust the upstream classifier — the heuristic regex already requires
    explicit offensive language).
    """
    from app.services.labeling import NoLLMFallbackProvider

    if not comments:
        return []

    if isinstance(llm_provider, NoLLMFallbackProvider):
        logger.debug(
            "Toxic precision filter: no LLM, retaining all %d heuristic results", len(comments)
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

        prompt = _build_toxic_precision_prompt(numbered, author_name, guest_names)
        valid_nums = set(batch_id_map.keys())

        try:
            data = request_llm_json(
                llm_provider,
                prompt,
                task="toxic_precision_filter",
                estimated_out_tokens=max(60, len(batch) * 8),
                max_output_tokens=max(200, len(batch) * 15),
                system_prompt=(
                    "Ты строгий модератор. Оцени уверенность 0.0-1.0 в том, что комментарий — "
                    f"прямое оскорбление автора канала «{author_name}», гостя выпуска или контента канала. "
                    "Отвечай строго JSON. При сомнении давай низкую оценку."
                ),
            )
        except (BudgetExceededError, ExternalServiceError) as exc:
            logger.warning("Toxic precision filter LLM call failed: %s — retaining batch", exc)
            kept_ids.extend(batch_id_map.values())
            continue
        except Exception as exc:
            logger.warning("Toxic precision filter unexpected error: %s — retaining batch", exc)
            kept_ids.extend(batch_id_map.values())
            continue

        parsed = _parse_toxic_precision_response(data, valid_nums)
        batch_kept = 0
        batch_dropped = 0
        for num in valid_nums:
            confidence = parsed.get(num, 0.0)
            cid = batch_id_map.get(num)
            if cid is None:
                continue
            if confidence >= confidence_threshold:
                kept_ids.append(cid)
                batch_kept += 1
            else:
                batch_dropped += 1

        logger.debug(
            "Toxic precision filter batch: kept %d, dropped %d (threshold=%.2f)",
            batch_kept,
            batch_dropped,
            confidence_threshold,
        )

    return kept_ids


def _build_auto_ban_verification_prompt(
    numbered_comments: list[tuple[int, str, str]],
    author_name: str,
    guest_names: list[str],
) -> str:
    guests = ", ".join(f"«{name}»" for name in guest_names if name.strip()) or "не указаны"
    lines = [
        f"{num}. target={target}; comment={text}"
        for num, text, target in numbered_comments
    ]
    comments_text = "\n".join(lines)

    return (
        f"Ты модератор YouTube-канала автора «{author_name}».\n"
        f"Гости текущего видео: {guests}.\n\n"
        "Ниже комментарии, которые предыдущий этап хочет отправить в AUTO-BAN.\n"
        "Твоя задача — ОЧЕНЬ СТРОГО проверить, действительно ли комментарий нужно банить автоматически.\n\n"
        "Разрешай auto-ban ТОЛЬКО если одновременно выполнено всё:\n"
        "1. Есть прямое личное оскорбление или унижение.\n"
        "2. Оно направлено именно на автора, гостя или контент канала.\n"
        "3. Нет разумного сомнения, что текст относится к третьему лицу, другому комментатору, цитате или сарказму без прямого оскорбления.\n\n"
        "Запрещено auto-ban, если:\n"
        "- оскорбление направлено на политика, общественного деятеля, страну, армию или другое третье лицо;\n"
        "- это спор между комментаторами;\n"
        "- это сарказм/ирония без явного личного оскорбления автора/гостя;\n"
        "- это резкая критика или насмешка над позицией, но не прямой insult;\n"
        "- адресат неясен.\n\n"
        "Для каждого комментария верни:\n"
        "- allow_auto_ban: true/false\n"
        "- confidence: 0.0-1.0\n"
        "- target_confirmed: author|guest|content|undefined|third_party\n\n"
        "Комментарии:\n"
        f"{comments_text}\n\n"
        "Верни строго JSON:\n"
        '{"results": {"1": {"allow_auto_ban": true, "confidence": 0.97, "target_confirmed": "author"}, "2": {"allow_auto_ban": false, "confidence": 0.21, "target_confirmed": "third_party"}}}'
    )


def _parse_auto_ban_verification_response(
    raw: dict | list | None,
    valid_nums: set[int],
) -> dict[int, dict[str, Any]]:
    """Parse auto-ban verification response into structured decisions."""
    if not isinstance(raw, dict):
        return {}

    results = raw.get("results")
    if not isinstance(results, dict):
        results = raw

    parsed: dict[int, dict[str, Any]] = {}
    for key, value in results.items():
        try:
            num = int(key)
        except (TypeError, ValueError):
            continue
        if num not in valid_nums or not isinstance(value, dict):
            continue

        allow_auto_ban = bool(value.get("allow_auto_ban", False))
        try:
            confidence = float(value.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        target_confirmed = str(value.get("target_confirmed", "undefined")).strip().lower()
        if target_confirmed not in {"author", "guest", "content", "undefined", "third_party"}:
            target_confirmed = "undefined"

        parsed[num] = {
            "allow_auto_ban": allow_auto_ban,
            "confidence": max(0.0, min(1.0, confidence)),
            "target_confirmed": target_confirmed,
        }
    return parsed


def verify_auto_ban_candidates(
    comments: list[Comment],
    *,
    author_name: str,
    guest_names: list[str],
    toxic_metadata: dict[int, dict[str, Any]],
    llm_provider: LLMProvider,
    request_llm_json: RequestLLMJsonFn,
    confidence_threshold: float = _AUTO_BAN_VERIFICATION_THRESHOLD,
) -> dict[int, dict[str, Any]]:
    """Strict final LLM veto for auto-ban candidates.

    Returns only candidates that passed the final verification.
    On missing LLM or any request failure it fails closed: nothing is approved
    for auto-ban, so the caller can downgrade the candidates to manual review.
    """
    from app.services.labeling import NoLLMFallbackProvider

    if not comments or isinstance(llm_provider, NoLLMFallbackProvider):
        return {}

    approved: dict[int, dict[str, Any]] = {}

    for batch_start in range(0, len(comments), _BATCH_SIZE):
        batch = comments[batch_start : batch_start + _BATCH_SIZE]
        numbered: list[tuple[int, str, str]] = []
        batch_id_map: dict[int, int] = {}

        for i, comment in enumerate(batch, start=1):
            text = (comment.text_raw or "").strip()
            candidate_target = str(
                toxic_metadata.get(comment.id, {}).get("insult_target", "undefined")
            )
            numbered.append((i, text, candidate_target))
            batch_id_map[i] = comment.id

        prompt = _build_auto_ban_verification_prompt(numbered, author_name, guest_names)
        valid_nums = set(batch_id_map.keys())

        try:
            data = request_llm_json(
                llm_provider,
                prompt,
                task="toxic_auto_ban_verification",
                estimated_out_tokens=max(80, len(batch) * 10),
                max_output_tokens=max(240, len(batch) * 18),
                system_prompt=(
                    "Ты строгий модератор. Одобряй auto-ban только для однозначных прямых "
                    "оскорблений автора, гостя или контента. При малейшем сомнении отвечай "
                    "allow_auto_ban=false. Формат ответа — строго JSON."
                ),
            )
        except (BudgetExceededError, ExternalServiceError) as exc:
            logger.warning("Auto-ban verification LLM call failed: %s — downgrading batch", exc)
            continue
        except Exception as exc:
            logger.warning("Auto-ban verification unexpected error: %s — downgrading batch", exc)
            continue

        parsed = _parse_auto_ban_verification_response(data, valid_nums)
        for num, decision in parsed.items():
            cid = batch_id_map.get(num)
            if cid is None:
                continue
            if (
                decision["allow_auto_ban"]
                and decision["target_confirmed"] in {"author", "guest", "content"}
                and decision["confidence"] >= confidence_threshold
            ):
                approved[cid] = decision

    return approved
