"""Unified single-pass LLM classifier for appeal analytics.

Instead of classifying comments through multiple separate LLM calls
(toxic, then criticism, then questions), this module sends each batch
through ONE prompt that classifies every comment into one of:
  - toxic: insults/offensive language directed at the channel author,
    configured guests, or the channel content
  - criticism: constructive criticism of the author's position/content
  - question: constructive questions addressed to the author
  - appeal: direct requests/proposals to the author (NOT gratitude)
  - skip: not directed at the author or configured toxic targets, or simple gratitude/praise

This single-pass approach gives the LLM full context to distinguish
between categories and avoids cascading misclassification errors.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.core.exceptions import BudgetExceededError, ExternalServiceError
from app.services.labeling import LLMProvider, NoLLMFallbackProvider

if TYPE_CHECKING:
    from app.db.models import Comment

logger = logging.getLogger(__name__)

# ---- Regex patterns for heuristic fallback ----

_QUESTION_RE = re.compile(r"\?|¿|？")
_QUESTION_SIGNAL_RE = re.compile(
    r"\b(почему|зачем|как|какой|какая|какие|когда|где|что|кто|можно ли|есть ли)\b",
    re.IGNORECASE,
)
_CRITICISM_SIGNAL_RE = re.compile(
    r"\b(не\s+соглас|неправ|ошиба|заблуждени|предвзят|"
    r"не\s+верю|искажа|нужно\s+было|стоило\s+бы|лучше\s+бы|надо\s+было|"
    r"не\s+так|неправильно|проблема\s+в\s+том|не\s+учит|"
    r"слабый\s+аргумент|не\s+убедительно|спорн|сомнительн)\b",
    re.IGNORECASE,
)
_OFFENSIVE_RE = re.compile(
    r"\b(бляд|бля\b|сука|суч|хуй|хуе|хуё|пизд|ебан|ебат|ёбан|мудак|мудил|долбо[её]|"
    r"дебил|идиот|мраз|твар|ублюд|гандон|гнид|чмо|"
    r"урод|падл|мразот|подонок|подонк|"
    r"fuck|shit|bitch|asshole|moron|retard|cunt|dumbass)\b",
    re.IGNORECASE,
)
_DEROGATORY_PHRASE_RE = re.compile(
    r"(?:"
    r"что за бред|закрой рот|заткни\w*|пошёл на\w*|пошел на\w*|"
    r"иди на\w*|катись|вали отсюда|рот закрой|"
    r"больной на голову|клоун|клоуны|позор|позорище|"
    r"несёт чушь|несет чушь|несёт бред|несет бред|городит чушь"
    r")",
    re.IGNORECASE | re.UNICODE,
)
_AUTHOR_ADDRESS_RE = re.compile(
    r"\b(автор\w*)\b",
    re.IGNORECASE,
)
_GRATITUDE_RE = re.compile(
    r"\b(спасибо|благодар|молодец|молодчин|лучший канал|так держать|"
    r"здоровья|удачи|успехов|браво|респект|класс|супер|отлично|"
    r"замечательн|прекрасн|великолепн|превосходн)\b",
    re.IGNORECASE,
)
_REQUEST_RE = re.compile(
    r"\b(сними|снимите|расскажи|расскажите|сделай|сделайте|пригласи|пригласите|"
    r"рассмотри|рассмотрите|разбери|разберите|прокомментируй|прокомментируйте|"
    r"пожалуйста|можно ли|хотелось бы|было бы|предлагаю|прошу|попрос)\b",
    re.IGNORECASE,
)

_BATCH_SIZE = 30  # Slightly smaller batches for the richer prompt

RequestLLMJsonFn = Any

# Valid category keys returned by the LLM
_VALID_CATEGORIES = {"toxic", "criticism", "question", "appeal", "skip"}


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


def _build_name_patterns(names: list[str] | None) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for name in names or []:
        pattern = _build_name_pattern(name)
        if pattern is not None:
            patterns.append(pattern)
    return patterns


def _has_offensive_language(text: str) -> bool:
    return bool(_OFFENSIVE_RE.search(text) or _DEROGATORY_PHRASE_RE.search(text))


def _clip_words(text: str, max_words: int = 220) -> str:
    """Clip text to at most max_words words using head+tail strategy.

    Default is 220 words (raised from 150) so long comments with a question
    near the end still reach the LLM intact.

    The head+tail approach preserves both the opening context and any
    question that appears at the end of a long comment:
      - If text fits within max_words, return as-is.
      - Otherwise keep the first (max_words - tail_size) words + '[...]' + last tail_size words.

    The tail window is 40 words (about 2 sentences), ensuring trailing
    questions are always visible in the LLM prompt.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    tail_size = min(40, max_words // 3)
    head_size = max_words - tail_size
    head = " ".join(words[:head_size])
    tail = " ".join(words[-tail_size:])
    return f"{head} [...] {tail}"


def has_question_signal(text: str) -> bool:
    """Return True if text contains a question-mark or an interrogative word.

    Used by the runner to identify criticism candidates that might be genuine
    questions and should be promoted after the Question Refiner pass.
    """
    return bool(_QUESTION_RE.search(text) or _QUESTION_SIGNAL_RE.search(text))


def _build_unified_prompt(
    numbered_comments: list[tuple[int, str]],
    author_name: str,
    guest_names: list[str] | None = None,
) -> str:
    lines = [f"{num}. {_clip_words(text)}" for num, text in numbered_comments]
    comments_text = "\n".join(lines)
    guests_text = (
        ", ".join(f"«{name}»" for name in guest_names or [] if name.strip()) or "не указаны"
    )

    return (
        f"Ты модератор и аналитик комментариев YouTube-канала автора по имени «{author_name}».\n"
        f"Дополнительные гости текущего видео: {guests_text}.\n"
        f"Канал — политический, русскоязычный. Комментаторы активно обсуждают политиков, общественных деятелей, "
        f"события в мире и стране. Большинство комментариев НЕ адресованы автору канала — "
        f"они обсуждают третьих лиц.\n\n"
        f"КРИТИЧЕСКИ ВАЖНО: Ты должен отличать комментарии, адресованные ЛИЧНО автору канала "
        f"«{author_name}», ГОСТЯМ из списка выше или самому контенту канала, от комментариев о ТРЕТЬИХ ЛИЦАХ "
        f"(политиках, общественных деятелях, "
        f"других комментаторах).\n\n"
        f"Признаки того, что комментарий адресован автору «{author_name}»:\n"
        f"- Прямое обращение по имени или фамилии автора\n"
        f"- Обращение «автор», «ведущий», «блогер» и т.п.\n"
        f"- Обращение на «вы»/«ты» В КОНТЕКСТЕ обращения к автору (а не к политикам/третьим лицам)\n"
        f"- Обсуждение позиции/мнения/слов/контента именно автора канала\n"
        f"- Реакция на то, что автор сказал в видео\n\n"
        f"Признаки того, что комментарий НЕ адресован автору:\n"
        f"- Обсуждает действия политиков, правительств, армий, стран\n"
        f"- «Вы»/«ты» обращено к политику, правительству, народу, другому комментатору\n"
        f"- Высказывает мнение о геополитических событиях без упоминания автора\n"
        f"- Отвечает другому комментатору\n"
        f"- Просто выражает эмоции о ситуации в мире/стране\n\n"
        f"Для каждого комментария определи его категорию:\n\n"
        f"«toxic» — Оскорбления, грубость, токсичность направленные ЛИЧНО на автора канала «{author_name}», "
        f"указанных гостей или контент канала:\n"
        f"  - Прямые оскорбления автора (по имени, «ты»/«вы», «автор»)\n"
        f"  - Прямые оскорбления гостей из списка выше\n"
        f"  - Грубая ругань в адрес автора, гостей или контента ("
        f'"несёшь бред", "чушь несёте", "клоун", "позор")\n'
        f"  - Унизительный сарказм, издёвки, уничижительные характеристики автора или гостя\n"
        f"  - Пожелания зла, угрозы автору или гостю\n"
        f"  НЕ включай: оскорбления политиков/третьих лиц, мат как эмоциональное усиление "
        f"при обсуждении событий, грубость в адрес других комментаторов\n\n"
        f"«criticism» — Конструктивная критика позиции/мнения/контента именно автора «{author_name}»:\n"
        f"  - Аргументированное несогласие с тем, что автор сказал в видео\n"
        f"  - Указание на фактические ошибки, логические противоречия или неточности автора\n"
        f"  - Замечания о предвзятости, однобокости или необъективности подачи автора\n"
        f"  - Предложения по улучшению контента или подхода автора\n"
        f'  - Комментарий должен содержать АРГУМЕНТ или КОНКРЕТНОЕ указание, а не просто "не согласен"\n'
        f"  НЕ включай: критику политиков/третьих лиц, простые негативные реакции без аргументов, "
        f"оскорбления (это toxic), риторические вопросы без содержательной критики\n\n"
        f"«question» — Конструктивные вопросы к автору «{author_name}»:\n"
        f"  - Просьбы раскрыть/уточнить позицию автора по теме\n"
        f"  - Вопросы о планах контента, источниках информации автора\n"
        f"  - Просьбы прокомментировать конкретное событие или факт\n"
        f"  - Вопрос должен быть ИСКРЕННИМ — реально запрашивать информацию или мнение\n"
        f"  - ВАЖНО: если комментарий содержит одновременно критику И вопрос к автору, "
        f"классифицируй как question (вопрос важнее)\n"
        f"  НЕ включай: риторические вопросы (способ выразить несогласие — это criticism или toxic), "
        f"вопросы к другим комментаторам/третьим лицам, тролль-вопросы, "
        f'вопросы-оскорбления ("ты совсем тупой?")\n\n'
        f"«appeal» — Конкретные просьбы и предложения к автору «{author_name}»:\n"
        f"  - Просьбы снять видео на определённую тему\n"
        f"  - Предложения пригласить гостя, связаться с кем-то, рассмотреть источник\n"
        f"  - Предложения по формату, оформлению, расписанию контента\n"
        f"  - Деловые предложения о сотрудничестве\n"
        f"  НЕ включай: благодарности, похвалы, выражения поддержки и солидарности "
        f'("спасибо", "молодец", "лучший канал", "так держать") — это skip.\n'
        f"  НЕ включай: комментарии, где просто упоминается имя автора, "
        f"но адресованы они другим людям или ситуации\n\n"
        f"«skip» — Всё остальное:\n"
        f"  - Обсуждение политиков, общественных деятелей, стран\n"
        f"  - Дискуссии между комментаторами\n"
        f"  - Общие высказывания о событиях\n"
        f"  - Спам, реклама, нерелевантные сообщения\n"
        f"  - Благодарности, похвалы, пожелания, выражения поддержки автору "
        f'("спасибо за видео", "отличный выпуск", "здоровья вам")\n'
        f"  skip — это ТОЛЬКО явно нерелевантное, пустое или совершенно неразбираемое.\n\n"
        f"ПРИОРИТЕТ при пересечении категорий (от высшего к низшему):\n"
        f"question > criticism > appeal > toxic > skip\n"
        f"При сомнении сохраняй комментарий в ближайшую полезную категорию — не теряй его в skip.\n\n"
        f"ОЦЕНКА КОНСТРУКТИВНОСТИ (только для criticism и question):\n"
        f"Для категорий criticism и question добавь оценку от 1 до 10 через двоеточие.\n"
        f"Оценка отражает, насколько комментарий полезен и интересен для автора канала:\n"
        f"  10 — конкретный аргумент с фактами/примерами, или точный вопрос по теме видео\n"
        f"  7-9 — содержательный, но менее конкретный\n"
        f"  4-6 — имеет смысл, но размытый или частично не по теме\n"
        f"  1-3 — едва конструктивный, очень общий\n"
        f"Для toxic, appeal, skip оценку НЕ ставь.\n\n"
        f"Комментарии:\n{comments_text}\n\n"
        f"Верни JSON-объект в формате:\n"
        f'{{"results": {{"1": "criticism:8", "2": "skip", "3": "question:6", "4": "toxic", ...}}}}\n'
        f"Где категория — одно из: toxic, criticism, question, appeal, skip.\n"
        f"Для criticism и question ОБЯЗАТЕЛЬНО добавь :оценка (число 1-10).\n"
        f"Включи ВСЕ номера комментариев из списка. "
        f"Приоритет при сомнениях: question > criticism > appeal > toxic > skip."
    )


def _parse_unified_response(
    raw: dict | list | None,
    valid_nums: set[int],
) -> dict[int, tuple[str, int]]:
    """Parse the LLM response into {comment_number: (category, score)}.

    Supports formats like "criticism:8" (category with score) and plain "toxic".
    Score defaults to 5 for criticism/question if not provided, 0 for others.
    """
    if not isinstance(raw, dict):
        return {}

    results = raw.get("results")
    if not isinstance(results, dict):
        results = raw

    parsed: dict[int, tuple[str, int]] = {}
    for key, value in results.items():
        try:
            num = int(key)
        except (ValueError, TypeError):
            continue
        if num not in valid_nums:
            continue
        raw_val = str(value).strip().lower()
        score = 0
        if ":" in raw_val:
            parts = raw_val.rsplit(":", 1)
            cat = parts[0].strip()
            try:
                score = max(1, min(10, int(parts[1].strip())))
            except (ValueError, TypeError):
                score = 5 if cat in ("criticism", "question") else 0
        else:
            cat = raw_val
            score = 5 if cat in ("criticism", "question") else 0
        if cat in _VALID_CATEGORIES:
            parsed[num] = (cat, score)
    return parsed


class ClassificationResult:
    """Holds classified comment IDs and their constructiveness scores."""

    __slots__ = ("ids", "scores")

    def __init__(self) -> None:
        self.ids: dict[str, list[int]] = {
            "toxic": [],
            "criticism": [],
            "question": [],
            "appeal": [],
        }
        self.scores: dict[int, int] = {}  # comment_id -> score (1-10)

    def get(self, key: str, default: list[int] | None = None) -> list[int]:
        return self.ids.get(key, default or [])


def classify_unified_llm(
    comments: list[Comment],
    llm_provider: LLMProvider,
    request_llm_json: RequestLLMJsonFn,
    author_name: str = "",
    guest_names: list[str] | None = None,
) -> ClassificationResult:
    """Classify all comments in a single pass.

    Returns ClassificationResult with category lists and per-comment scores.
    Categories: toxic, criticism, question, appeal.
    Comments classified as 'skip' are not included.
    """
    if isinstance(llm_provider, NoLLMFallbackProvider):
        ids, scores = _classify_batch_heuristic_scored(comments, author_name, guest_names)
        cr = ClassificationResult()
        cr.ids = ids
        cr.scores = scores
        return cr

    result = ClassificationResult()

    for batch_start in range(0, len(comments), _BATCH_SIZE):
        batch = comments[batch_start : batch_start + _BATCH_SIZE]
        numbered: list[tuple[int, str]] = []
        batch_id_map: dict[int, int] = {}

        for i, comment in enumerate(batch, start=1):
            text = (comment.text_raw or "").strip()
            numbered.append((i, text))
            batch_id_map[i] = comment.id

        prompt = _build_unified_prompt(numbered, author_name, guest_names)

        try:
            data = request_llm_json(
                llm_provider,
                prompt,
                task="appeal_unified",
                estimated_out_tokens=max(200, len(batch) * 18),
                max_output_tokens=max(400, len(batch) * 25),
                system_prompt=(
                    f"Ты модератор YouTube-канала «{author_name}». "
                    f"Классифицируй комментарии строго по указанным категориям. "
                    f"Для criticism и question обязательно ставь оценку через двоеточие. "
                    f"Отвечай строго JSON. Приоритет: question > criticism > appeal > toxic > skip."
                ),
            )
        except (BudgetExceededError, ExternalServiceError) as exc:
            logger.warning("Unified LLM classification failed: %s", exc)
            fallback_ids, fallback_scores = _classify_batch_heuristic_scored(
                batch, author_name, guest_names
            )
            for cat, category_ids in fallback_ids.items():
                result.ids[cat].extend(category_ids)
            result.scores.update(fallback_scores)
            continue
        except Exception as exc:
            logger.warning("Unexpected unified LLM error: %s", exc)
            fallback_ids, fallback_scores = _classify_batch_heuristic_scored(
                batch, author_name, guest_names
            )
            for cat, category_ids in fallback_ids.items():
                result.ids[cat].extend(category_ids)
            result.scores.update(fallback_scores)
            continue

        valid_nums = set(batch_id_map.keys())
        parsed = _parse_unified_response(data, valid_nums)

        for num, (cat, score) in parsed.items():
            if cat == "skip":
                # Enhanced safety: Don't silently drop skip-category comments
                # Re-evaluate with heuristic to catch valuable content LLM might have missed
                cid = batch_id_map.get(num)
                if cid is not None:
                    comment = next((c for c in batch if c.id == cid), None)
                    if comment:
                        heuristic_cat = _classify_heuristic_single(
                            comment, author_name, guest_names
                        )
                        if heuristic_cat != "skip":
                            # Heuristic found value - use it instead
                            result.ids[heuristic_cat].append(cid)
                            result.scores[cid] = 4  # Medium-low score for fallback
                            logger.debug(
                                "Skip-category comment %d reclassified to '%s' by heuristic",
                                num, heuristic_cat
                            )
                continue
            cid = batch_id_map.get(num)
            if cid is not None and cat in result.ids:
                result.ids[cat].append(cid)
                if score > 0:
                    result.scores[cid] = score

        # Missed-nums fallback: if LLM did not return some comment numbers,
        # apply heuristic so no comment is silently lost.
        missed_nums = valid_nums - set(parsed.keys())
        if missed_nums:
            missed_comments = [batch[num - 1] for num in missed_nums if 1 <= num <= len(batch)]
            if missed_comments:
                logger.debug(
                    "LLM omitted %d/%d comments in batch; applying heuristic fallback",
                    len(missed_comments),
                    len(batch),
                )
                fb_ids, fb_scores = _classify_batch_heuristic_scored(
                    missed_comments, author_name, guest_names
                )
                for cat, category_ids in fb_ids.items():
                    result.ids[cat].extend(category_ids)
                result.scores.update(fb_scores)

    return result


# ---- Heuristic fallback (when LLM is unavailable) ----

_ARGUMENT_SIGNAL_RE = re.compile(
    r"\b(потому что|так как|например|факт|доказательств|источник|ссылк|"
    r"по данным|согласно|статистик|исследовани|история|на самом деле|"
    r"в действительности|однако|вместе с тем|при этом)\b",
    re.IGNORECASE,
)


def _heuristic_score(text: str, category: str) -> int:
    """Estimate constructiveness score (1-10) using text features."""
    words = text.split()
    word_count = len(words)
    score = 5.0

    # Length bonus: 8-40 words is ideal
    if 8 <= word_count <= 40:
        score += 1.0
    elif word_count > 40:
        score += 0.5
    elif word_count < 5:
        score -= 1.5

    # Argument markers boost
    if _ARGUMENT_SIGNAL_RE.search(text):
        score += 1.5

    # Multiple criticism signals = more substantive
    criticism_matches = len(_CRITICISM_SIGNAL_RE.findall(text))
    if criticism_matches >= 2:
        score += 1.0

    # ALL CAPS penalty
    if text.isupper() and len(text) > 15:
        score -= 2.0

    # Very short = likely low quality
    if word_count <= 4:
        score -= 1.0

    # For questions: actual question mark is a strong signal
    if category == "question" and _QUESTION_SIGNAL_RE.search(text):
        score += 0.5

    return max(1, min(10, int(round(score))))


def _classify_heuristic_single(
    comment: Comment,
    author_name: str,
    guest_names: list[str] | None = None,
) -> str:
    """Classify a single comment using regex heuristics. Returns category string."""
    text = (comment.text_raw or "").strip()
    if not text or len(text.split()) < 3:
        return "skip"
    
    name_re = _build_name_pattern(author_name)
    guest_res = _build_name_patterns(guest_names)
    content_re = re.compile(
        r"\b(видео|ролик|канал|выпуск|эфир|контент|передач\w+|програм\w+)\b",
        re.IGNORECASE | re.UNICODE,
    )
    
    has_name_ref = name_re is not None and bool(name_re.search(text))
    has_author_word = bool(_AUTHOR_ADDRESS_RE.search(text))
    has_author_ref = has_name_ref or has_author_word
    has_guest_ref = any(pattern.search(text) for pattern in guest_res)
    has_content_ref = bool(content_re.search(text))
    
    has_offensive = _has_offensive_language(text)
    has_question = bool(_QUESTION_RE.search(text))
    has_criticism_signal = bool(_CRITICISM_SIGNAL_RE.search(text))
    has_gratitude = bool(_GRATITUDE_RE.search(text))
    has_request = bool(_REQUEST_RE.search(text))
    
    # Toxic if offensive + directed at author/guest/content
    if has_offensive and (has_author_ref or has_guest_ref or has_content_ref):
        return "toxic"
    
    # Not directed at author → skip
    if not has_author_ref:
        return "skip"
    
    # Offensive + author-directed = toxic
    if has_offensive:
        return "toxic"
    
    # Question priority over criticism
    if has_question and len(text.split()) >= 4:
        return "question"
    
    # Criticism signal
    if has_criticism_signal and len(text.split()) >= 5:
        return "criticism"
    
    # Appeal/request
    if has_request and not has_gratitude:
        return "appeal"
    
    # Default: skip (gratitude or no clear signal)
    return "skip"


def _classify_batch_heuristic(
    comments: list[Comment],
    author_name: str,
    guest_names: list[str] | None = None,
) -> dict[str, list[int]]:
    """Classify a batch of comments using regex heuristics."""
    result: dict[str, list[int]] = {
        "toxic": [],
        "criticism": [],
        "question": [],
        "appeal": [],
    }
    name_re = _build_name_pattern(author_name)
    guest_res = _build_name_patterns(guest_names)
    content_re = re.compile(
        r"\b(видео|ролик|канал|выпуск|эфир|контент|передач\w+|програм\w+)\b",
        re.IGNORECASE | re.UNICODE,
    )

    for comment in comments:
        text = (comment.text_raw or "").strip()
        if not text or len(text.split()) < 3:
            continue

        has_name_ref = name_re is not None and bool(name_re.search(text))
        has_author_word = bool(_AUTHOR_ADDRESS_RE.search(text))
        has_author_ref = has_name_ref or has_author_word
        has_guest_ref = any(pattern.search(text) for pattern in guest_res)
        has_content_ref = bool(content_re.search(text))

        has_offensive = _has_offensive_language(text)
        has_question = bool(_QUESTION_RE.search(text))
        has_criticism_signal = bool(_CRITICISM_SIGNAL_RE.search(text))
        has_gratitude = bool(_GRATITUDE_RE.search(text))
        has_request = bool(_REQUEST_RE.search(text))

        if has_offensive and (has_author_ref or has_guest_ref or has_content_ref):
            result["toxic"].append(comment.id)
            continue

        if not has_author_ref:
            continue

        if has_offensive:
            result["toxic"].append(comment.id)
        elif has_question and len(text.split()) >= 4:
            # Question priority over criticism (question > criticism)
            result["question"].append(comment.id)
        elif has_criticism_signal and len(text.split()) >= 5:
            result["criticism"].append(comment.id)
        elif has_request and not has_gratitude:
            result["appeal"].append(comment.id)
        # Gratitude without request → skip (not added to any category)

    return result


def _classify_batch_heuristic_scored(
    comments: list[Comment],
    author_name: str,
    guest_names: list[str] | None = None,
) -> tuple[dict[str, list[int]], dict[int, int]]:
    """Classify + score a batch of comments using regex heuristics."""
    ids = _classify_batch_heuristic(comments, author_name, guest_names)
    comment_map = {c.id: c for c in comments}
    scores: dict[int, int] = {}
    for cat in ("criticism", "question"):
        for cid in ids.get(cat, []):
            c = comment_map.get(cid)
            if c:
                scores[cid] = _heuristic_score((c.text_raw or "").strip(), cat)
    return ids, scores


def classify_unified_heuristic(
    comments: list[Comment],
    author_name: str = "",
    guest_names: list[str] | None = None,
) -> dict[str, list[int]]:
    """Full heuristic fallback for all comments."""
    return _classify_batch_heuristic(comments, author_name, guest_names)


# ---- Legacy compatibility wrappers (kept for potential direct usage) ----


def classify_criticism_heuristic(comments: list[Comment], author_name: str = "") -> list[int]:
    """Regex fallback: criticism signals + author-address signals."""
    result = _classify_batch_heuristic(comments, author_name)
    return result.get("criticism", [])


def classify_questions_heuristic(comments: list[Comment], author_name: str = "") -> list[int]:
    """Regex fallback: question + author-address signals."""
    result = _classify_batch_heuristic(comments, author_name)
    return result.get("question", [])
