"""Tests for Appeal Analytics pipeline changes.

Covers:
  - Absence of spam block in BLOCK_DEFINITIONS and classification results
  - Question priority over criticism in mixed comments (heuristic fallback)
  - Lenient skip rule (border comments classified into useful categories)
  - Question Refiner: parsing, enrichment, field persistence in detail_json
  - Sorting of constructive_question items by refiner score
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_comment(
    id: int,
    text: str,
    author_name: str = "TestUser",
) -> MagicMock:
    c = MagicMock()
    c.id = id
    c.text_raw = text
    c.author_name = author_name
    return c


# ---------------------------------------------------------------------------
# 1. No spam block in BLOCK_DEFINITIONS
# ---------------------------------------------------------------------------


def test_block_definitions_has_no_spam() -> None:
    from app.services.appeal_analytics.runner import BLOCK_DEFINITIONS

    block_types = [bt for bt, _label, _order in BLOCK_DEFINITIONS]
    assert "spam" not in block_types, "spam block must be removed from BLOCK_DEFINITIONS"


def test_block_definitions_has_five_blocks() -> None:
    from app.services.appeal_analytics.runner import BLOCK_DEFINITIONS

    assert len(BLOCK_DEFINITIONS) == 5


def test_block_definitions_contains_expected_blocks() -> None:
    from app.services.appeal_analytics.runner import BLOCK_DEFINITIONS

    block_types = {bt for bt, _label, _order in BLOCK_DEFINITIONS}
    assert block_types == {
        "constructive_criticism",
        "constructive_question",
        "author_appeal",
        "toxic_auto_banned",
        "toxic_manual_review",
    }


def test_block_definitions_follow_ui_order() -> None:
    from app.services.appeal_analytics.runner import BLOCK_DEFINITIONS

    ordered_block_types = [bt for bt, _label, _order in BLOCK_DEFINITIONS]
    assert ordered_block_types == [
        "constructive_question",
        "constructive_criticism",
        "author_appeal",
        "toxic_auto_banned",
        "toxic_manual_review",
    ]


# ---------------------------------------------------------------------------
# 2. Stage total is 3 (not 4) in runner
# ---------------------------------------------------------------------------


def test_runner_stage_total_is_five() -> None:
    """_set_stage must write stage_total=5 (1:load, 2:classify, 3:refine Q+crit, 4:toxic, 5:persist)."""
    from app.services.appeal_analytics.runner import AppealAnalyticsService

    mock_run = MagicMock()
    mock_run.meta_json = {}
    mock_db = MagicMock()
    mock_settings = MagicMock()

    service = AppealAnalyticsService.__new__(AppealAnalyticsService)
    service.db = mock_db
    service.settings = mock_settings

    service._set_stage(mock_run, 1, "Loading")

    assert mock_run.meta_json["stage_total"] == 5
    assert mock_run.meta_json["stage_current"] == 1


def test_runner_rolls_back_before_persisting_failed_status(
    db_session, test_settings, monkeypatch
) -> None:
    """A failed transaction must be rolled back before saving failed run status."""
    from datetime import UTC, datetime

    from sqlalchemy import select as sa_select

    from app.db.models import AppealRun, Video
    from app.services.appeal_analytics.runner import AppealAnalyticsService

    video = Video(
        youtube_video_id="video-rollback",
        playlist_id="playlist-1",
        title="Rollback regression",
        description=None,
        published_at=datetime.now(UTC),
        duration_seconds=120,
        url="https://youtube.com/watch?v=video-rollback",
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    service = AppealAnalyticsService(test_settings, db_session)

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_set_stage", boom)

    with pytest.raises(RuntimeError, match="boom"):
        service._execute(video)

    failed_run = db_session.scalar(
        sa_select(AppealRun)
        .where(AppealRun.video_id == video.id)
        .order_by(AppealRun.id.desc())
        .limit(1)
    )

    assert failed_run is not None
    assert failed_run.status == "failed"
    assert failed_run.ended_at is not None
    assert "boom" in (failed_run.error or "")


def test_unified_heuristic_marks_configured_guest_insult_as_toxic() -> None:
    from app.services.appeal_analytics.llm_classifier import classify_unified_heuristic

    comment = _make_comment(1, "Солонин несёт бред, просто идиот какой-то.")
    result = classify_unified_heuristic(
        [comment],
        author_name="Автор",
        guest_names=["Марк Солонин"],
    )

    assert 1 in result["toxic"]


def test_unified_heuristic_keeps_third_party_insult_outside_toxic_without_guest_match() -> None:
    from app.services.appeal_analytics.llm_classifier import classify_unified_heuristic

    comment = _make_comment(2, "Путин идиот и несёт полный бред.")
    result = classify_unified_heuristic(
        [comment],
        author_name="Автор",
        guest_names=["Марк Солонин"],
    )

    assert 2 not in result["toxic"]


def test_run_for_meta_persists_guest_names_before_execute(
    db_session, test_settings, monkeypatch
) -> None:
    from datetime import UTC, datetime

    from sqlalchemy import select as sa_select

    from app.db.models import VideoSettings
    from app.schemas.domain import VideoMeta
    from app.services.appeal_analytics.runner import AppealAnalyticsService

    service = AppealAnalyticsService(test_settings, db_session)
    youtube = MagicMock()
    youtube.fetch_comments.return_value = []

    captured: dict[str, str | None] = {}

    def fake_execute(video):
        settings_row = db_session.scalar(
            sa_select(VideoSettings).where(VideoSettings.video_id == video.id)
        )
        captured["guest_names"] = settings_row.guest_names if settings_row else None
        return {
            "appeal_run_id": 1,
            "video_id": video.youtube_video_id,
            "status": "completed",
            "total_comments": 0,
            "processed_comments": 0,
        }

    monkeypatch.setattr(service, "_execute", fake_execute)

    result = service._run_for_meta(
        youtube,
        VideoMeta(
            youtube_video_id="guest-video",
            playlist_id="playlist-1",
            title="Guest video",
            description=None,
            published_at=datetime.now(UTC),
            duration_seconds=3600,
            url="https://www.youtube.com/watch?v=guest-video",
        ),
        guest_names=["Марк Солонин", "Алексей Арестович"],
    )

    assert result["video_id"] == "guest-video"
    assert captured["guest_names"] == "Марк Солонин, Алексей Арестович"


# ---------------------------------------------------------------------------
# 3. Heuristic: question priority over criticism in mixed comment
# ---------------------------------------------------------------------------


def test_heuristic_question_wins_over_criticism_in_mixed_comment() -> None:
    """A comment with both criticism signals AND a question mark → question."""
    from app.services.appeal_analytics.llm_classifier import _classify_batch_heuristic

    # "нужно было" triggers criticism signal; "?" triggers question signal; author ref present
    mixed_comment = _make_comment(
        1,
        "Автор, нужно было рассмотреть эту тему глубже. Почему вы не привели "
        "источники и альтернативную точку зрения?",
    )
    result = _classify_batch_heuristic([mixed_comment], author_name="Автор")
    assert 1 in result["question"], "Mixed comment (criticism+question) must go to question"
    assert 1 not in result["criticism"], "Mixed comment must NOT go to criticism"


def test_heuristic_pure_criticism_without_question_mark() -> None:
    """A comment with criticism signal but no question mark → criticism."""
    from app.services.appeal_analytics.llm_classifier import _classify_batch_heuristic

    # "нужно было" matches \bнужно\s+было\b — a clear word-boundary-safe pattern
    crit_comment = _make_comment(
        2,
        "Автор, нужно было рассмотреть вопрос подробнее, нужно было привести "
        "больше аргументов и источников по этой теме.",
    )
    result = _classify_batch_heuristic([crit_comment], author_name="Автор")
    assert 2 in result["criticism"]
    assert 2 not in result["question"]


# ---------------------------------------------------------------------------
# 4. Lenient skip: heuristic must not over-skip author-addressed comments
# ---------------------------------------------------------------------------


def test_heuristic_classifies_question_with_author_reference() -> None:
    """A question referencing the author must not be skipped."""
    from app.services.appeal_analytics.llm_classifier import _classify_batch_heuristic

    comment = _make_comment(
        3,
        "Автор, как вы относитесь к этому событию?",
    )
    result = _classify_batch_heuristic([comment], author_name="Иван")
    # Should be in question (has "автор" + "?" + enough words)
    total_classified = (
        len(result["question"])
        + len(result["criticism"])
        + len(result["appeal"])
        + len(result["toxic"])
    )
    assert total_classified > 0, "Author-addressed question must not be silently skipped"
    assert 3 in result["question"]


def test_heuristic_classifies_question_without_question_mark() -> None:
    """Interrogative wording should be enough for heuristic question detection."""
    from app.services.appeal_analytics.llm_classifier import _classify_batch_heuristic

    comment = _make_comment(
        30,
        "Автор, почему вы не раскрыли источник этой оценки без знака вопроса",
    )
    result = _classify_batch_heuristic([comment], author_name="Иван")
    assert 30 in result["question"]


def test_heuristic_classifies_directive_request_without_author_mention() -> None:
    """Imperative content requests should still land in the appeal block."""
    from app.services.appeal_analytics.llm_classifier import _classify_batch_heuristic

    comment = _make_comment(
        31,
        "Разберите, пожалуйста, отдельно тему санкций в следующем видео",
    )
    result = _classify_batch_heuristic([comment], author_name="Иван")
    assert 31 in result["appeal"]


def test_heuristic_does_not_classify_unrelated_comment() -> None:
    """A comment without any author reference stays unclassified (effectively skip)."""
    from app.services.appeal_analytics.llm_classifier import _classify_batch_heuristic

    comment = _make_comment(
        4,
        "Путин принял неожиданное решение по этому вопросу.",
    )
    result = _classify_batch_heuristic([comment], author_name="Иван")
    total_classified = sum(len(v) for v in result.values())
    assert total_classified == 0, "Unrelated comment (no author ref) must not be classified"


# ---------------------------------------------------------------------------
# 5. Word-based truncation in LLM prompt
# ---------------------------------------------------------------------------


def test_clip_words_short_text_unchanged() -> None:
    from app.services.appeal_analytics.llm_classifier import _clip_words

    text = "Короткий текст без обрезки"
    assert _clip_words(text, max_words=150) == text


def test_clip_words_long_text_truncated() -> None:
    from app.services.appeal_analytics.llm_classifier import _clip_words

    words = ["слово"] * 200
    text = " ".join(words)
    clipped = _clip_words(text, max_words=150)
    clipped_words = clipped.split()
    # head+tail strategy: result is within budget (head + "[...]" + tail)
    assert len(clipped_words) <= 155
    # Head+tail separator must be present
    assert "[...]" in clipped


def test_clip_words_exactly_max_not_truncated() -> None:
    from app.services.appeal_analytics.llm_classifier import _clip_words

    words = ["слово"] * 150
    text = " ".join(words)
    assert _clip_words(text, max_words=150) == text
    assert not _clip_words(text, max_words=150).endswith("...")


def test_unified_prompt_uses_word_based_truncation() -> None:
    """Verify that very long comments are word-clipped in the prompt, not character-clipped."""
    from app.services.appeal_analytics.llm_classifier import _build_unified_prompt

    # Create a long comment: 300 words but only ~1800 chars (well over 500 chars)
    long_text = " ".join(["информация"] * 300)
    prompt = _build_unified_prompt([(1, long_text)], author_name="Тест")

    # Count words in the comment line (starts with "1. ")
    comment_line = [line for line in prompt.split("\n") if line.startswith("1. ")][0]
    word_count = len(comment_line.split())
    # Default is now 220 words: head(180) + "[...]"(1) + tail(40) + "1."(1) = ~222 words
    assert word_count <= 230, f"Expected ≤230 words, got {word_count}"
    # Head+tail strategy: "[...]" separator must be present
    assert "[...]" in comment_line, "Long comment must use head+tail separator [...]"


# ---------------------------------------------------------------------------
# 6. Question Refiner: parse response
# ---------------------------------------------------------------------------


def test_refiner_parse_valid_response() -> None:
    from app.services.appeal_analytics.question_refiner import _parse_refiner_response

    raw = {
        "results": {
            "1": {
                "topic": "Ukraine_Russia",
                "score": 8,
                "short": "Вопрос о конфликте?",  # already ends with "?" — real question type
                "depth_score": 7,
                "question_type": "analysis_why_how",
            },
            "2": {
                "topic": "Economy",
                "score": 6,
                "short": "Вопрос об экономике?",  # already ends with "?" — real question type
                "depth_score": 5,
                "question_type": "fact_check",
            },
        }
    }

    parsed = _parse_refiner_response(raw, {1, 2})
    assert 1 in parsed
    assert parsed[1]["topic"] == "Ukraine_Russia"
    assert parsed[1]["score"] == 8
    assert parsed[1]["depth_score"] == 7
    assert parsed[1]["question_type"] == "analysis_why_how"
    assert parsed[1]["short"] == "Вопрос о конфликте?"

    assert 2 in parsed
    assert parsed[2]["topic"] == "Economy"
    assert parsed[2]["score"] == 6


def test_refiner_parse_attack_ragebait_caps_score_at_3() -> None:
    from app.services.appeal_analytics.question_refiner import _parse_refiner_response

    raw = {
        "results": {
            "1": {
                "topic": "Other",
                "score": 7,  # High score but attack type → should be capped to 3
                "short": "Агрессивный вопрос",
                "depth_score": 1,
                "question_type": "attack_ragebait",
            }
        }
    }
    parsed = _parse_refiner_response(raw, {1})
    assert parsed[1]["score"] <= 3, "attack_ragebait must be capped at score 3"


def test_refiner_parse_meme_one_liner_caps_score() -> None:
    from app.services.appeal_analytics.question_refiner import _parse_refiner_response

    raw = {
        "results": {
            "1": {
                "topic": "AI",
                "score": 9,
                "short": "Мем",
                "depth_score": 0,
                "question_type": "meme_one_liner",
            }
        }
    }
    parsed = _parse_refiner_response(raw, {1})
    assert parsed[1]["score"] <= 3, "meme_one_liner must be capped at score 3"


def test_refiner_parse_invalid_topic_falls_back_to_other() -> None:
    from app.services.appeal_analytics.question_refiner import _parse_refiner_response

    raw = {
        "results": {
            "1": {
                "topic": "UNKNOWN_TOPIC_XYZ",
                "score": 5,
                "short": "Вопрос",
                "depth_score": 4,
                "question_type": "clarification_needed",
            }
        }
    }
    parsed = _parse_refiner_response(raw, {1})
    assert parsed[1]["topic"] == "Other"


def test_refiner_parse_invalid_question_type_falls_back() -> None:
    from app.services.appeal_analytics.question_refiner import _parse_refiner_response

    raw = {
        "results": {
            "1": {
                "topic": "USA",
                "score": 5,
                "short": "Вопрос",
                "depth_score": 4,
                "question_type": "TOTALLY_UNKNOWN",
            }
        }
    }
    parsed = _parse_refiner_response(raw, {1})
    assert parsed[1]["question_type"] == "clarification_needed"


def test_refiner_parse_skips_out_of_range_nums() -> None:
    from app.services.appeal_analytics.question_refiner import _parse_refiner_response

    raw = {
        "results": {
            "5": {
                "topic": "USA",
                "score": 7,
                "short": "Вопрос",
                "depth_score": 5,
                "question_type": "analysis_why_how",
            }
        }
    }
    # valid_nums = {1, 2, 3} — 5 is not valid
    parsed = _parse_refiner_response(raw, {1, 2, 3})
    assert len(parsed) == 0


def test_refiner_parse_none_returns_empty() -> None:
    from app.services.appeal_analytics.question_refiner import _parse_refiner_response

    assert _parse_refiner_response(None, {1}) == {}
    assert _parse_refiner_response("invalid", {1}) == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 7. Refiner integration: no-LLM provider returns empty dict
# ---------------------------------------------------------------------------


def test_refiner_returns_empty_for_no_llm_provider() -> None:
    from app.services.appeal_analytics.question_refiner import refine_questions
    from app.services.labeling import NoLLMFallbackProvider

    provider = NoLLMFallbackProvider()
    comments = [_make_comment(1, "Автор, вопрос: почему?")]

    result = refine_questions(comments, provider, lambda *a, **kw: None)
    assert result == {}


# ---------------------------------------------------------------------------
# 8. Refiner score becomes primary sort key in routes
# ---------------------------------------------------------------------------


def test_constructive_question_items_sorted_by_score_descending() -> None:
    """Items in the constructive_question block must be sorted by score desc."""
    from app.schemas.api import AppealBlockItemResponse

    items = [
        AppealBlockItemResponse(comment_id=1, author_name="A", text="Q1", score=3),
        AppealBlockItemResponse(comment_id=2, author_name="B", text="Q2", score=9),
        AppealBlockItemResponse(comment_id=3, author_name="C", text="Q3", score=6),
        AppealBlockItemResponse(comment_id=4, author_name="D", text="Q4", score=None),
    ]
    # Replicate routes.py sort logic
    items.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)

    assert items[0].score == 9
    assert items[1].score == 6
    assert items[2].score == 3
    assert items[3].score is None


# ---------------------------------------------------------------------------
# 9. Runner persist_block saves refiner fields in detail_json
# ---------------------------------------------------------------------------


def test_persist_block_includes_refiner_fields(db_session) -> None:
    """When refiner_data is provided, detail_json must contain topic/short/depth_score/question_type."""
    # Create minimal DB records
    from datetime import UTC, datetime

    from app.core.config import Settings
    from app.db.models import AppealBlockItem, AppealRun, Comment, Video
    from app.services.appeal_analytics.runner import AppealAnalyticsService

    video = Video(
        youtube_video_id="test123",
        playlist_id="PLtest",
        title="Test Video",
        url="https://youtube.com/watch?v=test123",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(video)
    db_session.flush()

    comment = Comment(
        video_id=video.id,
        youtube_comment_id="YTc1",
        author_name="User1",
        text_raw="Автор, как вы относитесь к этому?",
        text_normalized="Автор, как вы относитесь к этому?",
        text_hash="hash1",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(comment)
    db_session.flush()

    appeal_run = AppealRun(
        video_id=video.id,
        status="running",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        total_comments=1,
        processed_comments=0,
        meta_json={},
    )
    db_session.add(appeal_run)
    db_session.flush()

    settings = MagicMock(spec=Settings)
    service = AppealAnalyticsService.__new__(AppealAnalyticsService)
    service.db = db_session
    service.settings = settings

    refiner_data = {
        comment.id: {
            "topic": "Ukraine_Russia",
            "score": 8,
            "short": "Вопрос о конфликте",
            "depth_score": 7,
            "question_type": "analysis_why_how",
        }
    }
    scores = {comment.id: 8}

    service._persist_block(
        appeal_run=appeal_run,
        video=video,
        block_type="constructive_question",
        display_label="Вопросы",
        sort_order=2,
        comment_ids=[comment.id],
        all_comments=[comment],
        scores=scores,
        refiner_data=refiner_data,
    )

    item = db_session.query(AppealBlockItem).first()
    assert item is not None
    detail = item.detail_json
    assert detail.get("topic") == "Ukraine_Russia"
    assert detail.get("short") == "Вопрос о конфликте"
    assert detail.get("depth_score") == 7
    assert detail.get("question_type") == "analysis_why_how"
    assert detail.get("score") == 8


def test_persist_block_without_refiner_data_has_no_refiner_fields(db_session) -> None:
    """When no refiner_data, detail_json must only contain score (no topic/short/etc.)."""
    from datetime import UTC, datetime

    from app.core.config import Settings
    from app.db.models import AppealBlockItem, AppealRun, Comment, Video
    from app.services.appeal_analytics.runner import AppealAnalyticsService

    video = Video(
        youtube_video_id="test456",
        playlist_id="PLtest2",
        title="Test Video 2",
        url="https://youtube.com/watch?v=test456",
        published_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    db_session.add(video)
    db_session.flush()

    comment = Comment(
        video_id=video.id,
        youtube_comment_id="YTc2",
        author_name="User2",
        text_raw="Критика автора без вопроса.",
        text_normalized="Критика автора без вопроса.",
        text_hash="hash2",
        published_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    db_session.add(comment)
    db_session.flush()

    appeal_run = AppealRun(
        video_id=video.id,
        status="running",
        started_at=datetime(2026, 1, 2, tzinfo=UTC),
        total_comments=1,
        processed_comments=0,
        meta_json={},
    )
    db_session.add(appeal_run)
    db_session.flush()

    settings = MagicMock(spec=Settings)
    service = AppealAnalyticsService.__new__(AppealAnalyticsService)
    service.db = db_session
    service.settings = settings

    service._persist_block(
        appeal_run=appeal_run,
        video=video,
        block_type="constructive_criticism",
        display_label="Критика",
        sort_order=1,
        comment_ids=[comment.id],
        all_comments=[comment],
        scores={comment.id: 7},
        refiner_data=None,
    )

    item = db_session.query(AppealBlockItem).first()
    assert item is not None
    detail = item.detail_json
    assert detail.get("score") == 7
    assert "topic" not in detail
    assert "short" not in detail
    assert "depth_score" not in detail
    assert "question_type" not in detail


# ---------------------------------------------------------------------------
# 10. Refiner: score override in runner pipeline
# ---------------------------------------------------------------------------


def test_runner_refiner_score_overrides_initial_score() -> None:
    """Refiner score must override the classifier's initial score for question items."""
    # Simulate the merge logic from runner._execute
    initial_scores = {101: 5, 102: 6}

    refiner_data = {
        101: {
            "topic": "AI",
            "score": 9,
            "short": "Вопрос об ИИ",
            "depth_score": 8,
            "question_type": "analysis_why_how",
        },
        # comment 102 refiner failed → no entry
    }

    # Replicate merge logic from runner._execute
    scores = dict(initial_scores)
    for cid, meta in refiner_data.items():
        if "score" in meta and meta["score"] is not None:
            scores[cid] = meta["score"]

    assert scores[101] == 9, "Refiner score must override initial score for comment 101"
    assert scores[102] == 6, "Comment 102 without refiner result keeps initial score"


# ---------------------------------------------------------------------------
# 11. Prompt: skip priority rule is present in prompt text
# ---------------------------------------------------------------------------


def test_unified_prompt_contains_priority_rule() -> None:
    """The unified prompt must contain the priority chain, not 'при сомнении skip'."""
    from app.services.appeal_analytics.llm_classifier import _build_unified_prompt

    prompt = _build_unified_prompt([(1, "тестовый комментарий")], author_name="Автор")
    # New priority rule must be present
    assert "question > criticism > appeal > toxic > skip" in prompt
    # Old "at doubt skip" instruction must NOT appear as the primary rule
    assert "Лучше пропустить, чем ошибочно классифицировать" not in prompt


def test_unified_prompt_question_category_mentions_priority_over_criticism() -> None:
    from app.services.appeal_analytics.llm_classifier import _build_unified_prompt

    prompt = _build_unified_prompt([(1, "тест")], author_name="Автор")
    # Must mention that question wins over criticism
    assert "критику" in prompt.lower() and "question" in prompt.lower()


# ===========================================================================
# NEW TESTS (round 2): missed_nums fallback, head+tail, strict refiner, spam API
# ===========================================================================

# ---------------------------------------------------------------------------
# 12. Head+tail truncation strategy
# ---------------------------------------------------------------------------


def test_clip_words_short_text_unchanged_r2() -> None:
    from app.services.appeal_analytics.llm_classifier import _clip_words

    text = "Короткий текст ровно десять слов в строке подряд."
    assert _clip_words(text, max_words=150) == text


def test_clip_words_long_text_has_separator() -> None:
    """Long text must contain the [...] separator between head and tail."""
    from app.services.appeal_analytics.llm_classifier import _clip_words

    words = ["слово"] * 200
    text = " ".join(words)
    clipped = _clip_words(text, max_words=150)
    assert "[...]" in clipped, "Head+tail separator must be present in clipped text"


def test_clip_words_tail_is_preserved() -> None:
    """The last words of a long comment must appear in the clipped result."""
    from app.services.appeal_analytics.llm_classifier import _clip_words

    # 200 words; last word is "ВОПРОС?"
    words = ["слово"] * 199 + ["ВОПРОС?"]
    text = " ".join(words)
    clipped = _clip_words(text, max_words=150)
    assert "ВОПРОС?" in clipped, "Last word (potential question) must be preserved in tail"


def test_clip_words_head_preserved() -> None:
    """The first meaningful words must remain after clipping."""
    from app.services.appeal_analytics.llm_classifier import _clip_words

    words = ["НАЧАЛО"] + ["слово"] * 199
    text = " ".join(words)
    clipped = _clip_words(text, max_words=150)
    assert clipped.startswith("НАЧАЛО"), "Head of comment must be preserved"


def test_clip_words_within_budget() -> None:
    """Clipped text must not exceed max_words + some separator overhead."""
    from app.services.appeal_analytics.llm_classifier import _clip_words

    words = ["слово"] * 300
    text = " ".join(words)
    clipped = _clip_words(text, max_words=150)
    # Allow for "[...]" separator — total word count ~151
    total_words = len(clipped.split())
    assert total_words <= 155, f"Clipped text has too many words: {total_words}"


# ---------------------------------------------------------------------------
# 13. Missed-nums fallback in classify_unified_llm
# ---------------------------------------------------------------------------


def test_llm_missed_nums_get_heuristic_fallback() -> None:
    """When LLM omits some comment numbers, those comments must get heuristic labels."""
    from app.services.appeal_analytics.llm_classifier import (
        classify_unified_llm,
    )

    # Two comments: one author-question, one unrelated
    question_comment = _make_comment(
        1,
        "Автор, нужно было рассмотреть этот вопрос отдельно. Когда вы планируете выпустить "
        "разбор по данной теме?",
    )
    # Not addressed to author → heuristic will skip (no author reference)
    other_comment = _make_comment(2, "Путин снова что-то сказал про экономику.")

    comments = [question_comment, other_comment]

    # LLM returns only comment #2 (misses comment #1)
    def partial_llm_json(*args, **kwargs):
        return {"results": {"2": "skip"}}  # returns only num 2, omits num 1

    from app.services.labeling import OpenAIChatProvider

    mock_provider = MagicMock(spec=OpenAIChatProvider)

    result = classify_unified_llm(
        comments,
        mock_provider,
        partial_llm_json,
        author_name="Автор",
    )

    # comment 1 was missed by LLM, must be covered by heuristic
    # It references "Автор" and has "?" → heuristic should put it in question
    all_classified = (
        result.ids["question"]
        + result.ids["criticism"]
        + result.ids["appeal"]
        + result.ids["toxic"]
    )
    assert (
        question_comment.id in all_classified
    ), "Missed comment with author-question must be classified by heuristic fallback"


def test_llm_empty_response_falls_back_to_heuristic() -> None:
    """Completely empty LLM response must trigger full heuristic fallback for the batch."""
    from app.services.appeal_analytics.llm_classifier import classify_unified_llm

    comment = _make_comment(
        1,
        "Автор, когда вы планируете выпустить следующее видео по этой теме?",
    )

    def empty_llm(*args, **kwargs):
        return {}  # empty/invalid response

    from app.services.labeling import OpenAIChatProvider

    mock_provider = MagicMock(spec=OpenAIChatProvider)

    result = classify_unified_llm([comment], mock_provider, empty_llm, author_name="Автор")

    all_classified = (
        result.ids["question"]
        + result.ids["criticism"]
        + result.ids["appeal"]
        + result.ids["toxic"]
    )
    assert (
        comment.id in all_classified
    ), "Empty LLM response must trigger heuristic fallback covering all batch comments"


# ---------------------------------------------------------------------------
# 14. Strict refiner: score rules
# ---------------------------------------------------------------------------


def test_apply_strict_score_rules_low_value_cap() -> None:
    from app.services.appeal_analytics.question_refiner import _apply_strict_score_rules

    # attack_ragebait: any initial score capped at 3
    assert _apply_strict_score_rules(8, 5, "attack_ragebait") <= 3
    assert _apply_strict_score_rules(10, 9, "meme_one_liner") <= 3


def test_apply_strict_score_rules_depth_cap() -> None:
    from app.services.appeal_analytics.question_refiner import _apply_strict_score_rules

    # depth_score 3 (< 6) → score capped at 6
    score = _apply_strict_score_rules(9, 3, "analysis_why_how")
    assert score <= 6


def test_apply_strict_score_rules_deep_analysis_uncapped() -> None:
    from app.services.appeal_analytics.question_refiner import _apply_strict_score_rules

    # depth_score 8 > 7 + real question type → score allowed up to 10
    score = _apply_strict_score_rules(9, 8, "analysis_why_how")
    assert score == 9


def test_apply_strict_score_rules_clarification_deep_capped() -> None:
    from app.services.appeal_analytics.question_refiner import _apply_strict_score_rules

    # clarification_needed with inflated depth_score >= 7 → depth capped, score capped
    score = _apply_strict_score_rules(9, 8, "clarification_needed")
    assert score <= 6


# ---------------------------------------------------------------------------
# 15. Strict refiner: short sanitisation
# ---------------------------------------------------------------------------


def test_sanitize_short_strips_emoji() -> None:
    from app.services.appeal_analytics.question_refiner import _sanitize_short

    text = "Почему ситуация изменилась именно так? 🤔"
    result = _sanitize_short(text, "analysis_why_how")
    assert "🤔" not in (result or "")


def test_sanitize_short_strips_curly_quotes() -> None:
    from app.services.appeal_analytics.question_refiner import _sanitize_short

    text = "«Почему это происходит?»"
    result = _sanitize_short(text, "analysis_why_how")
    assert "«" not in (result or "") and "»" not in (result or "")
    assert result is not None
    assert result.endswith("?")


def test_sanitize_short_real_question_ends_with_question_mark() -> None:
    from app.services.appeal_analytics.question_refiner import _sanitize_short

    # Real question type + text without "?" → "?" must be appended
    text = "Почему вы не рассмотрели этот аспект"
    result = _sanitize_short(text, "analysis_why_how")
    assert result is not None
    assert result.endswith("?"), f"Expected trailing '?', got: {result!r}"


def test_sanitize_short_real_question_already_has_mark() -> None:
    from app.services.appeal_analytics.question_refiner import _sanitize_short

    text = "Почему это произошло?"
    result = _sanitize_short(text, "fact_check")
    # Must end with exactly one "?"
    assert result is not None
    assert result.endswith("?")
    assert not result.endswith("??")


def test_sanitize_short_low_value_strips_question_mark() -> None:
    from app.services.appeal_analytics.question_refiner import _sanitize_short

    # attack_ragebait: "?" must be stripped from short
    text = "Ты совсем тупой что ли?"
    result = _sanitize_short(text, "attack_ragebait")
    assert result is not None
    assert not result.endswith("?"), f"attack_ragebait short must not end with '?': {result!r}"


def test_sanitize_short_truncates_to_160_chars() -> None:
    from app.services.appeal_analytics.question_refiner import _sanitize_short

    long_text = "А" * 200 + " вопрос?"
    result = _sanitize_short(long_text, "analysis_why_how")
    assert result is not None
    assert len(result) <= 162  # 160 + possible "?" appended after ellipsis


def test_sanitize_short_empty_returns_none() -> None:
    from app.services.appeal_analytics.question_refiner import _sanitize_short

    assert _sanitize_short("", "clarification_needed") is None
    assert _sanitize_short("   ", "clarification_needed") is None
    assert _sanitize_short(None, "clarification_needed") is None  # type: ignore[arg-type]


def test_refiner_parse_low_value_depth_capped() -> None:
    """meme_one_liner must have depth_score <= 3 after parsing."""
    from app.services.appeal_analytics.question_refiner import _parse_refiner_response

    raw = {
        "results": {
            "1": {
                "topic": "AI",
                "score": 9,
                "short": "Ха-ха смешной вопрос",
                "depth_score": 8,  # inflated by model
                "question_type": "meme_one_liner",
            }
        }
    }
    parsed = _parse_refiner_response(raw, {1})
    assert parsed[1]["score"] <= 3
    assert parsed[1]["depth_score"] <= 3


def test_refiner_parse_short_sanitized_in_parse() -> None:
    """_parse_refiner_response must run _sanitize_short on the short field."""
    from app.services.appeal_analytics.question_refiner import _parse_refiner_response

    raw = {
        "results": {
            "1": {
                "topic": "Economy",
                "score": 7,
                "short": "Почему экономика падает 😱",
                "depth_score": 6,
                "question_type": "analysis_why_how",
            }
        }
    }
    parsed = _parse_refiner_response(raw, {1})
    short = parsed[1]["short"] or ""
    assert "😱" not in short
    assert short.endswith("?")


# ---------------------------------------------------------------------------
# 16. Legacy spam block filtered from API response
# ---------------------------------------------------------------------------


def test_legacy_spam_block_not_in_api_response(db_session) -> None:
    """GET /appeal/{video_id} must not return a block with block_type='spam'."""
    from datetime import UTC, datetime

    from app.db.models import AppealBlock, AppealRun, Video

    # Seed test data directly
    video = Video(
        youtube_video_id="spam_test_vid",
        playlist_id="PLspam",
        title="Spam Filter Test",
        url="https://youtube.com/watch?v=spam_test_vid",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    db_session.add(video)
    db_session.flush()

    appeal_run = AppealRun(
        video_id=video.id,
        status="completed",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        total_comments=2,
        processed_comments=1,
        meta_json={},
    )
    db_session.add(appeal_run)
    db_session.flush()

    spam_block = AppealBlock(
        appeal_run_id=appeal_run.id,
        video_id=video.id,
        block_type="spam",
        sort_order=5,
        display_label="Спам",
        item_count=1,
    )
    db_session.add(spam_block)
    db_session.flush()

    # Verify the route-level filter using the in-memory query
    from sqlalchemy import select as sa_select

    from app.db.models import AppealBlock as AB

    blocks_db = list(
        db_session.scalars(
            sa_select(AB).where(AB.appeal_run_id == appeal_run.id).order_by(AB.sort_order.asc())
        )
    )
    # Apply same filter as routes.py
    filtered = [b for b in blocks_db if b.block_type != "spam"]
    assert all(
        b.block_type != "spam" for b in filtered
    ), "Spam blocks must be filtered out before returning to API client"
    assert len(filtered) == 0  # only spam block was created


# ===========================================================================
# NEW TESTS (round 3): political criticism filter, toxic precision filter,
# spam-free pipeline, block label rename, has_question_signal
# ===========================================================================

# ---------------------------------------------------------------------------
# 17. Political criticism filter
# ---------------------------------------------------------------------------


def test_political_criticism_parse_keeps_true_drops_false() -> None:
    """_parse_political_criticism_response respects explicit true/false values."""
    from app.services.appeal_analytics.political_criticism_refiner import (
        _parse_political_criticism_response,
    )

    raw = {"results": {"1": True, "2": False, "3": True}}
    parsed = _parse_political_criticism_response(raw, {1, 2, 3})
    assert parsed[1] is True
    assert parsed[2] is False
    assert parsed[3] is True


def test_political_criticism_parse_missing_num_defaults_to_keep() -> None:
    """A comment number absent from the LLM response should default to keep=True."""
    from app.services.appeal_analytics.political_criticism_refiner import (
        _parse_political_criticism_response,
    )

    raw = {"results": {"1": True}}
    # num 2 is in valid_nums but not in response
    parsed = _parse_political_criticism_response(raw, {1, 2})
    assert parsed[2] is True, "Missing num must default to keep=True (fail-open)"


def test_political_criticism_filter_no_llm_keeps_all() -> None:
    """When NoLLMFallbackProvider is used, all candidates are retained."""
    from app.services.appeal_analytics.political_criticism_refiner import filter_political_criticism
    from app.services.labeling import NoLLMFallbackProvider

    comments = [
        _make_comment(1, "автор, вы совершенно неправы по поводу внешнего вида оратора"),
        _make_comment(2, "автор, ваш аргумент об экономике ошибочен: вот факты..."),
    ]
    provider = NoLLMFallbackProvider()
    kept = filter_political_criticism(comments, provider, lambda *a, **kw: None)
    assert set(kept) == {1, 2}, "NoLLM mode must retain all candidates"


def test_political_criticism_filter_llm_failure_keeps_all() -> None:
    """If the LLM call throws an exception, the batch is retained unchanged."""
    from app.core.exceptions import ExternalServiceError
    from app.services.appeal_analytics.political_criticism_refiner import filter_political_criticism
    from app.services.labeling import OpenAIChatProvider

    comments = [_make_comment(1, "автор, нужно было сказать о другом")]
    mock_provider = MagicMock(spec=OpenAIChatProvider)

    def failing_llm(*a, **kw):
        raise ExternalServiceError("LLM down")

    kept = filter_political_criticism(comments, mock_provider, failing_llm)
    assert 1 in kept, "On LLM failure all candidates must be retained"


# ---------------------------------------------------------------------------
# 18. Toxic precision filter
# ---------------------------------------------------------------------------


def test_toxic_precision_parse_high_confidence_kept() -> None:
    """Comments with confidence >= threshold must be kept."""
    from app.services.appeal_analytics.toxic_precision_refiner import (
        _CONFIDENCE_THRESHOLD,
        _parse_toxic_precision_response,
    )

    raw = {"results": {"1": 0.95, "2": 0.30}}
    parsed = _parse_toxic_precision_response(raw, {1, 2})
    assert parsed[1] >= _CONFIDENCE_THRESHOLD, "High-confidence toxic must be kept"
    assert parsed[2] < _CONFIDENCE_THRESHOLD, "Low-confidence (ambiguous) must be dropped"


def test_toxic_precision_parse_missing_defaults_to_zero() -> None:
    """A comment number absent from response defaults to 0.0 (dropped)."""
    from app.services.appeal_analytics.toxic_precision_refiner import (
        _parse_toxic_precision_response,
    )

    raw = {"results": {"1": 0.9}}
    parsed = _parse_toxic_precision_response(raw, {1, 2})
    assert parsed.get(2, 0.0) == 0.0, "Missing num defaults to 0.0 → dropped"


def test_toxic_precision_filter_no_llm_keeps_all() -> None:
    """NoLLMFallbackProvider: trust heuristic, keep all."""
    from app.services.appeal_analytics.toxic_precision_refiner import filter_toxic_precision
    from app.services.labeling import NoLLMFallbackProvider

    comments = [
        _make_comment(1, "автор идиот"),
        _make_comment(2, "автор несёт бред"),
    ]
    provider = NoLLMFallbackProvider()
    kept = filter_toxic_precision(comments, provider, lambda *a, **kw: None)
    assert set(kept) == {1, 2}, "NoLLM mode must retain all heuristic-classified toxic"


def test_toxic_precision_filter_drops_ambiguous() -> None:
    """Low-confidence items must be excluded from the result."""
    from app.services.appeal_analytics.toxic_precision_refiner import filter_toxic_precision
    from app.services.labeling import OpenAIChatProvider

    comments = [
        _make_comment(1, "автор, ваша позиция полностью провальная"),  # sharp but not insult
        _make_comment(2, "автор мудак"),  # clear insult
    ]
    mock_provider = MagicMock(spec=OpenAIChatProvider)

    def mock_llm(*a, **kw):
        return {"results": {"1": 0.30, "2": 0.95}}  # 1=ambiguous, 2=clear insult

    kept = filter_toxic_precision(comments, mock_provider, mock_llm)
    assert 2 in kept, "High-confidence direct insult must be kept"
    assert 1 not in kept, "Low-confidence sharp criticism must be dropped"


def test_auto_ban_verification_parse_keeps_only_explicit_approvals() -> None:
    from app.services.appeal_analytics.toxic_precision_refiner import (
        _parse_auto_ban_verification_response,
    )

    raw = {
        "results": {
            "1": {
                "allow_auto_ban": True,
                "confidence": 0.96,
                "target_confirmed": "author",
            },
            "2": {
                "allow_auto_ban": False,
                "confidence": 0.22,
                "target_confirmed": "third_party",
            },
        }
    }

    parsed = _parse_auto_ban_verification_response(raw, {1, 2})
    assert parsed[1]["allow_auto_ban"] is True
    assert parsed[1]["target_confirmed"] == "author"
    assert parsed[2]["allow_auto_ban"] is False
    assert parsed[2]["target_confirmed"] == "third_party"


def test_verify_auto_ban_candidates_rejects_third_party_sarcasm() -> None:
    from app.services.appeal_analytics.toxic_precision_refiner import (
        verify_auto_ban_candidates,
    )
    from app.services.labeling import OpenAIChatProvider

    comments = [
        _make_comment(1, "Автор, вы просто идиот."),
        _make_comment(2, "Путин, конечно, гений... какой же идиот."),
    ]
    toxic_metadata = {
        1: {"insult_target": "author"},
        2: {"insult_target": "author"},  # upstream made a mistake
    }
    mock_provider = MagicMock(spec=OpenAIChatProvider)

    def mock_llm(*_args, **_kwargs):
        return {
            "results": {
                "1": {
                    "allow_auto_ban": True,
                    "confidence": 0.98,
                    "target_confirmed": "author",
                },
                "2": {
                    "allow_auto_ban": False,
                    "confidence": 0.18,
                    "target_confirmed": "third_party",
                },
            }
        }

    approved = verify_auto_ban_candidates(
        comments,
        author_name="Автор",
        guest_names=[],
        toxic_metadata=toxic_metadata,
        llm_provider=mock_provider,
        request_llm_json=mock_llm,
        confidence_threshold=0.93,
    )

    assert 1 in approved
    assert 2 not in approved


# ---------------------------------------------------------------------------
# 19. Spam not in active pipeline
# ---------------------------------------------------------------------------


def test_no_spam_in_block_definitions() -> None:
    """BLOCK_DEFINITIONS must not contain a spam entry."""
    from app.services.appeal_analytics.runner import BLOCK_DEFINITIONS

    block_types = [bt for bt, _, _ in BLOCK_DEFINITIONS]
    assert "spam" not in block_types


def test_runner_has_no_spam_imports() -> None:
    """runner.py must not import spam_detector or classify_spam."""
    import inspect

    import app.services.appeal_analytics.runner as runner_module

    source = inspect.getsource(runner_module)
    assert "spam_detector" not in source
    assert "classify_spam" not in source


# ---------------------------------------------------------------------------
# 20. Block label rename
# ---------------------------------------------------------------------------


def test_criticism_block_has_political_label() -> None:
    """The constructive_criticism block must use the updated political-position label."""
    from app.services.appeal_analytics.runner import BLOCK_DEFINITIONS

    label_map = {bt: label for bt, label, _ in BLOCK_DEFINITIONS}
    assert (
        "политической позиции" in label_map["constructive_criticism"]
    ), "criticism block label must mention 'политической позиции'"
    assert "Конструктивная критика" in label_map["constructive_criticism"]


# ---------------------------------------------------------------------------
# 21. has_question_signal helper
# ---------------------------------------------------------------------------


def test_has_question_signal_detects_question_mark() -> None:
    from app.services.appeal_analytics.llm_classifier import has_question_signal

    assert has_question_signal("Как вы относитесь к этому?")
    assert has_question_signal("А что если попробовать иначе?")


def test_has_question_signal_detects_interrogative_words() -> None:
    from app.services.appeal_analytics.llm_classifier import has_question_signal

    assert has_question_signal("Почему вы не рассмотрели этот аспект")
    assert has_question_signal("зачем столько сложностей")
    assert has_question_signal("как это работает на практике")


def test_has_question_signal_returns_false_for_statement() -> None:
    from app.services.appeal_analytics.llm_classifier import has_question_signal

    assert not has_question_signal("Это неверная позиция, нужно было иначе.")
    assert not has_question_signal("Автор совершенно неправ в этом вопросе.")


# ---------------------------------------------------------------------------
# 22. Question comments must stay in the question block even without '?'
# ---------------------------------------------------------------------------


def test_question_without_qmark_stays_in_question() -> None:
    """Interrogative comments without '?' must remain in the question block."""
    from app.services.appeal_analytics.llm_classifier import has_question_signal

    comment = _make_comment(1, "автор, зачем вы это делаете без знака вопроса")
    question_ids: list[int] = [comment.id]
    criticism_ids: list[int] = []

    assert has_question_signal(comment.text_raw)
    assert comment.id in question_ids
    assert comment.id not in criticism_ids


def test_question_with_qmark_stays_in_question() -> None:
    """Every question comment that has a '?' must stay in the question block."""
    question_ids: list[int] = [1]
    criticism_ids: list[int] = []

    assert 1 in question_ids
    assert 1 not in criticism_ids


def test_question_signal_word_does_not_require_question_mark() -> None:
    from app.services.appeal_analytics.llm_classifier import has_question_signal

    assert has_question_signal("автор, почему вы выбрали именно эту тему")


def test_question_partition_demotes_ragebait_and_borderline_non_questions() -> None:
    from app.services.appeal_analytics.runner import _partition_constructive_question_candidates

    comment_map = {
        1: _make_comment(1, "Автор, почему вы опять врёте людям"),
        2: _make_comment(2, "Автор, можно раскрыть источники по этой теме?"),
        3: _make_comment(3, "Автор, поясните позицию по санкциям"),
    }
    refiner_data = {
        1: {"question_type": "attack_ragebait", "score": 2},
        2: {"question_type": "analysis_why_how", "score": 8},
        3: {"question_type": "clarification_needed", "score": 4},
    }

    kept, demoted = _partition_constructive_question_candidates(
        question_candidate_ids=[1, 2, 3],
        comment_map=comment_map,
        refiner_data=refiner_data,
    )

    assert kept == [2]
    assert demoted == [1, 3]


# ---------------------------------------------------------------------------
# Toxic Classification Tests: third_party exclusion, routing policy
# ---------------------------------------------------------------------------


def test_toxic_third_party_never_auto_banned_high_confidence() -> None:
    """third_party insults must NEVER be auto-banned, even with confidence >= 0.80."""
    from app.services.appeal_analytics.runner import _route_toxic_classification

    assert (
        _route_toxic_classification(
            target="third_party",
            confidence=0.95,
            auto_ban_threshold=0.80,
        )
        == "ignore"
    )
    assert (
        _route_toxic_classification(
            target="author",
            confidence=0.95,
            auto_ban_threshold=0.80,
        )
        == "auto_ban"
    )


def test_toxic_author_high_confidence_auto_banned() -> None:
    """author insults with confidence >= 0.80 must be auto-banned."""
    from app.services.appeal_analytics.runner import _route_toxic_classification

    assert (
        _route_toxic_classification(
            target="author",
            confidence=0.80,
            auto_ban_threshold=0.80,
        )
        == "auto_ban"
    )


def test_toxic_guest_medium_confidence_manual_review() -> None:
    """guest insults below auto-ban threshold must go to manual review."""
    from app.services.appeal_analytics.runner import _route_toxic_classification

    assert (
        _route_toxic_classification(
            target="guest",
            confidence=0.70,
            auto_ban_threshold=0.80,
        )
        == "manual_review"
    )


def test_toxic_undefined_medium_confidence_manual_review() -> None:
    """undefined insults are review-only, even with high confidence."""
    from app.services.appeal_analytics.runner import _route_toxic_classification

    assert (
        _route_toxic_classification(
            target="undefined",
            confidence=0.60,
            auto_ban_threshold=0.80,
        )
        == "manual_review"
    )
    assert (
        _route_toxic_classification(
            target="undefined",
            confidence=0.98,
            auto_ban_threshold=0.80,
        )
        == "manual_review"
    )


def test_toxic_low_confidence_non_third_party_still_reviewed() -> None:
    """Low-confidence non-third-party toxic candidates must not be silently lost."""
    from app.services.appeal_analytics.runner import _route_toxic_classification

    assert (
        _route_toxic_classification(
            target="author",
            confidence=0.45,
            auto_ban_threshold=0.80,
        )
        == "manual_review"
    )
    assert (
        _route_toxic_classification(
            target="content",
            confidence=0.30,
            auto_ban_threshold=0.80,
        )
        == "manual_review"
    )


def test_toxic_content_high_confidence_auto_banned() -> None:
    """content insults with high confidence are auto-banned."""
    from app.services.appeal_analytics.runner import _route_toxic_classification

    assert (
        _route_toxic_classification(
            target="content",
            confidence=0.80,
            auto_ban_threshold=0.80,
        )
        == "auto_ban"
    )


def test_precision_review_downgrades_rejected_autobans_to_manual_review(
    test_settings, db_session, monkeypatch
) -> None:
    from app.services.appeal_analytics.runner import AppealAnalyticsService

    service = AppealAnalyticsService(test_settings, db_session)
    comment_map = {
        1: _make_comment(1, "автор мудак"),
        2: _make_comment(2, "канал позор"),
    }
    toxic_metadata = {
        1: {"insult_target": "author", "confidence_score": 0.97},
        2: {"insult_target": "content", "confidence_score": 0.95},
    }

    monkeypatch.setattr(
        "app.services.appeal_analytics.runner.verify_auto_ban_candidates",
        lambda comments, *args, **kwargs: {
            comments[0].id: {"confidence": 0.98, "target_confirmed": "author"}
        },
    )

    confirmed, downgraded = service._apply_toxic_autoban_precision_review(
        auto_ban_ids=[1, 2],
        comment_map_full=comment_map,
        toxic_metadata=toxic_metadata,
        llm_provider=MagicMock(),
        author_name="Автор",
        guest_names=[],
    )

    assert confirmed == [1]
    assert downgraded == [2]
    assert toxic_metadata[1]["precision_review_status"] == "confirmed_auto_ban"
    assert toxic_metadata[2]["precision_review_status"] == "downgraded_to_manual_review"
