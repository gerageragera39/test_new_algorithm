from __future__ import annotations

from datetime import UTC, datetime

from app.core.exceptions import BudgetExceededError, ExternalServiceError
from app.schemas.domain import ProcessedComment, VideoMeta
from app.services.budget import BudgetGovernor
from app.services.labeling import LLMProvider
from app.services.pipeline import DailyRunService
from app.services.preprocessing import PreprocessResult


def _video_meta() -> VideoMeta:
    return VideoMeta(
        youtube_video_id="video_moderation_test",
        playlist_id="playlist",
        title="Policy strategy review",
        description="episode description",
        published_at=datetime(2026, 2, 25, 10, 0, tzinfo=UTC),
        duration_seconds=600,
        url="https://www.youtube.com/watch?v=video_moderation_test",
    )


def _borderline_comment() -> ProcessedComment:
    return ProcessedComment(
        youtube_comment_id="c1",
        text_raw="unclear short statement",
        text_normalized="unclear short statement",
        text_hash="hash-c1",
        published_at=datetime(2026, 2, 25, 10, 0, tzinfo=UTC),
        weight=1.0,
        like_count=0,
        reply_count=0,
        moderation_action="keep",
        moderation_reason="borderline_for_llm",
        moderation_source="fallback",
        moderation_score=0.44,
    )


def _preprocess_result(comment: ProcessedComment) -> PreprocessResult:
    return PreprocessResult(
        processed=[comment],
        all_comments=[comment],
        filtered_count=0,
        total_count=1,
        dropped_count=0,
        flagged_count=0,
        kept_count=1,
        borderline_comment_ids=[comment.youtube_comment_id],
    )


class _DummyProvider(LLMProvider):
    provider_name = "dummy"

    def is_available(self) -> bool:
        return True

    def analyze_cluster(self, ctx):  # noqa: ANN001
        raise RuntimeError("not used")


def test_llm_borderline_fallback_when_provider_unavailable(
    db_session, test_settings, monkeypatch
) -> None:
    service = DailyRunService(test_settings, db_session)
    comment = _borderline_comment()
    preprocessed = _preprocess_result(comment)
    budget = BudgetGovernor(test_settings, db_session)
    monkeypatch.setattr(
        service,
        "_build_llm_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ExternalServiceError("provider unavailable")
        ),
    )

    result, provider = service._apply_llm_borderline_moderation(
        preprocessed=preprocessed,
        video_meta=_video_meta(),
        budget=budget,
        llm_provider=None,
    )

    assert provider is None
    assert result.processed[0].moderation_action == "flag"
    assert result.processed[0].moderation_reason == "llm_unavailable_fallback"
    assert result.flagged_count == 1
    assert result.llm_moderation_stats["disabled_reason"] == "provider_unavailable"


def test_llm_borderline_timeout_falls_back_to_flag(db_session, test_settings, monkeypatch) -> None:
    service = DailyRunService(test_settings, db_session)
    comment = _borderline_comment()
    preprocessed = _preprocess_result(comment)
    budget = BudgetGovernor(test_settings, db_session)
    monkeypatch.setattr(
        service,
        "_request_llm_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ExternalServiceError("timeout")),
    )

    result, _provider = service._apply_llm_borderline_moderation(
        preprocessed=preprocessed,
        video_meta=_video_meta(),
        budget=budget,
        llm_provider=_DummyProvider(),
    )

    assert result.processed[0].moderation_action == "flag"
    assert result.processed[0].moderation_reason == "llm_error_fallback"
    assert int(result.llm_moderation_stats["fail_count"]) == 1
    assert int(result.llm_moderation_stats["flag_count"]) == 1


def test_llm_borderline_reserved_for_labeling_falls_back_to_reserved_flag(
    db_session,
    test_settings,
    monkeypatch,
) -> None:
    service = DailyRunService(test_settings, db_session)
    comment = _borderline_comment()
    preprocessed = _preprocess_result(comment)
    budget = BudgetGovernor(test_settings, db_session)
    monkeypatch.setattr(
        service,
        "_request_llm_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            BudgetExceededError("OpenAI moderation calls reserved for topic labeling.")
        ),
    )

    result, _provider = service._apply_llm_borderline_moderation(
        preprocessed=preprocessed,
        video_meta=_video_meta(),
        budget=budget,
        llm_provider=_DummyProvider(),
    )

    assert result.processed[0].moderation_action == "flag"
    assert result.processed[0].moderation_reason == "llm_reserved_for_labeling"
    assert str(result.llm_moderation_stats["disabled_reason"]) == "reserved_for_labeling"
