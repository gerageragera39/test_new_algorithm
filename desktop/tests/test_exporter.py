from __future__ import annotations

import json
from datetime import UTC, datetime

from app.schemas.domain import ActionItem, DailyBriefing, TopicPosition, TopicSummary
from app.services.exporter import ReportExporter


def test_markdown_has_positions_details_and_disagreement_section(test_settings) -> None:
    exporter = ReportExporter(test_settings)
    briefing = DailyBriefing(
        video_id="abc123xyz89",
        video_title="Ð¢ÐµÑÑ‚Ð¾Ð²Ñ‹Ð¹ Ð²Ñ‹Ð¿ÑƒÑÐº",
        published_at=datetime(2026, 2, 23, 10, 0, tzinfo=UTC),
        mode="free",
        executive_summary="ÐšÑ€Ð°Ñ‚ÐºÐ¾Ðµ Ñ€ÐµÐ·ÑŽÐ¼Ðµ.",
        top_topics=[
            TopicSummary(
                cluster_key="c1",
                label="ÐÐ¾Ð²Ð°Ñ Ñ‚ÐµÐ¼Ð°: Ñ„Ð°ÐºÑ‚Ñ‡ÐµÐºÐ¸Ð½Ð³",
                description="ÐžÐ¿Ð¸ÑÐ°Ð½Ð¸Ðµ Ñ‚ÐµÐ¼Ñ‹.",
                author_actions=["Ð”ÐµÐ¹ÑÑ‚Ð²Ð¸Ðµ 1"],
                sentiment="negative",
                emotion_tags=["Ñ‚Ñ€ÐµÐ²Ð¾Ð³Ð°"],
                intent_distribution={"question": 2},
                representative_quotes=[
                    "Ð¦Ð¸Ñ‚Ð°Ñ‚Ð° 1",
                    "Ð¦Ð¸Ñ‚Ð°Ñ‚Ð° 2",
                    "Ð¦Ð¸Ñ‚Ð°Ñ‚Ð° 3",
                    "Ð¦Ð¸Ñ‚Ð°Ñ‚Ð° 4",
                ],
                positions=[
                    TopicPosition(
                        key="pos_1",
                        title="ÐŸÐ¾Ð´Ð´ÐµÑ€Ð¶Ð¸Ð²Ð°ÑŽÑ‚ Ð°Ð²Ñ‚Ð¾Ñ€Ð°",
                        summary="Ð¡Ð¾Ð³Ð»Ð°ÑÐ½Ñ‹ Ñ Ð°Ñ€Ð³ÑƒÐ¼ÐµÐ½Ñ‚Ð°Ð¼Ð¸ Ð¸ Ð¿Ñ€Ð¸Ð²Ð¾Ð´ÑÑ‚ Ð»Ð¸Ñ‡Ð½Ñ‹Ðµ Ð¿Ñ€Ð¸Ð¼ÐµÑ€Ñ‹.",
                        prototype="Ð­Ñ‚Ð° Ð¿Ð¾Ð·Ð¸Ñ†Ð¸Ñ Ð¾Ñ‚Ñ€Ð°Ð¶Ð°ÐµÑ‚ Ð¿Ð¾Ð´Ð´ÐµÑ€Ð¶ÐºÑƒ Ð°Ñ€Ð³ÑƒÐ¼ÐµÐ½Ñ‚Ð¾Ð² Ð°Ð²Ñ‚Ð¾Ñ€Ð°.",
                        count=6,
                        pct=60.0,
                        weighted_count=6.0,
                        weighted_pct=60.0,
                        comments=["ÐšÐ¾Ð¼Ð¼ÐµÐ½Ñ‚Ð°Ñ€Ð¸Ð¹ 1", "ÐšÐ¾Ð¼Ð¼ÐµÐ½Ñ‚Ð°Ñ€Ð¸Ð¹ 2"],
                    ),
                    TopicPosition(
                        key="undetermined",
                        title="ÐÐµÐ¾Ð¿Ñ€ÐµÐ´ÐµÐ»ÐµÐ½Ð½Ñ‹Ðµ",
                        summary="ÐšÐ¾Ð¼Ð¼ÐµÐ½Ñ‚Ð°Ñ€Ð¸Ð¸ Ð±ÐµÐ· Ð½Ð°Ð´ÐµÐ¶Ð½Ð¾Ð³Ð¾ ÑÐ¾Ð¾Ñ‚Ð²ÐµÑ‚ÑÑ‚Ð²Ð¸Ñ Ð¿Ð¾Ð·Ð¸Ñ†Ð¸ÑÐ¼.",
                        prototype="ÐšÐ¾Ð¼Ð¼ÐµÐ½Ñ‚Ð°Ñ€Ð¸Ð¹ Ð½Ðµ Ð´Ð°ÐµÑ‚ Ð½Ð°Ð´ÐµÐ¶Ð½Ð¾Ð³Ð¾ ÑÐµÐ¼Ð°Ð½Ñ‚Ð¸Ñ‡ÐµÑÐºÐ¾Ð³Ð¾ ÑÐ¾Ð²Ð¿Ð°Ð´ÐµÐ½Ð¸Ñ.",
                        count=4,
                        pct=40.0,
                        weighted_count=4.0,
                        weighted_pct=40.0,
                        comments=["ÐšÐ¾Ð¼Ð¼ÐµÐ½Ñ‚Ð°Ñ€Ð¸Ð¹ X"],
                        is_undetermined=True,
                    ),
                ],
                size_count=10,
                share_pct=50.0,
                weighted_share=55.0,
                source="comment_topic",
            )
        ],
        actions_for_tomorrow=[
            "Ð¡ÐµÑ€Ð³ÐµÐ¹ Ð›ÑŽÐ±Ð°Ñ€ÑÐºÐ¸Ð¹: Ð´Ð¾Ð±Ð°Ð²ÑŒÑ‚Ðµ Ð² Ð¾Ð¿Ð¸ÑÐ°Ð½Ð¸Ðµ 2 Ð¸ÑÑ‚Ð¾Ñ‡Ð½Ð¸ÐºÐ° Ð¸ Ð·Ð°ÐºÑ€ÐµÐ¿Ð¸Ñ‚Ðµ ÐºÐ¾Ð¼Ð¼ÐµÐ½Ñ‚Ð°Ñ€Ð¸Ð¹ Ñ Ñ„Ð°ÐºÑ‚Ð°Ð¼Ð¸."
        ],
        action_items=[
            ActionItem(
                topic_cluster_key="c1",
                topic_label="ÐÐ¾Ð²Ð°Ñ Ñ‚ÐµÐ¼Ð°: Ñ„Ð°ÐºÑ‚Ñ‡ÐµÐºÐ¸Ð½Ð³",
                share_pct=50.0,
                priority=95,
                action="Ð’ ÑÐ»ÐµÐ´ÑƒÑŽÑ‰ÐµÐ¼ Ð²Ñ‹Ð¿ÑƒÑÐºÐµ Ð½ÑƒÐ¶Ð½Ð¾ Ð·Ð°ÐºÑ€Ñ‹Ñ‚ÑŒ Ð¿Ñ€ÐµÑ‚ÐµÐ½Ð·Ð¸Ð¸ Ð¿Ð¾ Ñ„Ð°ÐºÑ‚Ð°Ð¼.",
                key_criticism="Ð“Ð´Ðµ Ð¸ÑÑ‚Ð¾Ñ‡Ð½Ð¸ÐºÐ¸ Ð¸ Ð¿Ñ€Ð¾Ð²ÐµÑ€ÑÐµÐ¼Ñ‹Ðµ Ñ„Ð°ÐºÑ‚Ñ‹?",
                key_question="ÐŸÐ¾Ñ‡ÐµÐ¼Ñƒ Ð½Ðµ Ð¿Ð¾ÐºÐ°Ð·Ð°Ð½Ñ‹ Ð¿ÐµÑ€Ð²Ð¾Ð¸ÑÑ‚Ð¾Ñ‡Ð½Ð¸ÐºÐ¸?",
            )
        ],
        misunderstandings_and_controversies=[],
        audience_requests_and_questions=[],
        risks_and_toxicity=[],
        representative_quotes=["Ð¦Ð¸Ñ‚Ð°Ñ‚Ð° 1"],
        author_disagreement_comments=[
            "Ð¡ÐµÑ€Ð³ÐµÐ¹, Ð²Ñ‹ Ð½Ðµ Ð¿Ñ€Ð°Ð²Ñ‹ Ð¿Ð¾ ÑÑ‚Ð¾Ð¼Ñƒ Ñ‚ÐµÐ·Ð¸ÑÑƒ."
        ],
        metadata={
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_provider": "local_st",
            "cluster_assignment_confidence_avg": 0.81,
            "cluster_ambiguous_comment_share": 12.5,
            "cluster_reduction_summary": {"method": "pca", "input_dim": 1024, "output_dim": 32},
            "cluster_parameter_summary": {
                "algorithm": "hdbscan",
                "min_cluster_size": 12,
                "min_samples": 4,
                "cluster_selection_epsilon": 0.025,
            },
        },
    )

    content = exporter.to_markdown(briefing)

    assert "https://www.youtube.com/watch?v=abc123xyz89" in content
    assert "<details><summary><b>" in content
    assert "(6/10)</summary>" in content
    assert "(4/10)</summary>" in content
    assert "Главный вопрос аудитории" in content
    assert "Диагностика пайплайна" not in content
    assert "Репрезентативные цитаты" not in content


def test_persist_cluster_diagnostics_writes_json(tmp_path, test_settings) -> None:
    settings = test_settings.model_copy(update={"reports_dir": tmp_path})
    exporter = ReportExporter(settings)

    diagnostics = {
        "run_id": 42,
        "video_id": "abc123xyz89",
        "metrics": {"degraded": True, "fallback_title_rate": 44.2},
    }
    path = exporter.persist_cluster_diagnostics("abc123xyz89", diagnostics)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == 42
    assert payload["metrics"]["degraded"] is True
