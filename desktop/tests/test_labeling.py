from __future__ import annotations

from datetime import UTC, datetime

from app.schemas.domain import ClusterDraft, ProcessedComment
from app.services.labeling import ClusterContext, _normalize_llm_result


def _comment(idx: int, text: str) -> ProcessedComment:
    return ProcessedComment(
        youtube_comment_id=f"c{idx}",
        text_raw=text,
        text_normalized=text,
        text_hash=f"h{idx}",
        published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        weight=1.0,
    )


def test_normalize_llm_result_rejects_generic_actions() -> None:
    ctx = ClusterContext(
        cluster=ClusterDraft(
            cluster_key="cluster_1",
            member_indices=[0, 1, 2],
            representative_indices=[0, 1],
            centroid=[0.1, 0.2, 0.3],
            size_count=3,
            share_pct=10.0,
            weighted_share=12.0,
        ),
        representative_comments=[_comment(1, "нужно больше фактов"), _comment(2, "где источники?")],
        all_comments=[
            _comment(1, "нужно больше фактов"),
            _comment(2, "где источники?"),
            _comment(3, "спасибо"),
        ],
        episode_topics=["Фактчекинг: проверка заявлений в выпуске"],
        matched_episode_topic="Фактчекинг",
    )
    raw = {
        "label": "Comments about video",
        "description": "short",
        "author_actions": ["like", "comment", "share"],
        "sentiment": "negative",
        "emotion_tags": ["anger"],
        "intent_distribution": {"question": 2},
        "representative_quotes": ["где источники?"],
    }
    result = _normalize_llm_result(raw, ctx)
    assert result.label == "Comments about video"
    assert all(
        action.lower() not in {"like", "comment", "share"} for action in result.author_actions
    )
    assert len(result.author_actions) >= 1


def test_normalize_llm_result_does_not_force_irrelevant_episode_topic() -> None:
    ctx = ClusterContext(
        cluster=ClusterDraft(
            cluster_key="cluster_2",
            member_indices=[0, 1, 2],
            representative_indices=[0, 1],
            centroid=[0.1, 0.2, 0.3],
            size_count=3,
            share_pct=10.0,
            weighted_share=12.0,
        ),
        representative_comments=[
            _comment(1, "please verify fake claims before publishing"),
            _comment(2, "the channel needs better fact checking"),
        ],
        all_comments=[
            _comment(1, "please verify fake claims before publishing"),
            _comment(2, "the channel needs better fact checking"),
            _comment(3, "more sources and less speculation"),
        ],
        episode_topics=["Energy sanctions and oil prices"],
        matched_episode_topic="Energy sanctions and oil prices",
    )
    raw = {
        "label": "Comments about video",
        "description": "Cluster discussion about verification standards and source quality in this episode.",
        "author_actions": [
            "Add sources for each disputed claim",
            "Pin a correction if factual errors are found",
        ],
        "sentiment": "neutral",
        "emotion_tags": ["concern"],
        "intent_distribution": {"question": 1, "request": 1},
        "representative_quotes": ["please verify fake claims before publishing"],
    }
    result = _normalize_llm_result(raw, ctx)
    assert "energy sanctions" not in result.label.lower()


def test_normalize_llm_result_flattens_list_like_description() -> None:
    ctx = ClusterContext(
        cluster=ClusterDraft(
            cluster_key="cluster_3",
            member_indices=[0, 1, 2],
            representative_indices=[0, 1],
            centroid=[0.1, 0.2, 0.3],
            size_count=3,
            share_pct=10.0,
            weighted_share=12.0,
        ),
        representative_comments=[
            _comment(1, "users ask for cleaner structure in the episode"),
            _comment(2, "more clarity in transitions would help"),
        ],
        all_comments=[
            _comment(1, "users ask for cleaner structure in the episode"),
            _comment(2, "more clarity in transitions would help"),
            _comment(3, "chapter timestamps are also requested"),
        ],
        episode_topics=["Episode structure and pacing"],
        matched_episode_topic="Episode structure and pacing",
    )
    raw = {
        "label": "Episode structure feedback",
        "description": (
            "['Audience asks for clearer block transitions and fewer abrupt switches between subtopics.', "
            "'Comments request chapter timestamps and explicit recap after each major segment.']"
        ),
        "author_actions": [
            "Add a one-line recap after each major block",
            "Publish chapter timestamps in description",
        ],
        "sentiment": "neutral",
        "emotion_tags": ["focus"],
        "intent_distribution": {"request": 2},
        "representative_quotes": ["chapter timestamps are also requested"],
    }
    result = _normalize_llm_result(raw, ctx)
    assert not result.description.startswith("[")


def test_normalize_llm_result_filters_non_realistic_actions() -> None:
    ctx = ClusterContext(
        cluster=ClusterDraft(
            cluster_key="cluster_4",
            member_indices=[0, 1, 2],
            representative_indices=[0, 1],
            centroid=[0.2, 0.1, 0.4],
            size_count=3,
            share_pct=11.0,
            weighted_share=13.0,
        ),
        representative_comments=[
            _comment(1, "Сергей, вы не правы по этой теме"),
            _comment(2, "нужны источники и конкретика"),
        ],
        all_comments=[
            _comment(1, "Сергей, вы не правы по этой теме"),
            _comment(2, "нужны источники и конкретика"),
            _comment(3, "сделайте FAQ в следующем выпуске"),
        ],
        episode_topics=["Политический конфликт и заявления сторон"],
        matched_episode_topic="Политический конфликт и заявления сторон",
    )
    raw = {
        "label": "Критика выпуска",
        "description": "Комментарии требуют более четких источников и объяснений спорных тезисов в выпуске.",
        "author_actions": [
            "Ввести механизм обратной связи от фронтовых командиров для корректировки приказов в реальном времени",
            "Провести общественные слушания по реформе военного управления",
            "Добавьте в следующий выпуск короткий FAQ из трех вопросов и ответов",
        ],
        "sentiment": "negative",
        "emotion_tags": ["anger"],
        "intent_distribution": {"request": 2, "complaint": 1},
        "representative_quotes": ["Сергей, вы не правы", "нужны источники"],
    }
    result = _normalize_llm_result(raw, ctx)
    merged_actions = " ".join(result.author_actions).lower()
    assert "фронт" not in merged_actions
    assert "обществен" not in merged_actions
    assert result.author_actions


def test_normalize_llm_result_keeps_more_than_three_quotes() -> None:
    ctx = ClusterContext(
        cluster=ClusterDraft(
            cluster_key="cluster_5",
            member_indices=[0, 1, 2, 3, 4, 5],
            representative_indices=[0, 1, 2],
            centroid=[0.3, 0.2, 0.5],
            size_count=6,
            share_pct=18.0,
            weighted_share=19.0,
        ),
        representative_comments=[
            _comment(1, "цитата 1"),
            _comment(2, "цитата 2"),
            _comment(3, "цитата 3"),
        ],
        all_comments=[
            _comment(1, "цитата 1"),
            _comment(2, "цитата 2"),
            _comment(3, "цитата 3"),
            _comment(4, "цитата 4"),
            _comment(5, "цитата 5"),
            _comment(6, "цитата 6"),
        ],
        episode_topics=["Тема"],
        matched_episode_topic="Тема",
    )
    raw = {
        "label": "Тема кластера",
        "description": "Подборка комментариев отражает устойчивый спор и повторяемые запросы на уточнение источников.",
        "author_actions": [
            "Добавьте в описание ссылки на первоисточники и закрепите уточнение в комментарии"
        ],
        "sentiment": "neutral",
        "emotion_tags": ["focus"],
        "intent_distribution": {"request": 2},
        "representative_quotes": [
            "цитата 1",
            "цитата 2",
            "цитата 3",
            "цитата 4",
            "цитата 5",
            "цитата 6",
        ],
    }
    result = _normalize_llm_result(raw, ctx)
    assert len(result.representative_quotes) >= 4


def test_normalize_llm_result_keeps_long_label_without_word_limit() -> None:
    ctx = ClusterContext(
        cluster=ClusterDraft(
            cluster_key="cluster_6",
            member_indices=[0, 1, 2],
            representative_indices=[0, 1],
            centroid=[0.2, 0.3, 0.4],
            size_count=3,
            share_pct=12.0,
            weighted_share=13.0,
        ),
        representative_comments=[
            _comment(1, "нужны ссылки на источники"),
            _comment(2, "автор смешал два разных кейса"),
        ],
        all_comments=[
            _comment(1, "нужны ссылки на источники"),
            _comment(2, "автор смешал два разных кейса"),
            _comment(3, "нужно разделить аргументы по блокам"),
        ],
        episode_topics=["Проверка аргументов и источников"],
        matched_episode_topic="Проверка аргументов и источников",
    )
    long_label = (
        "Очень длинное название категории без искусственного ограничения по количеству слов"
    )
    raw = {
        "label": long_label,
        "description": "Коротко.",
        "author_actions": [
            "Добавьте список источников в описание и закрепите комментарий с ссылками"
        ],
        "sentiment": "neutral",
        "emotion_tags": ["concern"],
        "intent_distribution": {"request": 1},
        "representative_quotes": ["нужны ссылки на источники"],
    }

    result = _normalize_llm_result(raw, ctx)

    assert result.label == long_label
