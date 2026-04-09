from __future__ import annotations

from datetime import UTC, datetime

from app.schemas.domain import (
    ClusterDraft,
    EpisodeContext,
    EpisodeTopic,
    ProcessedComment,
    TopicSummary,
)
from app.services.pipeline import _UNCERTAIN_TOPIC_LABEL, DailyRunService


def _comment(comment_id: str, text: str, weight: float = 1.0) -> ProcessedComment:
    return ProcessedComment(
        youtube_comment_id=comment_id,
        text_raw=text,
        text_normalized=text,
        text_hash=f"hash-{comment_id}",
        published_at=datetime(2026, 2, 23, 9, 0, tzinfo=UTC),
        weight=weight,
        like_count=0,
        reply_count=0,
    )


def _topic(
    cluster_key: str,
    label: str,
    source: str,
    weighted_share: float,
    size_count: int,
    centroid: list[float],
) -> TopicSummary:
    return TopicSummary(
        cluster_key=cluster_key,
        label=label,
        description=f"Topic details: {label}",
        author_actions=["Add one fact-check block and source links in the next episode."],
        sentiment="negative",
        emotion_tags=["concern"],
        intent_distribution={"complaint": 2, "question": 1},
        representative_quotes=[f"{label} sample quote"],
        question_comments=[],
        size_count=size_count,
        share_pct=round(size_count * 10.0, 2),
        weighted_share=weighted_share,
        is_emerging=False,
        source=source,
        coherence_score=0.61,
        centroid=centroid,
    )


def test_merge_similar_clusters_reduces_duplicates(db_session, test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "cluster_merge_enabled": True,
            "cluster_merge_similarity_threshold": 0.88,
            "cluster_merge_keyword_jaccard_min": 0.15,
        }
    )
    service = DailyRunService(settings, db_session)
    comments = [
        _comment("1", "санкции нефть рынок"),
        _comment("2", "цены на нефть санкции"),
        _comment("3", "санкции рынок курс"),
        _comment("4", "курс и санкции на нефть"),
        _comment("5", "футбол матч лига"),
        _comment("6", "лига футбол тренер"),
    ]
    vectors = [
        [1.0, 0.0],
        [0.98, 0.02],
        [0.97, 0.03],
        [0.99, 0.01],
        [0.0, 1.0],
        [0.02, 0.98],
    ]
    clusters = [
        ClusterDraft(
            cluster_key="cluster_a",
            member_indices=[0, 1],
            representative_indices=[0],
            centroid=[0.99, 0.01],
            size_count=2,
            share_pct=33.3,
            weighted_share=33.3,
        ),
        ClusterDraft(
            cluster_key="cluster_b",
            member_indices=[2, 3],
            representative_indices=[2],
            centroid=[0.98, 0.02],
            size_count=2,
            share_pct=33.3,
            weighted_share=33.3,
        ),
        ClusterDraft(
            cluster_key="cluster_c",
            member_indices=[4, 5],
            representative_indices=[4],
            centroid=[0.01, 0.99],
            size_count=2,
            share_pct=33.3,
            weighted_share=33.3,
        ),
    ]

    merged = service.cluster_enricher._merge_similar_clusters(clusters, comments, vectors)

    assert len(merged) == 2
    assert any(cluster.size_count == 4 for cluster in merged)


def test_coherence_skip_logic_uses_cluster_enricher(db_session, test_settings) -> None:
    service = DailyRunService(test_settings, db_session)
    comments = [
        _comment("1", "Сергей, вы не правы по фактам", 1.0),
        _comment("2", "Сергей, поддерживаю вашу позицию", 1.0),
    ]
    vectors = [[1.0, 0.0], [-1.0, 0.0]]
    cluster = ClusterDraft(
        cluster_key="cluster_x",
        member_indices=[0, 1],
        representative_indices=[0],
        centroid=[0.0, 0.0],
        size_count=2,
        share_pct=3.0,
        weighted_share=3.0,
    )

    coherence = service.cluster_enricher._estimate_cluster_coherence(cluster, comments, vectors)

    assert coherence == 0.0
    assert service.cluster_enricher._skip_low_coherence_cluster(cluster, coherence)


def test_postprocess_labeled_topics_merges_duplicates_and_keeps_episode_source(
    db_session, test_settings
) -> None:
    service = DailyRunService(test_settings, db_session)
    comments = [
        _comment("1", "elite children avoid war duty and people call it unfair", 1.3),
        _comment("2", "why elite children are not in the army", 1.2),
        _comment("3", "elite families avoid mobilization", 1.1),
        _comment("4", "elites do not share war responsibility", 1.0),
    ]
    vectors = [
        [1.0, 0.0],
        [0.98, 0.02],
        [0.99, 0.01],
        [0.97, 0.03],
    ]
    clusters = [
        ClusterDraft(
            cluster_key="cluster_a",
            member_indices=[0, 1],
            representative_indices=[0, 1],
            centroid=[0.99, 0.01],
            size_count=2,
            share_pct=50.0,
            weighted_share=52.0,
            is_emerging=False,
        ),
        ClusterDraft(
            cluster_key="cluster_b",
            member_indices=[2, 3],
            representative_indices=[2, 3],
            centroid=[0.98, 0.02],
            size_count=2,
            share_pct=50.0,
            weighted_share=48.0,
            is_emerging=False,
        ),
    ]
    topics = [
        _topic(
            cluster_key="cluster_a",
            label="Elite children and war responsibility",
            source="episode_topic",
            weighted_share=52.0,
            size_count=2,
            centroid=[0.99, 0.01],
        ),
        _topic(
            cluster_key="cluster_b",
            label="Elite children do not go to war",
            source="comment_topic",
            weighted_share=48.0,
            size_count=2,
            centroid=[0.98, 0.02],
        ),
    ]
    context = EpisodeContext(
        source="metadata",
        topics=[
            EpisodeTopic(
                title="Elite children and war responsibility",
                summary="Segment about elites and mobilization burden.",
            )
        ],
    )

    merged_clusters, merged_topics = service.cluster_enricher._postprocess_labeled_topics(
        clusters=clusters,
        topics=topics,
        comments=comments,
        vectors=vectors,
        episode_context=context,
    )

    assert len(merged_clusters) == 2
    assert len(merged_topics) == 2
    assert merged_topics[0].source == "episode_topic"
    assert any(topic.label == _UNCERTAIN_TOPIC_LABEL for topic in merged_topics)


def test_postprocess_relabels_empty_duplicate_topic_as_uncertain(db_session, test_settings) -> None:
    service = DailyRunService(test_settings, db_session)
    comments = [
        _comment("1", "helmet symbol and IOC double standards discussion", 1.2),
        _comment("2", "people are upset about IOC decisions", 1.1),
        _comment("3", "another off-topic thread", 1.0),
        _comment("4", "random side argument", 1.0),
    ]
    vectors = [
        [1.0, 0.0],
        [0.98, 0.02],
        [0.0, 1.0],
        [0.02, 0.98],
    ]
    clusters = [
        ClusterDraft(
            cluster_key="cluster_a",
            member_indices=[0, 1],
            representative_indices=[0, 1],
            centroid=[0.99, 0.01],
            size_count=2,
            share_pct=50.0,
            weighted_share=55.0,
            is_emerging=False,
        ),
        ClusterDraft(
            cluster_key="cluster_b",
            member_indices=[2, 3],
            representative_indices=[2, 3],
            centroid=[0.01, 0.99],
            size_count=2,
            share_pct=50.0,
            weighted_share=45.0,
            is_emerging=False,
        ),
    ]
    topic_main = _topic(
        cluster_key="cluster_a",
        label="IOC double standards and helmet symbol",
        source="episode_topic",
        weighted_share=55.0,
        size_count=2,
        centroid=[0.99, 0.01],
    )
    topic_empty = _topic(
        cluster_key="cluster_b",
        label="Helmet memory issue and IOC double standards",
        source="comment_topic",
        weighted_share=45.0,
        size_count=2,
        centroid=[0.01, 0.99],
    ).model_copy(
        update={
            "representative_quotes": [],
            "question_comments": [],
            "description": "This topic has no clear supporting comments.",
        }
    )
    context = EpisodeContext(
        source="metadata",
        topics=[
            EpisodeTopic(
                title="IOC double standards and helmet symbol",
                summary="Main segment of the episode.",
            )
        ],
    )

    _, merged_topics = service.cluster_enricher._postprocess_labeled_topics(
        clusters=clusters,
        topics=[topic_main, topic_empty],
        comments=comments,
        vectors=vectors,
        episode_context=context,
    )

    labels = [topic.label for topic in merged_topics]
    assert any("IOC double standards" in label for label in labels)
    assert any(label == _UNCERTAIN_TOPIC_LABEL for label in labels)


def test_resolve_topic_description_keeps_detailed_llm_text_on_low_support(
    db_session, test_settings
) -> None:
    service = DailyRunService(test_settings, db_session)
    llm_description = (
        "В обсуждении сталкиваются две линии. Часть аудитории винит руководство Украины "
        "за ошибки перед вторжением и плохую подготовку, другая часть считает главным "
        "фактором российскую агрессию, пропаганду и завышенные ожидания Кремля."
    )
    fallback_description = (
        "Кластер отражает спорные комментарии и критику позиции автора. "
        "Перед следующим выпуском важно заранее закрыть фактические разночтения."
    )
    ordered_comments = [
        "Зеленский проигнорировал разведданные и не подготовил оборону.",
        "Главный виновник — Кремль, ракеты и вторжение в 2022 году.",
        "Не надо снимать ответственность с России, это была явная агрессия.",
        "Почему автор выпуска не сказал о провале ожиданий по плану трех дней?",
    ]

    resolved = service.cluster_enricher._resolve_topic_description(
        description=llm_description,
        fallback_description=fallback_description,
        ordered_comments=ordered_comments,
        sentiment="negative",
    )

    assert resolved == " ".join(llm_description.split())
