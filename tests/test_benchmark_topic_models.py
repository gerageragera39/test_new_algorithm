from __future__ import annotations

from scripts.benchmark_topic_models import (
    ModelBenchmarkSummary,
    VideoBenchmarkResult,
    compute_composite_score,
    summarize_model_results,
)


def test_summarize_model_results_computes_composite_score() -> None:
    rows = [
        VideoBenchmarkResult(
            video_id="v1",
            video_title="Video 1",
            comment_count=400,
            cluster_count=6,
            emerging_cluster_count=1,
            avg_cluster_confidence=0.82,
            ambiguous_share_pct=14.0,
            avg_cluster_size=66.0,
            max_cluster_share_pct=31.0,
        ),
        VideoBenchmarkResult(
            video_id="v2",
            video_title="Video 2",
            comment_count=520,
            cluster_count=7,
            emerging_cluster_count=1,
            avg_cluster_confidence=0.78,
            ambiguous_share_pct=18.0,
            avg_cluster_size=72.0,
            max_cluster_share_pct=29.0,
        ),
    ]

    summary = summarize_model_results("Qwen/Qwen3-Embedding-0.6B", rows)

    assert summary.videos_evaluated == 2
    assert summary.composite_score == compute_composite_score(summary)
    assert summary.avg_cluster_confidence > 0.7


def test_compute_composite_score_rewards_lower_ambiguity() -> None:
    high = ModelBenchmarkSummary(
        model="high",
        videos_evaluated=1,
        avg_comment_count=400,
        avg_cluster_count=6,
        avg_emerging_cluster_count=1,
        avg_cluster_confidence=0.85,
        avg_ambiguous_share_pct=10.0,
        avg_cluster_size=60.0,
        avg_max_cluster_share_pct=28.0,
        composite_score=0.0,
        videos=[],
    )
    low = ModelBenchmarkSummary(
        model="low",
        videos_evaluated=1,
        avg_comment_count=400,
        avg_cluster_count=6,
        avg_emerging_cluster_count=1,
        avg_cluster_confidence=0.85,
        avg_ambiguous_share_pct=35.0,
        avg_cluster_size=60.0,
        avg_max_cluster_share_pct=28.0,
        composite_score=0.0,
        videos=[],
    )

    assert compute_composite_score(high) > compute_composite_score(low)
