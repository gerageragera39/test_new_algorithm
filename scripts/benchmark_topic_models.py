#!/usr/bin/env python3
"""Benchmark local embedding models for topic clustering quality.

This script replays the clustering stage on historically stored processed comments
and compares candidate embedding models without touching production reports.

Outputs:
- JSON summary under data/benchmarks/YYYY-MM-DD/
- Markdown summary under data/benchmarks/YYYY-MM-DD/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.models import Comment, Report, Video
from app.schemas.domain import ProcessedComment
from app.services.budget import BudgetGovernor
from app.services.clustering import ClusteringResult, ClusteringService
from app.services.embeddings import (
    EmbeddingCacheStore,
    EmbeddingService,
    LocalSentenceTransformerProvider,
)


@dataclass
class VideoBenchmarkResult:
    video_id: str
    video_title: str
    comment_count: int
    cluster_count: int
    emerging_cluster_count: int
    avg_cluster_confidence: float
    ambiguous_share_pct: float
    avg_cluster_size: float
    max_cluster_share_pct: float


@dataclass
class ModelBenchmarkSummary:
    model: str
    videos_evaluated: int
    avg_comment_count: float
    avg_cluster_count: float
    avg_emerging_cluster_count: float
    avg_cluster_confidence: float
    avg_ambiguous_share_pct: float
    avg_cluster_size: float
    avg_max_cluster_share_pct: float
    composite_score: float
    videos: list[VideoBenchmarkResult]


def _comment_from_row(row: Comment) -> ProcessedComment:
    return ProcessedComment(
        youtube_comment_id=row.youtube_comment_id,
        parent_comment_id=row.parent_comment_id,
        author_name=row.author_name,
        author_channel_id=row.author_channel_id,
        text_raw=row.text_raw,
        text_normalized=row.text_normalized,
        text_hash=row.text_hash,
        language=row.language,
        like_count=row.like_count,
        reply_count=row.reply_count,
        published_at=row.published_at,
        weight=row.weight,
        is_top_level=row.is_top_level,
        is_filtered=row.is_filtered,
        filter_reason=row.filter_reason,
        moderation_action=(row.moderation_action or "keep"),
        moderation_reason=row.moderation_reason,
        moderation_source=(row.moderation_source or "rule"),
        moderation_score=row.moderation_score,
    )


def compute_composite_score(summary: ModelBenchmarkSummary) -> float:
    return round(
        summary.avg_cluster_confidence * 45.0
        + max(0.0, 100.0 - summary.avg_ambiguous_share_pct) * 0.35
        + max(0.0, 100.0 - summary.avg_max_cluster_share_pct) * 0.10
        + min(summary.avg_cluster_count, 12.0) * 0.8
        - summary.avg_emerging_cluster_count * 1.5,
        3,
    )


def summarize_model_results(model: str, rows: list[VideoBenchmarkResult]) -> ModelBenchmarkSummary:
    if not rows:
        empty = ModelBenchmarkSummary(
            model=model,
            videos_evaluated=0,
            avg_comment_count=0.0,
            avg_cluster_count=0.0,
            avg_emerging_cluster_count=0.0,
            avg_cluster_confidence=0.0,
            avg_ambiguous_share_pct=0.0,
            avg_cluster_size=0.0,
            avg_max_cluster_share_pct=0.0,
            composite_score=0.0,
            videos=[],
        )
        return empty

    summary = ModelBenchmarkSummary(
        model=model,
        videos_evaluated=len(rows),
        avg_comment_count=round(mean(row.comment_count for row in rows), 2),
        avg_cluster_count=round(mean(row.cluster_count for row in rows), 2),
        avg_emerging_cluster_count=round(mean(row.emerging_cluster_count for row in rows), 2),
        avg_cluster_confidence=round(mean(row.avg_cluster_confidence for row in rows), 4),
        avg_ambiguous_share_pct=round(mean(row.ambiguous_share_pct for row in rows), 2),
        avg_cluster_size=round(mean(row.avg_cluster_size for row in rows), 2),
        avg_max_cluster_share_pct=round(mean(row.max_cluster_share_pct for row in rows), 2),
        composite_score=0.0,
        videos=rows,
    )
    summary.composite_score = compute_composite_score(summary)
    return summary


def benchmark_model(
    db: Session,
    settings: Settings,
    *,
    model_name: str,
    limit_videos: int,
    min_comments: int,
) -> ModelBenchmarkSummary:
    benchmark_settings = settings.model_copy(
        update={
            "embedding_mode": "local",
            "local_embedding_model": model_name,
        }
    )
    provider = LocalSentenceTransformerProvider(model_name, benchmark_settings)
    cache_store = EmbeddingCacheStore(benchmark_settings, db, provider.provider_name, provider.model_name)
    embedding_service = EmbeddingService(provider, cache_store)
    clustering = ClusteringService(benchmark_settings)
    budget = BudgetGovernor(benchmark_settings, db)
    _ = budget

    stmt = (
        select(Video)
        .join(Report, Report.video_id == Video.id)
        .order_by(Report.created_at.desc())
        .limit(limit_videos * 3)
    )
    candidate_videos = db.scalars(stmt).all()
    results: list[VideoBenchmarkResult] = []
    seen_video_ids: set[int] = set()

    for video in candidate_videos:
        if video.id in seen_video_ids:
            continue
        seen_video_ids.add(video.id)
        comments_stmt = (
            select(Comment)
            .where(Comment.video_id == video.id)
            .where(Comment.is_filtered.is_(False))
            .where((Comment.moderation_action.is_(None)) | (Comment.moderation_action != "drop"))
            .order_by(Comment.published_at.asc())
        )
        rows = db.scalars(comments_stmt).all()
        if len(rows) < min_comments:
            continue
        comments = [_comment_from_row(row) for row in rows]
        vectors = embedding_service.get_embeddings(
            [comment.text_normalized for comment in comments],
            [comment.text_hash for comment in comments],
            task="topic",
        )
        clustering_result: ClusteringResult = clustering.cluster(comments, vectors)
        cluster_count = len(clustering_result.clusters)
        if cluster_count == 0:
            continue
        emerging_count = sum(1 for cluster in clustering_result.clusters if cluster.is_emerging)
        avg_conf = mean(cluster.assignment_confidence for cluster in clustering_result.clusters)
        ambiguous_share = (
            sum(cluster.ambiguous_member_count for cluster in clustering_result.clusters)
            / max(1, len(comments))
            * 100.0
        )
        avg_cluster_size = mean(cluster.size_count for cluster in clustering_result.clusters)
        max_cluster_share = max(cluster.share_pct for cluster in clustering_result.clusters)
        results.append(
            VideoBenchmarkResult(
                video_id=video.youtube_video_id,
                video_title=video.title,
                comment_count=len(comments),
                cluster_count=cluster_count,
                emerging_cluster_count=emerging_count,
                avg_cluster_confidence=round(avg_conf, 4),
                ambiguous_share_pct=round(ambiguous_share, 2),
                avg_cluster_size=round(avg_cluster_size, 2),
                max_cluster_share_pct=round(max_cluster_share, 2),
            )
        )
        if len(results) >= limit_videos:
            break

    return summarize_model_results(model_name, results)


def write_benchmark_artifacts(output_dir: Path, summaries: list[ModelBenchmarkSummary]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "models": [asdict(summary) for summary in summaries],
    }
    json_path = output_dir / "topic_models_benchmark.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Topic Model Benchmark", ""]
    for summary in summaries:
        lines.append(f"## {summary.model}")
        lines.append(f"- Videos evaluated: {summary.videos_evaluated}")
        lines.append(f"- Composite score: {summary.composite_score:.3f}")
        lines.append(f"- Avg cluster confidence: {summary.avg_cluster_confidence:.3f}")
        lines.append(f"- Avg ambiguous share: {summary.avg_ambiguous_share_pct:.2f}%")
        lines.append(f"- Avg cluster count: {summary.avg_cluster_count:.2f}")
        lines.append(f"- Avg max cluster share: {summary.avg_max_cluster_share_pct:.2f}%")
        lines.append("")
    md_path = output_dir / "topic_models_benchmark.md"
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return json_path, md_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local embedding models for topic clustering")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen3-Embedding-0.6B",
            "BAAI/bge-m3",
            "intfloat/multilingual-e5-large",
        ],
    )
    parser.add_argument("--limit-videos", type=int, default=8)
    parser.add_argument("--min-comments", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    settings = Settings()
    engine = create_engine(settings.database_url)
    session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = session_local()
    try:
        summaries = [
            benchmark_model(
                db,
                settings,
                model_name=model_name,
                limit_videos=args.limit_videos,
                min_comments=args.min_comments,
            )
            for model_name in args.models
        ]
        summaries.sort(key=lambda item: item.composite_score, reverse=True)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        output_dir = args.output_dir or settings.data_dir / "benchmarks" / today
        json_path, md_path = write_benchmark_artifacts(output_dir, summaries)
        print(f"Saved benchmark JSON: {json_path}")
        print(f"Saved benchmark Markdown: {md_path}")
        if summaries:
            print(f"Best model: {summaries[0].model} (score={summaries[0].composite_score:.3f})")
        return 0
    finally:
        db.close()
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
