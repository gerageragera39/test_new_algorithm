from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.schemas.domain import ProcessedComment
from app.services.clustering import ClusteringService


def _comment(idx: int, text: str, weight: float = 1.0) -> ProcessedComment:
    return ProcessedComment(
        youtube_comment_id=f"c{idx}",
        text_raw=text,
        text_normalized=text,
        text_hash=f"h{idx}",
        published_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        weight=weight,
    )


def test_clustering_returns_topics_and_representatives(test_settings) -> None:
    comments = [
        _comment(1, "sanctions and market", 1.3),
        _comment(2, "economic effects of sanctions", 1.4),
        _comment(3, "currency and inflation", 1.2),
        _comment(4, "football match result", 1.0),
        _comment(5, "league coach lineup", 1.1),
        _comment(6, "tournament match summary", 1.2),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.95, 0.05, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.05, 0.95, 0.0],
        [0.1, 0.9, 0.0],
    ]
    service = ClusteringService(test_settings)
    result = service.cluster(comments, embeddings)
    assert len(result.clusters) >= 2
    total_members = sum(len(cluster.member_indices) for cluster in result.clusters)
    assert total_members == len(comments)
    assert any(cluster.representative_indices for cluster in result.clusters)


def test_clustering_splits_large_noise_cluster(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "cluster_noise_split_enabled": True,
            "cluster_noise_split_min_size": 12,
            "cluster_noise_split_min_share_pct": 10.0,
            "cluster_noise_split_max_groups": 3,
            "cluster_min_size": 2,
            "cluster_min_samples": 1,
        }
    )
    service = ClusteringService(settings)
    matrix = np.array(
        [
            [0.98, 0.01, 0.0],
            [1.00, 0.02, 0.0],
            [0.96, -0.01, 0.0],
            [0.99, 0.03, 0.0],
            [0.97, -0.02, 0.0],
            [0.95, 0.01, 0.0],
            [0.01, 0.98, 0.0],
            [0.00, 1.00, 0.0],
            [-0.02, 0.97, 0.0],
            [0.02, 0.99, 0.0],
            [-0.01, 0.96, 0.0],
            [0.03, 0.95, 0.0],
        ],
        dtype=np.float32,
    )
    weights = np.ones(matrix.shape[0], dtype=np.float32)
    noise_members = list(range(matrix.shape[0]))
    split = service._split_large_noise_cluster(
        noise_members=noise_members,
        matrix=matrix,
        weights=weights,
        total_count=matrix.shape[0],
        total_weight=float(weights.sum()),
    )
    assert len(split) >= 2
    covered = sorted(idx for cluster in split for idx in cluster.member_indices)
    assert covered == noise_members


def test_clustering_noise_split_respects_quality_gate(test_settings, monkeypatch) -> None:
    settings = test_settings.model_copy(
        update={
            "cluster_noise_split_enabled": True,
            "cluster_noise_split_min_size": 12,
            "cluster_noise_split_min_share_pct": 10.0,
            "cluster_noise_split_max_groups": 3,
            "cluster_min_size": 2,
            "cluster_min_samples": 1,
        }
    )
    service = ClusteringService(settings)
    monkeypatch.setattr(service, "_passes_noise_split_quality_gate", lambda **_: False)

    matrix = np.array(
        [
            [0.98, 0.01, 0.0],
            [1.00, 0.02, 0.0],
            [0.96, -0.01, 0.0],
            [0.99, 0.03, 0.0],
            [0.97, -0.02, 0.0],
            [0.95, 0.01, 0.0],
            [0.01, 0.98, 0.0],
            [0.00, 1.00, 0.0],
            [-0.02, 0.97, 0.0],
            [0.02, 0.99, 0.0],
            [-0.01, 0.96, 0.0],
            [0.03, 0.95, 0.0],
        ],
        dtype=np.float32,
    )
    weights = np.ones(matrix.shape[0], dtype=np.float32)
    noise_members = list(range(matrix.shape[0]))

    split = service._split_large_noise_cluster(
        noise_members=noise_members,
        matrix=matrix,
        weights=weights,
        total_count=matrix.shape[0],
        total_weight=float(weights.sum()),
    )

    assert split == []


def test_clustering_reduces_embedding_dim_for_large_batches(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "cluster_reduction_enabled": True,
            "cluster_reduction_min_comments": 5,
            "cluster_reduction_target_dim": 3,
        }
    )
    service = ClusteringService(settings)
    matrix = np.eye(6, dtype=np.float32)

    reduced, summary = service._prepare_clustering_matrix(matrix)

    assert reduced.shape == (6, 3)
    assert summary["enabled"] is True
    assert summary["method"] == "pca"


def test_clustering_soft_assigns_noise_when_near_existing_centroid(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "cluster_soft_assignment_enabled": True,
            "cluster_soft_assignment_min_similarity": 0.4,
            "cluster_ambiguity_confidence_threshold": 0.7,
            "cluster_ambiguity_margin_threshold": 0.12,
        }
    )
    service = ClusteringService(settings)
    clustering_matrix = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.05, 0.95],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, -1, 1, 1], dtype=np.int32)
    probabilities = np.array([0.9, 0.88, 0.0, 0.9, 0.91], dtype=np.float32)

    assignments = service._build_assignments(
        labels=labels,
        probabilities=probabilities,
        clustering_matrix=clustering_matrix,
    )
    new_labels, new_assignments = service._soft_assign_noise_points(
        labels=labels,
        assignments=assignments,
        clustering_matrix=clustering_matrix,
    )

    assert new_labels[2] == 0
    assert new_assignments[2].used_soft_assignment is True


def test_large_cluster_split_uses_default_quality_gate_settings(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "cluster_large_split_enabled": True,
            "cluster_large_split_min_share_pct": 10.0,
            "cluster_large_split_max_subgroups": 3,
            "cluster_noise_split_target_group_size": 4,
        }
    )
    service = ClusteringService(settings)
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.96, 0.04],
            [0.0, 1.0],
            [0.02, 0.98],
            [0.03, 0.97],
            [0.04, 0.96],
            [0.70, 0.30],
            [0.68, 0.32],
            [0.32, 0.68],
            [0.30, 0.70],
        ],
        dtype=np.float32,
    )
    weights = np.ones(matrix.shape[0], dtype=np.float32)
    cluster = service._build_cluster(
        cluster_key="cluster_0",
        members=list(range(matrix.shape[0])),
        matrix=matrix,
        weights=weights,
        total_count=matrix.shape[0],
        total_weight=float(weights.sum()),
    )

    split = service._split_large_clusters(
        clusters=[cluster],
        matrix=matrix,
        weights=weights,
        total_count=matrix.shape[0],
        total_weight=float(weights.sum()),
    )

    assert split
