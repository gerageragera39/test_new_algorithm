"""Adaptive comment clustering with dimensionality reduction and soft assignment.

The production clustering path keeps HDBSCAN as the primary algorithm, but
improves stability for 300-2000 comments by:

* optionally reducing dense embeddings via PCA before clustering,
* adapting HDBSCAN parameters to dataset size,
* using confidence-aware soft assignment for borderline/noise comments,
* keeping original-vector centroids for downstream topic similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from app.core.config import Settings
from app.schemas.domain import ClusterDraft, ProcessedComment


@dataclass
class CommentClusterAssignment:
    """Assignment diagnostics for a single comment."""

    primary_label: int | None
    primary_confidence: float
    secondary_label: int | None = None
    secondary_confidence: float = 0.0
    is_ambiguous: bool = False
    used_soft_assignment: bool = False
    remained_noise: bool = False


@dataclass
class ClusteringResult:
    """Result of the clustering stage."""

    clusters: list[ClusterDraft]
    label_by_index: dict[int, str]
    assignment_by_index: dict[int, CommentClusterAssignment] = field(default_factory=dict)
    reduction_summary: dict[str, Any] = field(default_factory=dict)
    parameter_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FitResult:
    labels: np.ndarray
    probabilities: np.ndarray
    parameter_summary: dict[str, Any]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class ClusteringService:
    """Clusters preprocessed comments using adaptive HDBSCAN."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)

    def cluster(
        self, comments: list[ProcessedComment], embeddings: list[list[float]]
    ) -> ClusteringResult:
        if not comments:
            return ClusteringResult(clusters=[], label_by_index={})
        if len(comments) != len(embeddings):
            msg = "comments and embeddings length mismatch."
            raise ValueError(msg)

        matrix = np.array(embeddings, dtype=np.float32)
        matrix = _normalize_matrix(matrix)
        clustering_matrix, reduction_summary = self._prepare_clustering_matrix(matrix)
        weights = np.array([comment.weight for comment in comments], dtype=np.float32)
        total_count = len(comments)
        total_weight = float(max(weights.sum(), 1e-6))

        if total_count < max(3, self.settings.cluster_min_size):
            cluster = self._single_emerging_cluster(matrix, weights, total_count)
            assignment = {
                idx: CommentClusterAssignment(primary_label=0, primary_confidence=1.0)
                for idx in range(total_count)
            }
            return ClusteringResult(
                clusters=[cluster],
                label_by_index=dict.fromkeys(range(total_count), cluster.cluster_key),
                assignment_by_index=assignment,
                reduction_summary=reduction_summary,
                parameter_summary={"algorithm": "single_emerging_cluster"},
            )

        fit_result = self._fit_predict_adaptive(clustering_matrix)
        labels = fit_result.labels.copy()
        assignments = self._build_assignments(
            labels=labels,
            probabilities=fit_result.probabilities,
            clustering_matrix=clustering_matrix,
        )
        labels, assignments = self._soft_assign_noise_points(
            labels=labels,
            assignments=assignments,
            clustering_matrix=clustering_matrix,
        )

        by_label: dict[int, list[int]] = {}
        for idx, label in enumerate(labels.tolist()):
            by_label.setdefault(label, []).append(idx)

        clusters: list[ClusterDraft] = []
        label_by_index: dict[int, str] = {}
        for next_cluster_index, label in enumerate(sorted(key for key in by_label if key != -1)):
            members = by_label[label]
            cluster = self._build_cluster(
                cluster_key=f"cluster_{next_cluster_index}",
                members=members,
                matrix=matrix,
                weights=weights,
                total_count=total_count,
                total_weight=total_weight,
                assignments=assignments,
            )
            clusters.append(cluster)
            for idx in members:
                label_by_index[idx] = cluster.cluster_key

        noise_members = by_label.get(-1, [])
        if noise_members:
            split_noise = self._split_large_noise_cluster(
                noise_members=noise_members,
                matrix=matrix,
                weights=weights,
                total_count=total_count,
                total_weight=total_weight,
                assignments=assignments,
            )
            if split_noise:
                for cluster in split_noise:
                    clusters.append(cluster)
                    for idx in cluster.member_indices:
                        label_by_index[idx] = cluster.cluster_key
            else:
                cluster = self._build_cluster(
                    cluster_key="emerging_other",
                    members=noise_members,
                    matrix=matrix,
                    weights=weights,
                    total_count=total_count,
                    total_weight=total_weight,
                    force_emerging=True,
                    assignments=assignments,
                )
                clusters.append(cluster)
                for idx in noise_members:
                    label_by_index[idx] = cluster.cluster_key

        if self.settings.cluster_large_split_enabled:
            clusters = self._split_large_clusters(
                clusters=clusters,
                matrix=matrix,
                weights=weights,
                total_count=total_count,
                total_weight=total_weight,
                assignments=assignments,
            )

        clusters = self._enforce_cluster_limit(
            clusters=clusters,
            matrix=matrix,
            weights=weights,
            total_count=total_count,
            total_weight=total_weight,
            assignments=assignments,
        )
        clusters = self._collapse_low_quality_clusters(
            clusters=clusters,
            matrix=matrix,
            weights=weights,
            total_count=total_count,
            total_weight=total_weight,
            assignments=assignments,
        )
        clusters.sort(key=lambda item: item.weighted_share, reverse=True)
        label_by_index = {}
        for cluster in clusters:
            for idx in cluster.member_indices:
                label_by_index[idx] = cluster.cluster_key
        return ClusteringResult(
            clusters=clusters,
            label_by_index=label_by_index,
            assignment_by_index=assignments,
            reduction_summary=reduction_summary,
            parameter_summary=fit_result.parameter_summary,
        )

    def _prepare_clustering_matrix(self, matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        total_count, input_dim = matrix.shape
        summary: dict[str, Any] = {
            "enabled": False,
            "method": "none",
            "input_dim": int(input_dim),
            "output_dim": int(input_dim),
        }
        if not self.settings.cluster_reduction_enabled:
            return matrix, summary
        if total_count < int(self.settings.cluster_reduction_min_comments):
            return matrix, summary
        target_dim = min(
            int(self.settings.cluster_reduction_target_dim),
            input_dim,
            max(2, total_count - 1),
        )
        if target_dim >= input_dim:
            return matrix, summary

        reducer = PCA(n_components=target_dim, random_state=42)
        reduced = reducer.fit_transform(matrix).astype(np.float32)
        reduced = _normalize_matrix(reduced)
        explained = float(np.sum(reducer.explained_variance_ratio_)) if hasattr(reducer, "explained_variance_ratio_") else 0.0
        summary.update(
            {
                "enabled": True,
                "method": "pca",
                "output_dim": int(target_dim),
                "explained_variance": round(explained, 4),
            }
        )
        return reduced, summary

    def _build_cluster(
        self,
        *,
        cluster_key: str,
        members: list[int],
        matrix: np.ndarray,
        weights: np.ndarray,
        total_count: int,
        total_weight: float,
        assignments: dict[int, CommentClusterAssignment] | None = None,
        force_emerging: bool = False,
    ) -> ClusterDraft:
        centroid = matrix[members].mean(axis=0)
        representatives = self._representatives(members, matrix, centroid, weights)
        size_count = len(members)
        share_pct = round(size_count / total_count * 100, 2)
        weighted_share = round(float(weights[members].sum() / total_weight * 100), 2)
        confidence = 0.0
        ambiguous_member_count = 0
        if assignments:
            member_assignments = [assignments[idx] for idx in members if idx in assignments]
            if member_assignments:
                confidence = float(
                    np.mean([assignment.primary_confidence for assignment in member_assignments])
                )
                ambiguous_member_count = sum(
                    1 for assignment in member_assignments if assignment.is_ambiguous
                )
        ambiguous_share_pct = round(ambiguous_member_count / max(1, size_count) * 100, 2)
        return ClusterDraft(
            cluster_key=cluster_key,
            member_indices=members,
            representative_indices=representatives,
            centroid=centroid.tolist(),
            size_count=size_count,
            share_pct=share_pct,
            weighted_share=weighted_share,
            is_emerging=force_emerging or size_count < self.settings.cluster_min_size,
            assignment_confidence=round(confidence, 4),
            ambiguous_member_count=ambiguous_member_count,
            ambiguous_share_pct=ambiguous_share_pct,
        )

    def _split_large_noise_cluster(
        self,
        *,
        noise_members: list[int],
        matrix: np.ndarray,
        weights: np.ndarray,
        total_count: int,
        total_weight: float,
        assignments: dict[int, CommentClusterAssignment] | None = None,
    ) -> list[ClusterDraft]:
        if not self.settings.cluster_noise_split_enabled:
            return []

        noise_size = len(noise_members)
        noise_share_pct = noise_size / max(1, total_count) * 100
        if noise_size < self.settings.cluster_noise_split_min_size:
            return []
        if noise_share_pct < self.settings.cluster_noise_split_min_share_pct:
            return []

        target_group_size = max(6, int(self.settings.cluster_noise_split_target_group_size))
        group_count = max(2, int(round(noise_size / max(1, target_group_size))))
        group_count = min(group_count, self.settings.cluster_noise_split_max_groups)
        if group_count < 2:
            return []

        noise_matrix = matrix[noise_members]
        try:
            kmeans = KMeans(n_clusters=group_count, n_init=10, random_state=42)
            labels = kmeans.fit_predict(noise_matrix)
        except Exception as exc:
            self.logger.warning("Could not split large noise cluster: %s", exc)
            return []

        grouped: dict[int, list[int]] = {}
        for local_idx, label in enumerate(labels.tolist()):
            grouped.setdefault(int(label), []).append(noise_members[local_idx])

        valid_groups = [
            members
            for members in grouped.values()
            if len(members) >= max(3, self.settings.cluster_min_samples)
        ]
        if len(valid_groups) < 2:
            return []
        if not self._passes_noise_split_quality_gate(valid_groups=valid_groups, matrix=matrix):
            self.logger.info(
                "Noise split quality gate rejected KMeans split (noise_size=%s share=%.1f%%).",
                noise_size,
                noise_share_pct,
            )
            return []

        valid_groups.sort(key=len, reverse=True)
        clusters: list[ClusterDraft] = []
        for idx, members in enumerate(valid_groups, start=1):
            clusters.append(
                self._build_cluster(
                    cluster_key=f"emerging_{idx}",
                    members=members,
                    matrix=matrix,
                    weights=weights,
                    total_count=total_count,
                    total_weight=total_weight,
                    force_emerging=True,
                    assignments=assignments,
                )
            )
        self.logger.info(
            "Split large noise cluster into %s groups (noise_size=%s share=%.1f%%).",
            len(clusters),
            noise_size,
            noise_share_pct,
        )
        return clusters

    def _passes_noise_split_quality_gate(
        self, *, valid_groups: list[list[int]], matrix: np.ndarray
    ) -> bool:
        if len(valid_groups) < 2:
            return False

        local_vectors: list[np.ndarray] = []
        local_labels: list[int] = []
        for group_idx, members in enumerate(valid_groups):
            for member_idx in members:
                local_vectors.append(matrix[member_idx])
                local_labels.append(group_idx)
        if len(local_vectors) < 4:
            return False

        local_matrix = np.array(local_vectors, dtype=np.float32)
        labels = np.array(local_labels, dtype=np.int32)
        unique_labels = set(labels.tolist())
        if len(unique_labels) < 2:
            return False
        try:
            silhouette = float(silhouette_score(local_matrix, labels, metric="euclidean"))
        except Exception:
            silhouette = -1.0
        silhouette_min = (
            float(self.settings.cluster_noise_split_min_silhouette_small)
            if local_matrix.shape[0] < 80
            else float(self.settings.cluster_noise_split_min_silhouette_large)
        )
        if silhouette < silhouette_min:
            return False

        coherence_scores = [
            self._group_coherence(members=members, matrix=matrix) for members in valid_groups
        ]
        avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        min_coherence = min(coherence_scores) if coherence_scores else 0.0
        return avg_coherence >= float(
            self.settings.cluster_noise_split_min_avg_coherence
        ) and min_coherence >= float(self.settings.cluster_noise_split_min_group_coherence)

    def _group_coherence(self, *, members: list[int], matrix: np.ndarray) -> float:
        if not members:
            return 0.0
        group_matrix = matrix[members]
        centroid = group_matrix.mean(axis=0)
        scores = [_cosine_similarity(vec, centroid) for vec in group_matrix]
        if not scores:
            return 0.0
        return float(max(0.0, min(1.0, np.mean(scores))))

    def _fit_predict_adaptive(self, matrix: np.ndarray) -> _FitResult:
        total_count = int(matrix.shape[0])
        candidates = self._candidate_params(total_count)
        best_labels: np.ndarray | None = None
        best_probs: np.ndarray | None = None
        best_summary: dict[str, Any] = {"algorithm": "hdbscan"}
        best_score = float("-inf")
        accept_noise_ratio = float(self.settings.cluster_accept_noise_ratio)
        accept_smallset_noise_ratio = float(self.settings.cluster_smallset_accept_noise_ratio)
        expected_clusters = max(2, min(self.settings.cluster_max_count, int(round(np.sqrt(total_count / 12.0))) + 1))

        for params in candidates:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=params["min_cluster_size"],
                min_samples=params["min_samples"],
                metric="euclidean",
                cluster_selection_epsilon=params["cluster_selection_epsilon"],
                prediction_data=True,
            )
            labels = clusterer.fit_predict(matrix)
            unique_labels = set(labels.tolist())
            cluster_count = len([label for label in unique_labels if label != -1])
            noise_count = int((labels == -1).sum())
            noise_ratio = noise_count / max(1, total_count)
            probs = np.array(getattr(clusterer, "probabilities_", np.ones(total_count)), dtype=np.float32)
            try:
                silhouette = (
                    float(silhouette_score(matrix, labels, metric="euclidean"))
                    if cluster_count >= 2
                    else -1.0
                )
            except Exception:
                silhouette = -1.0
            cluster_distance_penalty = abs(cluster_count - expected_clusters) / max(1, expected_clusters)
            score = (1.0 - noise_ratio) * 2.5 + max(silhouette, 0.0) * 2.0 - cluster_distance_penalty
            self.logger.info(
                "Clustering attempt min_cluster_size=%s min_samples=%s epsilon=%.3f -> clusters=%s noise_ratio=%.2f silhouette=%.3f score=%.3f",
                params["min_cluster_size"],
                params["min_samples"],
                params["cluster_selection_epsilon"],
                cluster_count,
                noise_ratio,
                silhouette,
                score,
            )

            if score > best_score:
                best_score = score
                best_labels = labels
                best_probs = probs
                best_summary = {
                    "algorithm": "hdbscan",
                    **params,
                    "cluster_count": cluster_count,
                    "noise_ratio": round(noise_ratio, 4),
                    "silhouette": round(silhouette, 4),
                    "expected_cluster_count": expected_clusters,
                }

            if cluster_count >= 2 and noise_ratio <= accept_noise_ratio and silhouette >= -0.05:
                return _FitResult(labels=labels, probabilities=probs, parameter_summary=best_summary)
            if total_count < 40 and cluster_count >= 1 and noise_ratio <= accept_smallset_noise_ratio:
                return _FitResult(labels=labels, probabilities=probs, parameter_summary=best_summary)

        if (
            bool(self.settings.cluster_kmeans_fallback_enabled)
            and (best_labels is None or len({label for label in best_labels.tolist() if label != -1}) <= 0)
            and total_count >= max(4, min(int(self.settings.cluster_kmeans_fallback_min_size), 6))
        ):
            fallback = self._fit_predict_kmeans_fallback(matrix)
            if fallback is not None:
                return fallback

        if best_labels is not None and best_probs is not None:
            return _FitResult(labels=best_labels, probabilities=best_probs, parameter_summary=best_summary)
        return _FitResult(
            labels=np.array([-1] * total_count, dtype=np.int32),
            probabilities=np.zeros(total_count, dtype=np.float32),
            parameter_summary={"algorithm": "all_noise"},
        )

    def _fit_predict_kmeans_fallback(self, matrix: np.ndarray) -> _FitResult | None:
        total_count = int(matrix.shape[0])
        if total_count < 4:
            return None

        configured_cap = int(self.settings.cluster_max_count)
        if configured_cap <= 0:
            max_k = min(total_count - 1, max(6, int(round(np.sqrt(total_count)))))
        else:
            max_k = min(total_count - 1, max(2, configured_cap))
        if max_k < 2:
            return None

        min_cluster_size = max(2, int(self.settings.cluster_min_size))
        best_labels: np.ndarray | None = None
        best_score = -1.0
        best_k = 0
        for k in range(2, max_k + 1):
            try:
                model = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = model.fit_predict(matrix).astype(np.int32)
            except Exception:
                continue
            counts = np.bincount(labels, minlength=k)
            if int(counts.min()) < max(2, min_cluster_size // 2):
                continue
            try:
                score = float(silhouette_score(matrix, labels, metric="euclidean"))
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_labels = labels
                best_k = k

        if best_labels is None:
            return None
        self.logger.info(
            "KMeans fallback selected (k=%s silhouette=%.3f).",
            best_k,
            best_score,
        )
        return _FitResult(
            labels=best_labels,
            probabilities=np.ones(total_count, dtype=np.float32) * 0.75,
            parameter_summary={
                "algorithm": "kmeans_fallback",
                "k": best_k,
                "silhouette": round(best_score, 4),
            },
        )

    def _candidate_params(self, total_count: int) -> list[dict[str, Any]]:
        dynamic_min_cluster = max(
            int(self.settings.cluster_min_size),
            min(36, max(4, int(round(total_count * 0.02)))),
        )
        variants = [
            dynamic_min_cluster,
            max(4, int(round(dynamic_min_cluster * 0.8))),
            min(40, int(round(dynamic_min_cluster * 1.25))),
        ]
        candidates: list[dict[str, Any]] = []
        for min_cluster_size in variants:
            min_samples_values = {
                max(1, int(self.settings.cluster_min_samples)),
                max(1, int(round(min_cluster_size * 0.35))),
                max(1, int(round(min_cluster_size * 0.5))),
            }
            epsilon_values = (0.0, 0.025, 0.05)
            for min_samples in sorted(min_samples_values):
                for epsilon in epsilon_values:
                    candidates.append(
                        {
                            "min_cluster_size": int(min_cluster_size),
                            "min_samples": int(min_samples),
                            "cluster_selection_epsilon": float(epsilon),
                        }
                    )
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[int, int, float]] = set()
        for candidate in candidates:
            key = (
                candidate["min_cluster_size"],
                candidate["min_samples"],
                candidate["cluster_selection_epsilon"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _build_assignments(
        self,
        *,
        labels: np.ndarray,
        probabilities: np.ndarray,
        clustering_matrix: np.ndarray,
    ) -> dict[int, CommentClusterAssignment]:
        assignments: dict[int, CommentClusterAssignment] = {}
        centroids = self._reduced_centroids(labels, clustering_matrix)
        confidence_threshold = float(self.settings.cluster_ambiguity_confidence_threshold)
        margin_threshold = float(self.settings.cluster_ambiguity_margin_threshold)

        for idx, label in enumerate(labels.tolist()):
            sims = self._sorted_centroid_similarities(clustering_matrix[idx], centroids)
            primary_sim = sims[0][1] if sims else 0.0
            secondary_label = sims[1][0] if len(sims) > 1 else None
            secondary_sim = sims[1][1] if len(sims) > 1 else 0.0
            if label == -1:
                assignments[idx] = CommentClusterAssignment(
                    primary_label=None,
                    primary_confidence=primary_sim,
                    secondary_label=secondary_label,
                    secondary_confidence=secondary_sim,
                    is_ambiguous=True,
                    remained_noise=True,
                )
                continue
            primary_conf = max(float(probabilities[idx]), primary_sim)
            assignments[idx] = CommentClusterAssignment(
                primary_label=int(label),
                primary_confidence=min(1.0, primary_conf),
                secondary_label=secondary_label,
                secondary_confidence=secondary_sim,
                is_ambiguous=(primary_conf < confidence_threshold)
                or ((primary_conf - secondary_sim) < margin_threshold),
            )
        return assignments

    def _soft_assign_noise_points(
        self,
        *,
        labels: np.ndarray,
        assignments: dict[int, CommentClusterAssignment],
        clustering_matrix: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, CommentClusterAssignment]]:
        if not self.settings.cluster_soft_assignment_enabled:
            return labels, assignments
        centroids = self._reduced_centroids(labels, clustering_matrix)
        if not centroids:
            return labels, assignments
        min_similarity = float(self.settings.cluster_soft_assignment_min_similarity)
        confidence_threshold = float(self.settings.cluster_ambiguity_confidence_threshold)
        margin_threshold = float(self.settings.cluster_ambiguity_margin_threshold)

        for idx, label in enumerate(labels.tolist()):
            if label != -1:
                continue
            sims = self._sorted_centroid_similarities(clustering_matrix[idx], centroids)
            if not sims:
                continue
            primary_label, primary_sim = sims[0]
            secondary_label = sims[1][0] if len(sims) > 1 else None
            secondary_sim = sims[1][1] if len(sims) > 1 else 0.0
            if primary_sim < min_similarity:
                continue
            labels[idx] = int(primary_label)
            assignments[idx] = CommentClusterAssignment(
                primary_label=int(primary_label),
                primary_confidence=primary_sim,
                secondary_label=secondary_label,
                secondary_confidence=secondary_sim,
                is_ambiguous=(primary_sim < confidence_threshold)
                or ((primary_sim - secondary_sim) < margin_threshold),
                used_soft_assignment=True,
                remained_noise=False,
            )
        return labels, assignments

    def _reduced_centroids(
        self,
        labels: np.ndarray,
        clustering_matrix: np.ndarray,
    ) -> dict[int, np.ndarray]:
        centroids: dict[int, np.ndarray] = {}
        for label in sorted({int(item) for item in labels.tolist() if int(item) != -1}):
            members = np.where(labels == label)[0]
            if len(members) == 0:
                continue
            centroid = clustering_matrix[members].mean(axis=0)
            centroids[int(label)] = centroid
        return centroids

    def _sorted_centroid_similarities(
        self,
        vector: np.ndarray,
        centroids: dict[int, np.ndarray],
    ) -> list[tuple[int, float]]:
        sims = [
            (label, max(0.0, _cosine_similarity(vector, centroid)))
            for label, centroid in centroids.items()
        ]
        sims.sort(key=lambda item: item[1], reverse=True)
        return sims

    def _split_large_clusters(
        self,
        *,
        clusters: list[ClusterDraft],
        matrix: np.ndarray,
        weights: np.ndarray,
        total_count: int,
        total_weight: float,
        assignments: dict[int, CommentClusterAssignment] | None = None,
    ) -> list[ClusterDraft]:
        min_share = float(self.settings.cluster_large_split_min_share_pct)
        max_subgroups = int(self.settings.cluster_large_split_max_subgroups)

        result: list[ClusterDraft] = []
        for cluster in clusters:
            if cluster.share_pct < min_share or cluster.size_count < 12:
                result.append(cluster)
                continue

            members = cluster.member_indices
            target_group_size = max(6, int(self.settings.cluster_noise_split_target_group_size))
            k = max(2, min(max_subgroups, int(round(len(members) / target_group_size))))

            sub_matrix = matrix[members]
            try:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(sub_matrix)
            except Exception as exc:
                self.logger.warning("Could not split large cluster %s: %s", cluster.cluster_key, exc)
                result.append(cluster)
                continue

            grouped: dict[int, list[int]] = {}
            for local_idx, label in enumerate(labels.tolist()):
                grouped.setdefault(int(label), []).append(members[local_idx])

            valid_groups = [group for group in grouped.values() if len(group) >= 3]
            if len(valid_groups) < 2:
                result.append(cluster)
                continue

            if not self._passes_noise_split_quality_gate(valid_groups=valid_groups, matrix=matrix):
                self.logger.info(
                    "Large cluster split quality gate rejected %s (size=%s share=%.1f%%).",
                    cluster.cluster_key,
                    cluster.size_count,
                    cluster.share_pct,
                )
                result.append(cluster)
                continue

            valid_groups.sort(key=len, reverse=True)
            for sub_idx, sub_members in enumerate(valid_groups):
                result.append(
                    self._build_cluster(
                        cluster_key=f"{cluster.cluster_key}_sub{sub_idx}",
                        members=sub_members,
                        matrix=matrix,
                        weights=weights,
                        total_count=total_count,
                        total_weight=total_weight,
                        assignments=assignments,
                    )
                )
            self.logger.info(
                "Split large cluster %s into %s sub-groups (size=%s share=%.1f%%).",
                cluster.cluster_key,
                len(valid_groups),
                cluster.size_count,
                cluster.share_pct,
            )

        return result

    def _enforce_cluster_limit(
        self,
        *,
        clusters: list[ClusterDraft],
        matrix: np.ndarray,
        weights: np.ndarray,
        total_count: int,
        total_weight: float,
        assignments: dict[int, CommentClusterAssignment] | None = None,
    ) -> list[ClusterDraft]:
        max_count = int(self.settings.cluster_max_count)
        if max_count <= 0 or len(clusters) <= max_count:
            return clusters
        if max_count == 1:
            all_members = sorted({idx for cluster in clusters for idx in cluster.member_indices})
            single = self._build_cluster(
                cluster_key="emerging_other",
                members=all_members,
                matrix=matrix,
                weights=weights,
                total_count=total_count,
                total_weight=total_weight,
                force_emerging=True,
                assignments=assignments,
            )
            return [single]

        ordered = sorted(clusters, key=lambda item: item.weighted_share, reverse=True)
        keep_count = max(1, max_count - 1)
        kept = [cluster.model_copy(deep=True) for cluster in ordered[:keep_count]]
        overflow = ordered[keep_count:]
        overflow_members: list[int] = []
        for cluster in overflow:
            overflow_members.extend(cluster.member_indices)

        overflow_members = sorted(set(overflow_members))
        if not overflow_members:
            return kept

        overflow_cluster = self._build_cluster(
            cluster_key="emerging_other",
            members=overflow_members,
            matrix=matrix,
            weights=weights,
            total_count=total_count,
            total_weight=total_weight,
            force_emerging=True,
            assignments=assignments,
        )

        merged_existing = False
        for idx, cluster in enumerate(kept):
            if cluster.cluster_key != "emerging_other":
                continue
            merged_members = sorted(set(cluster.member_indices).union(overflow_members))
            kept[idx] = self._build_cluster(
                cluster_key="emerging_other",
                members=merged_members,
                matrix=matrix,
                weights=weights,
                total_count=total_count,
                total_weight=total_weight,
                force_emerging=True,
                assignments=assignments,
            )
            merged_existing = True
            break
        if not merged_existing:
            kept.append(overflow_cluster)

        self.logger.info(
            "Cluster count capped: %s -> %s (overflow members moved to emerging_other=%s).",
            len(clusters),
            len(kept),
            len(overflow_members),
        )
        return kept

    def _collapse_low_quality_clusters(
        self,
        *,
        clusters: list[ClusterDraft],
        matrix: np.ndarray,
        weights: np.ndarray,
        total_count: int,
        total_weight: float,
        assignments: dict[int, CommentClusterAssignment] | None = None,
    ) -> list[ClusterDraft]:
        if len(clusters) < 2:
            return clusters

        keep: list[ClusterDraft] = []
        overflow_members: list[int] = []
        for cluster in clusters:
            coherence = self._group_coherence(members=cluster.member_indices, matrix=matrix)
            is_small = cluster.size_count < max(12, self.settings.cluster_min_size * 2)
            is_low_share = cluster.share_pct < max(6.0, self.settings.cluster_large_split_min_share_pct / 4)
            is_ambiguous = cluster.ambiguous_share_pct >= 45.0
            if (
                is_small
                and is_low_share
                and (
                    coherence < max(0.28, self.settings.topic_coherence_min - 0.08)
                    or is_ambiguous
                )
            ):
                overflow_members.extend(cluster.member_indices)
                self.logger.info(
                    "Collapsed low-quality cluster %s into emerging_other (size=%s share=%.1f%% coherence=%.3f ambiguous=%.1f%%).",
                    cluster.cluster_key,
                    cluster.size_count,
                    cluster.share_pct,
                    coherence,
                    cluster.ambiguous_share_pct,
                )
                continue
            keep.append(cluster)

        if not overflow_members:
            return keep

        overflow_cluster = self._build_cluster(
            cluster_key="emerging_other",
            members=sorted(set(overflow_members)),
            matrix=matrix,
            weights=weights,
            total_count=total_count,
            total_weight=total_weight,
            force_emerging=True,
            assignments=assignments,
        )
        for idx, cluster in enumerate(keep):
            if cluster.cluster_key != "emerging_other":
                continue
            merged_members = sorted(set(cluster.member_indices).union(overflow_cluster.member_indices))
            keep[idx] = self._build_cluster(
                cluster_key="emerging_other",
                members=merged_members,
                matrix=matrix,
                weights=weights,
                total_count=total_count,
                total_weight=total_weight,
                force_emerging=True,
                assignments=assignments,
            )
            break
        else:
            keep.append(overflow_cluster)
        return keep

    def _single_emerging_cluster(
        self,
        matrix: np.ndarray,
        weights: np.ndarray,
        total_count: int,
    ) -> ClusterDraft:
        members = list(range(total_count))
        centroid = matrix.mean(axis=0)
        representatives = self._representatives(members, matrix, centroid, weights)
        return ClusterDraft(
            cluster_key="emerging_other",
            member_indices=members,
            representative_indices=representatives,
            centroid=centroid.tolist(),
            size_count=total_count,
            share_pct=100.0,
            weighted_share=100.0,
            is_emerging=True,
            assignment_confidence=1.0,
            ambiguous_member_count=0,
            ambiguous_share_pct=0.0,
        )

    def _adaptive_representative_count(self, member_count: int) -> int:
        if member_count <= 0:
            return 1
        target = int(round(float(np.sqrt(member_count))))
        target = min(30, max(8, target))
        return min(member_count, target)

    def _representatives(
        self,
        members: list[int],
        matrix: np.ndarray,
        centroid: np.ndarray,
        weights: np.ndarray,
    ) -> list[int]:
        max_w = float(max(weights[members].max(), 1e-6))
        scored: list[tuple[int, float]] = []
        for idx in members:
            sim = _cosine_similarity(matrix[idx], centroid)
            norm_weight = float(weights[idx]) / max_w
            score = sim * 0.7 + norm_weight * 0.3
            scored.append((idx, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        limit = self._adaptive_representative_count(len(scored))
        return [idx for idx, _ in scored[:limit]]
