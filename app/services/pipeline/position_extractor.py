"""Position extraction logic for the analysis pipeline.

Extracts distinct audience positions (sub-opinions) within a thematic
cluster of YouTube comments.  Supports two strategies:

1. **LLM-based** -- a single LLM call identifies all positions and assigns
   comments to them.
2. **Embedding-based fallback** -- HDBSCAN / KMeans subclustering followed
   by per-position LLM naming.

The public entry point is
:meth:`PositionExtractor.build_positions_for_cluster`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from typing import Any

import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.core.config import Settings
from app.core.exceptions import BudgetExceededError, ExternalServiceError
from app.schemas.domain import ClusterDraft, ProcessedComment, TopicPosition
from app.services.labeling import LLMProvider, OpenAIChatProvider

from .text_utils import (
    _UNDETERMINED_POSITION_KEY,
    _UNDETERMINED_POSITION_TITLE,
    clip_text_for_llm,
    comment_text_for_output,
    extract_salient_comment_keywords,
    position_title_single_claim_passed,
    sanitize_cluster_title,
)

# ---------------------------------------------------------------------------
# Type aliases for callback signatures
# ---------------------------------------------------------------------------

#: Signature for the LLM JSON request callable.
#: ``(provider, prompt, *, task, estimated_out_tokens, max_output_tokens, system_prompt) -> dict | None``
RequestLLMJsonFn = Callable[..., dict[str, Any] | None]

#: Signature for the cluster-member ranking callable.
#: ``(cluster, comments, vectors) -> list[int]``
RankClusterMemberIndicesFn = Callable[
    [ClusterDraft, list[ProcessedComment], list[list[float]]],
    list[int],
]


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the public API)
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    vec_a = np.array(a, dtype=np.float32)
    vec_b = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _mark_llm_disabled(diagnostics: dict[str, Any], *, reason: str) -> None:
    """Record an LLM-disable event in the diagnostics dict."""
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return
    events = diagnostics.setdefault("llm_disable_events", [])
    if isinstance(events, list):
        events.append(normalized)
    if not diagnostics.get("llm_disabled_reason"):
        diagnostics["llm_disabled_reason"] = normalized


# ---------------------------------------------------------------------------
# PositionExtractor
# ---------------------------------------------------------------------------


class PositionExtractor:
    """Extracts audience positions from within a thematic cluster.

    Parameters
    ----------
    settings:
        Application configuration.
    logger:
        Logger instance for diagnostic messages.
    """

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_positions_for_cluster(
        self,
        *,
        llm_provider: LLMProvider,
        cluster: ClusterDraft,
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        llm_disabled: bool,
        cluster_title: str,
        request_llm_json: RequestLLMJsonFn,
        rank_cluster_member_indices: RankClusterMemberIndicesFn,
        diagnostics: dict[str, Any] | None = None,
    ) -> tuple[list[TopicPosition], bool]:
        """Build a list of audience positions for a single cluster.

        Attempts LLM-based identification first; falls back to
        embedding-based subclustering when the LLM is unavailable or
        returns unusable output.

        Parameters
        ----------
        llm_provider:
            LLM provider instance to use for position identification.
        cluster:
            The cluster draft whose members are being analysed.
        comments:
            Full list of processed comments (indexed by global index).
        vectors:
            Embedding vectors aligned with *comments*.
        llm_disabled:
            Whether LLM calls are currently disabled (e.g. budget exhausted).
        cluster_title:
            Human-readable title already assigned to the cluster.
        request_llm_json:
            Callable that sends a prompt to the LLM and returns parsed JSON.
        rank_cluster_member_indices:
            Callable that ranks member indices by relevance within a cluster.
        diagnostics:
            Optional mutable dict for recording diagnostic counters.

        Returns
        -------
        tuple[list[TopicPosition], bool]
            The extracted positions and the (possibly updated) ``llm_disabled`` flag.
        """
        member_indices = [
            idx
            for idx in cluster.member_indices
            if 0 <= idx < len(comments) and 0 <= idx < len(vectors)
        ]
        if not member_indices:
            return [], llm_disabled

        # Try LLM-based position identification first (produces granular positions).
        llm_positions, llm_disabled = self.identify_positions_with_llm(
            llm_provider=llm_provider,
            member_indices=member_indices,
            comments=comments,
            vectors=vectors,
            cluster_title=cluster_title,
            llm_disabled=llm_disabled,
            request_llm_json=request_llm_json,
            rank_cluster_member_indices=rank_cluster_member_indices,
            diagnostics=diagnostics,
        )
        if llm_positions is not None:
            return llm_positions, llm_disabled

        # Fallback: embedding-based subclustering + per-position naming.
        grouped_members, undetermined_members = self.subcluster_member_groups(
            member_indices=member_indices, vectors=vectors
        )
        total_count = len(member_indices)
        total_weight = float(sum(max(0.0, float(comments[idx].weight)) for idx in member_indices))
        llm_naming_min_group_size = max(1, int(self.settings.position_llm_naming_min_group_size))

        positions: list[TopicPosition] = []
        for pos_idx, members in enumerate(grouped_members, start=1):
            summary_placeholder = f"Комментарии с близким нарративом внутри темы «{cluster_title}»."
            markers_placeholder = extract_salient_comment_keywords(
                [comment_text_for_output(comments[idx]) for idx in members],
                max_keywords=4,
            )
            prototype = self.build_position_prototype(members, comments, vectors)

            title = ""
            summary = summary_placeholder
            markers = markers_placeholder[:5]
            used_fallback_title = True
            if not llm_disabled and len(members) >= llm_naming_min_group_size:
                named, llm_disabled, used_fallback_title = self.name_position_with_llm(
                    llm_provider=llm_provider,
                    members=members,
                    comments=comments,
                    vectors=vectors,
                    cluster_title=cluster_title,
                    fallback_title=cluster_title,
                    fallback_summary=summary_placeholder,
                    fallback_markers=markers_placeholder,
                    llm_disabled=llm_disabled,
                    request_llm_json=request_llm_json,
                    rank_cluster_member_indices=rank_cluster_member_indices,
                    diagnostics=diagnostics,
                )
                title = named["title"]
                summary = named["summary"]
                markers = named["markers"]

            if not title:
                title = f"[Ошибка] Позиция #{pos_idx}"
                self.logger.error(
                    "Position naming failed for group #%d in cluster '%s' — LLM unavailable or returned empty.",
                    pos_idx,
                    cluster_title,
                )

            single_claim_passed = position_title_single_claim_passed(title)

            count = len(members)
            weighted_count = float(sum(max(0.0, float(comments[idx].weight)) for idx in members))
            coherence_score = self.estimate_position_coherence(
                members=members, comments=comments, vectors=vectors
            )
            positions.append(
                TopicPosition(
                    key=f"pos_{pos_idx}",
                    title=title,
                    summary=summary,
                    markers=markers[:5],
                    prototype=prototype,
                    count=count,
                    pct=(count / total_count) * 100.0,
                    weighted_count=weighted_count,
                    weighted_pct=(
                        ((weighted_count / total_weight) * 100.0) if total_weight > 0 else 0.0
                    ),
                    comments=[
                        comment_text_for_output(comments[idx])
                        for idx in members
                        if comment_text_for_output(comments[idx])
                    ],
                    is_undetermined=False,
                    coherence_score=coherence_score,
                    single_claim_passed=single_claim_passed,
                )
            )

        if undetermined_members:
            count = len(undetermined_members)
            weighted_count = float(
                sum(max(0.0, float(comments[idx].weight)) for idx in undetermined_members)
            )
            positions.append(
                TopicPosition(
                    key=_UNDETERMINED_POSITION_KEY,
                    title=_UNDETERMINED_POSITION_TITLE,
                    summary="Комментарии из малых/шумовых подгрупп без устойчивого нарратива (другое/неясно).",
                    markers=[],
                    prototype="Комментарии не образуют устойчивую позицию и объединены в fallback-категорию.",
                    count=count,
                    pct=(count / total_count) * 100.0,
                    weighted_count=weighted_count,
                    weighted_pct=(
                        ((weighted_count / total_weight) * 100.0) if total_weight > 0 else 0.0
                    ),
                    comments=[
                        comment_text_for_output(comments[idx])
                        for idx in undetermined_members
                        if comment_text_for_output(comments[idx])
                    ],
                    is_undetermined=True,
                    coherence_score=0.0,
                    single_claim_passed=True,
                )
            )

        # Safety net: if no positions were produced at all, create a single
        # catch-all position so the report never falls back to flat comments.
        if not positions:
            self.logger.warning(
                "No positions produced for cluster '%s' (size=%d) — creating catch-all position.",
                cluster_title,
                total_count,
            )
            weighted_all = float(
                sum(max(0.0, float(comments[idx].weight)) for idx in member_indices)
            )
            prototype = self.build_position_prototype(member_indices, comments, vectors)
            positions.append(
                TopicPosition(
                    key="pos_1",
                    title=cluster_title or "[Ошибка] Позиция не определена",
                    summary=f"Все комментарии в кластере «{cluster_title}».",
                    markers=extract_salient_comment_keywords(
                        [comment_text_for_output(comments[idx]) for idx in member_indices],
                        max_keywords=4,
                    )[:5],
                    prototype=prototype,
                    count=total_count,
                    pct=100.0,
                    weighted_count=weighted_all,
                    weighted_pct=100.0,
                    comments=[
                        comment_text_for_output(comments[idx])
                        for idx in member_indices
                        if comment_text_for_output(comments[idx])
                    ],
                    is_undetermined=False,
                    coherence_score=0.0,
                    single_claim_passed=True,
                )
            )

        positions.sort(key=lambda item: (item.is_undetermined, -item.count))
        return positions, llm_disabled

    # ------------------------------------------------------------------
    # LLM-based position identification
    # ------------------------------------------------------------------

    def identify_positions_with_llm(
        self,
        *,
        llm_provider: LLMProvider,
        member_indices: list[int],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        cluster_title: str,
        llm_disabled: bool,
        request_llm_json: RequestLLMJsonFn,
        rank_cluster_member_indices: RankClusterMemberIndicesFn,
        diagnostics: dict[str, Any] | None = None,
    ) -> tuple[list[TopicPosition] | None, bool]:
        """Use a single LLM call to identify all distinct positions within a cluster and assign comments.

        Parameters
        ----------
        llm_provider:
            LLM provider instance.
        member_indices:
            Global indices of comments belonging to the cluster.
        comments:
            Full list of processed comments.
        vectors:
            Embedding vectors aligned with *comments*.
        cluster_title:
            Human-readable cluster title for context.
        llm_disabled:
            Whether LLM calls are currently disabled.
        request_llm_json:
            Callable that sends a prompt to the LLM and returns parsed JSON.
        rank_cluster_member_indices:
            Callable that ranks member indices by relevance within a cluster.
        diagnostics:
            Optional mutable dict for recording diagnostic counters.

        Returns
        -------
        tuple[list[TopicPosition] | None, bool]
            Extracted positions (or ``None`` on failure) and the updated
            ``llm_disabled`` flag.
        """
        if llm_disabled:
            return None, llm_disabled
        if len(member_indices) < 4:
            return None, llm_disabled

        # Build a ranked sample of comments for the LLM.
        max_sample = min(60, max(20, len(member_indices)))
        comment_char_cap = max(80, int(self.settings.position_llm_comment_char_cap))
        payload_char_budget = max(4000, int(self.settings.position_llm_payload_char_budget))

        local_cluster = ClusterDraft(
            cluster_key="pos_id",
            member_indices=list(member_indices),
            representative_indices=list(member_indices[:4]),
            centroid=self.compute_group_centroid(member_indices, vectors),
            size_count=len(member_indices),
            share_pct=100.0,
            weighted_share=100.0,
            is_emerging=False,
        )
        ranked = rank_cluster_member_indices(local_cluster, comments, vectors)
        if not ranked:
            ranked = list(member_indices)

        if len(ranked) <= max_sample:
            sample_indices = list(ranked)
        else:
            step = max(1, len(ranked) // max_sample)
            sample_indices = [ranked[i] for i in range(0, len(ranked), step)][:max_sample]

        used_chars = 0
        payload: list[dict[str, Any]] = []
        id_to_member_idx: dict[str, int] = {}
        for seq, idx in enumerate(sample_indices):
            text = comment_text_for_output(comments[idx])
            if not text:
                continue
            text = clip_text_for_llm(text, max_chars=comment_char_cap)
            remaining = payload_char_budget - used_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = clip_text_for_llm(text, max_chars=remaining)
            if not text:
                continue
            cid = f"c{seq + 1:03d}"
            payload.append({"id": cid, "text": text})
            id_to_member_idx[cid] = idx
            used_chars += len(text)

        if len(payload) < 4:
            return None, llm_disabled

        prompt = (
            "Ты анализируешь комментарии из одного тематического кластера YouTube-канала.\n"
            "Задача: определить ВСЕ различные позиции, точки зрения и мнения в этих комментариях.\n\n"
            "Правила:\n"
            "- Каждая позиция должна выражать ОДНУ конкретную точку зрения, запрос или мнение\n"
            "- Позиции должны быть максимально гранулярными: НЕ объединяй разные мнения в одну позицию\n"
            "- Примеры хороших позиций для кластера 'Отношение к политике автора': "
            "'Поддержка позиции автора', 'Критика предвзятости автора', "
            "'Запрос на источники и факты', 'Благодарность за аналитику'\n"
            "- Заголовок позиции: 3-8 слов на русском, выражающих суть именно этой позиции\n"
            "- summary: одно предложение, описывающее общую суть комментариев в этой позиции\n"
            "- markers: 2-4 ключевых аргумента или маркера этой позиции\n"
            "- comment_ids: список ID комментариев, относящихся к данной позиции\n"
            "- Каждый комментарий должен быть отнесён ровно к ОДНОЙ позиции\n"
            "- Стремись к 3-12 позициям в зависимости от разнообразия комментариев\n"
            "- Не создавай позиции с одним комментарием\n"
            "- Все заголовки и текст должны быть на русском языке\n"
            f"- is_author_disagreement: true если позиция выражает несогласие, критику или спор с автором канала ({self.settings.author_name or 'автор канала'}). "
            "Это включает: оспаривание фактов, критику методов, обвинения в предвзятости, несогласие с выводами автора\n\n"
            "Верни строго JSON:\n"
            "{\n"
            '  "positions": [\n'
            "    {\n"
            '      "title": "заголовок позиции на русском",\n'
            '      "summary": "одно предложение описание",\n'
            '      "markers": ["аргумент 1", "аргумент 2"],\n'
            '      "comment_ids": ["c001", "c002"],\n'
            '      "is_author_disagreement": false\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Название кластера: \u00ab{cluster_title}\u00bb\n"
            f"Комментарии ({len(payload)}):\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        author = self.settings.author_name or "автор канала"
        system_prompt = (
            f"Ты аналитик комментариев YouTube-канала {author}. "
            "Найди все различные позиции аудитории внутри одного тематического кластера. "
            "Позиции должны быть гранулярными \u2014 не смешивай разные мнения. "
            "Отвечай строго JSON на русском языке."
        )

        try:
            data = request_llm_json(
                llm_provider,
                prompt,
                task="position_naming",
                estimated_out_tokens=1200,
                max_output_tokens=2048,
                system_prompt=system_prompt,
            )
        except (BudgetExceededError, ExternalServiceError) as exc:
            message = str(exc).lower()
            if (
                isinstance(exc, BudgetExceededError)
                or "429" in message
                or "rate-limit" in message
                or "rate limit" in message
            ):
                llm_disabled = True
                if diagnostics is not None:
                    _mark_llm_disabled(
                        diagnostics,
                        reason=(
                            "budget_exceeded"
                            if isinstance(exc, BudgetExceededError)
                            else "rate_limited"
                        ),
                    )
            elif diagnostics is not None:
                _mark_llm_disabled(diagnostics, reason="external_service_error")
            self.logger.warning(
                "Position identification failed via %s: %s", llm_provider.provider_name, exc
            )
            return None, llm_disabled

        if not data or "positions" not in data:
            self.logger.warning(
                "Position identification returned empty/invalid response for cluster '%s'",
                cluster_title,
            )
            return None, llm_disabled

        raw_positions = data.get("positions", [])
        if not isinstance(raw_positions, list) or len(raw_positions) < 2:
            self.logger.warning(
                "Position identification returned fewer than 2 positions for cluster '%s'",
                cluster_title,
            )
            return None, llm_disabled

        # Parse positions and assign sample comments.
        total_count = len(member_indices)
        total_weight = float(sum(max(0.0, float(comments[idx].weight)) for idx in member_indices))

        position_groups: list[dict[str, Any]] = []
        assigned_indices: set[int] = set()

        for pos_data in raw_positions:
            if not isinstance(pos_data, dict):
                continue
            title = " ".join(str(pos_data.get("title", "") or "").split()).strip()
            if not title or len(title) < 3:
                continue
            summary = " ".join(str(pos_data.get("summary", "") or "").split()).strip()
            raw_markers = pos_data.get("markers", [])
            markers: list[str] = []
            if isinstance(raw_markers, list):
                seen_markers: set[str] = set()
                for m in raw_markers:
                    cleaned = " ".join(str(m or "").split()).strip(" -:;,.")
                    if cleaned and cleaned.lower() not in seen_markers:
                        seen_markers.add(cleaned.lower())
                        markers.append(cleaned)
                    if len(markers) >= 5:
                        break

            comment_ids = pos_data.get("comment_ids", [])
            if not isinstance(comment_ids, list):
                comment_ids = []

            members: list[int] = []
            for cid in comment_ids:
                cid_str = str(cid).strip()
                if cid_str in id_to_member_idx:
                    idx = id_to_member_idx[cid_str]
                    if idx not in assigned_indices:
                        members.append(idx)
                        assigned_indices.add(idx)

            if members:
                is_author_disagreement = bool(pos_data.get("is_author_disagreement", False))
                position_groups.append(
                    {
                        "title": title,
                        "summary": summary or f"Комментарии с позицией \u00ab{title}\u00bb.",
                        "markers": markers,
                        "members": members,
                        "is_author_disagreement": is_author_disagreement,
                    }
                )

        if len(position_groups) < 2:
            self.logger.warning(
                "Position identification produced fewer than 2 valid groups for cluster '%s'",
                cluster_title,
            )
            return None, llm_disabled

        # Assign remaining members (unsampled + unassigned) via embedding similarity.
        # Comments below the minimum similarity threshold go to an "undetermined"
        # fallback group instead of polluting the closest named position.
        min_sim_threshold = max(0.0, float(self.settings.position_assignment_min_similarity))
        undetermined_members: list[int] = []
        unassigned = [idx for idx in member_indices if idx not in assigned_indices]
        if unassigned:
            centroids = [
                self.compute_group_centroid(pg["members"], vectors) for pg in position_groups
            ]
            for idx in unassigned:
                vec = vectors[idx]
                sims = [_cosine_similarity(vec, c) for c in centroids]
                if not sims:
                    undetermined_members.append(idx)
                    continue
                best_pos = max(range(len(sims)), key=lambda i: sims[i])
                if sims[best_pos] < min_sim_threshold:
                    undetermined_members.append(idx)
                else:
                    position_groups[best_pos]["members"].append(idx)

        # Build TopicPosition objects.
        positions: list[TopicPosition] = []
        for pos_idx, pg in enumerate(position_groups, start=1):
            members = pg["members"]
            if not members:
                continue
            count = len(members)
            weighted_count = float(sum(max(0.0, float(comments[idx].weight)) for idx in members))
            coherence_score = self.estimate_position_coherence(
                members=members,
                comments=comments,
                vectors=vectors,
            )
            prototype = self.build_position_prototype(members, comments, vectors)
            title = pg["title"]
            single_claim_passed = position_title_single_claim_passed(title)

            positions.append(
                TopicPosition(
                    key=f"pos_{pos_idx}",
                    title=title,
                    summary=pg["summary"],
                    markers=pg["markers"][:5],
                    prototype=prototype,
                    count=count,
                    pct=(count / total_count) * 100.0 if total_count > 0 else 0.0,
                    weighted_count=weighted_count,
                    weighted_pct=(
                        ((weighted_count / total_weight) * 100.0) if total_weight > 0 else 0.0
                    ),
                    comments=[
                        comment_text_for_output(comments[idx])
                        for idx in members
                        if comment_text_for_output(comments[idx])
                    ],
                    is_undetermined=False,
                    is_author_disagreement=bool(pg.get("is_author_disagreement", False)),
                    coherence_score=coherence_score,
                    single_claim_passed=single_claim_passed,
                )
            )

        if not positions:
            return None, llm_disabled

        # Add undetermined position for comments that didn't meet the similarity threshold
        if undetermined_members:
            und_count = len(undetermined_members)
            und_weighted = float(
                sum(max(0.0, float(comments[idx].weight)) for idx in undetermined_members)
            )
            positions.append(
                TopicPosition(
                    key=_UNDETERMINED_POSITION_KEY,
                    title=_UNDETERMINED_POSITION_TITLE,
                    summary="Комментарии, не вошедшие ни в одну из выделенных позиций по порогу сходства.",
                    markers=[],
                    prototype=(
                        self.build_position_prototype(undetermined_members, comments, vectors)
                        if undetermined_members
                        else ""
                    ),
                    count=und_count,
                    pct=(und_count / total_count) * 100.0 if total_count > 0 else 0.0,
                    weighted_count=und_weighted,
                    weighted_pct=(
                        ((und_weighted / total_weight) * 100.0) if total_weight > 0 else 0.0
                    ),
                    comments=[
                        comment_text_for_output(comments[idx])
                        for idx in undetermined_members
                        if comment_text_for_output(comments[idx])
                    ],
                    is_undetermined=True,
                    coherence_score=0.0,
                    single_claim_passed=False,
                )
            )

        positions.sort(key=lambda p: (p.is_undetermined, -p.count))
        self.logger.info(
            "LLM position identification succeeded: %d positions for cluster '%s'",
            len(positions),
            cluster_title,
        )
        return positions, llm_disabled

    # ------------------------------------------------------------------
    # Embedding-based subclustering
    # ------------------------------------------------------------------

    def subcluster_member_groups(
        self,
        *,
        member_indices: list[int],
        vectors: list[list[float]],
    ) -> tuple[list[list[int]], list[int]]:
        """Split *member_indices* into sub-groups using HDBSCAN or KMeans.

        Parameters
        ----------
        member_indices:
            Global comment indices belonging to the cluster.
        vectors:
            Embedding vectors aligned with the full comment list.

        Returns
        -------
        tuple[list[list[int]], list[int]]
            A list of major groups and a list of undetermined (noise) indices.
        """
        if len(member_indices) <= 2:
            return [list(member_indices)], []

        local_matrix = np.array([vectors[idx] for idx in member_indices], dtype=np.float32)
        labels = self.fit_subcluster_labels(local_matrix)

        groups: dict[int, list[int]] = {}
        noise_members: list[int] = []
        for local_idx, raw_label in enumerate(labels.tolist()):
            member_idx = member_indices[local_idx]
            label = int(raw_label)
            if label < 0:
                noise_members.append(member_idx)
                continue
            groups.setdefault(label, []).append(member_idx)

        if not groups:
            return [list(member_indices)], []

        min_group_size = max(
            max(1, int(self.settings.position_subcluster_min_group_size)),
            int(
                math.ceil(
                    len(member_indices)
                    * (
                        max(0.0, float(self.settings.position_subcluster_min_group_share_pct))
                        / 100.0
                    )
                )
            ),
        )
        major_groups: list[list[int]] = []
        tiny_groups: list[list[int]] = []
        for members in groups.values():
            if len(members) >= min_group_size:
                major_groups.append(sorted(members))
            else:
                tiny_groups.append(sorted(members))

        if not major_groups:
            largest = max(groups.values(), key=len)
            major_groups = [sorted(largest)]
            leftovers = [idx for idx in member_indices if idx not in set(largest)]
            if leftovers:
                noise_members.extend(leftovers)

        centroids = [self.compute_group_centroid(members, vectors) for members in major_groups]
        undetermined: list[int] = []
        assign_min_similarity = max(0.0, float(self.settings.position_assignment_min_similarity))
        assign_min_margin = max(0.0, float(self.settings.position_assignment_min_margin))

        for group_members in tiny_groups + [noise_members]:
            for member_idx in group_members:
                member_vec = vectors[member_idx]
                sims = [_cosine_similarity(member_vec, centroid) for centroid in centroids]
                if not sims:
                    undetermined.append(member_idx)
                    continue
                best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
                best_sim = float(sims[best_idx])
                second_sim = max(
                    (float(sim) for idx, sim in enumerate(sims) if idx != best_idx), default=-1.0
                )
                margin = best_sim - second_sim
                if best_sim >= assign_min_similarity and margin >= assign_min_margin:
                    major_groups[best_idx].append(member_idx)
                else:
                    undetermined.append(member_idx)

        # Ensure deterministic order and full one-to-one coverage.
        normalized_groups: list[list[int]] = []
        covered: set[int] = set()
        for group in major_groups:
            uniq = sorted(idx for idx in set(group) if idx not in covered)
            if not uniq:
                continue
            covered.update(uniq)
            normalized_groups.append(uniq)
        undetermined = sorted(idx for idx in set(undetermined) if idx not in covered)
        normalized_groups.sort(key=len, reverse=True)
        return normalized_groups, undetermined

    def fit_subcluster_labels(self, matrix: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN or KMeans and return cluster labels for each row.

        Parameters
        ----------
        matrix:
            2-D array of shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Integer labels of shape ``(n_samples,)``.  Noise is ``-1`` for
            HDBSCAN; all labels are ``>= 0`` for KMeans.
        """
        total_count = int(matrix.shape[0])
        if total_count <= 2:
            return np.zeros(total_count, dtype=np.int32)

        if total_count >= 6:
            min_cluster_size = max(
                2,
                min(
                    int(self.settings.position_subcluster_hdbscan_max_cluster_size),
                    int(
                        round(
                            float(np.sqrt(total_count))
                            * float(self.settings.position_subcluster_hdbscan_scale)
                        )
                    ),
                ),
            )
            min_samples = max(1, min(4, int(round(min_cluster_size * 0.5))))
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric="euclidean",
                    prediction_data=False,
                )
                labels = clusterer.fit_predict(matrix)
                unique_clusters = {int(label) for label in labels.tolist() if int(label) >= 0}
                if len(unique_clusters) >= 2:
                    return labels.astype(np.int32)
            except Exception as exc:
                self.logger.warning("Subclustering (HDBSCAN) fallback to kmeans: %s", exc)

        best_labels: np.ndarray | None = None
        best_score = -1.0
        configured_max_k = int(self.settings.position_subcluster_max_k)
        if configured_max_k <= 0:
            adaptive_max_k = max(8, int(round(float(np.sqrt(total_count)) * 2.5)))
            max_k = min(adaptive_max_k, total_count - 1)
        else:
            max_k = min(max(2, configured_max_k), total_count - 1)
        for k in range(2, max_k + 1):
            try:
                model = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = model.fit_predict(matrix).astype(np.int32)
            except Exception:
                continue
            unique = set(labels.tolist())
            if len(unique) < 2:
                continue
            try:
                score = float(silhouette_score(matrix, labels, metric="euclidean"))
            except Exception:
                score = -1.0
            if score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is not None:
            return best_labels
        return np.zeros(total_count, dtype=np.int32)

    # ------------------------------------------------------------------
    # Centroid and prototype helpers
    # ------------------------------------------------------------------

    def compute_group_centroid(self, members: list[int], vectors: list[list[float]]) -> list[float]:
        """Compute the mean embedding vector for a group of comments.

        Parameters
        ----------
        members:
            Global indices into *vectors*.
        vectors:
            Full list of embedding vectors.

        Returns
        -------
        list[float]
            The centroid vector.
        """
        if not members:
            return []
        matrix = np.array([vectors[idx] for idx in members], dtype=np.float32)
        return matrix.mean(axis=0).tolist()

    def build_position_prototype(
        self,
        members: list[int],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
    ) -> str:
        """Select representative quotes closest to the group centroid.

        Parameters
        ----------
        members:
            Global comment indices forming the position.
        comments:
            Full list of processed comments.
        vectors:
            Embedding vectors aligned with *comments*.

        Returns
        -------
        str
            Space-joined representative quotes (up to 3).
        """
        if not members:
            return ""
        centroid = self.compute_group_centroid(members, vectors)
        ranked = sorted(
            members,
            key=lambda idx: _cosine_similarity(vectors[idx], centroid),
            reverse=True,
        )
        lines: list[str] = []
        for idx in ranked:
            text = comment_text_for_output(comments[idx])
            if not text:
                continue
            lines.append(text)
            if len(lines) >= 3:
                break
        return " ".join(lines)

    # ------------------------------------------------------------------
    # LLM naming of individual position
    # ------------------------------------------------------------------

    def name_position_with_llm(
        self,
        *,
        llm_provider: LLMProvider,
        members: list[int],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        cluster_title: str,
        fallback_title: str,
        fallback_summary: str,
        fallback_markers: list[str],
        llm_disabled: bool,
        request_llm_json: RequestLLMJsonFn,
        rank_cluster_member_indices: RankClusterMemberIndicesFn,
        diagnostics: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool, bool]:
        """Name a single position using the LLM.

        Parameters
        ----------
        llm_provider:
            LLM provider instance.
        members:
            Global indices of comments in this position group.
        comments:
            Full list of processed comments.
        vectors:
            Embedding vectors aligned with *comments*.
        cluster_title:
            Parent cluster title for context.
        fallback_title:
            Title to use when LLM fails.
        fallback_summary:
            Summary to use when LLM fails.
        fallback_markers:
            Markers to use when LLM fails.
        llm_disabled:
            Whether LLM calls are currently disabled.
        request_llm_json:
            Callable that sends a prompt to the LLM and returns parsed JSON.
        rank_cluster_member_indices:
            Callable that ranks member indices by relevance within a cluster.
        diagnostics:
            Optional mutable dict for recording diagnostic counters.

        Returns
        -------
        tuple[dict[str, Any], bool, bool]
            A dict with ``title``, ``summary``, ``markers``; the updated
            ``llm_disabled`` flag; and a boolean indicating whether the
            fallback title was used.
        """
        error_payload = {
            "title": "",
            "summary": fallback_summary,
            "markers": fallback_markers[:5],
        }
        if llm_disabled:
            return error_payload, llm_disabled, True

        sample_comments = self.build_position_naming_sample(
            llm_provider=llm_provider,
            members=members,
            comments=comments,
            vectors=vectors,
            rank_cluster_member_indices=rank_cluster_member_indices,
        )
        if not sample_comments:
            return error_payload, llm_disabled, True
        comment_char_cap = max(40, int(self.settings.position_llm_comment_char_cap))
        payload_char_budget = max(800, int(self.settings.position_llm_payload_char_budget))
        used_chars = 0
        payload: list[dict[str, Any]] = []
        for idx, comment in enumerate(sample_comments):
            text = comment_text_for_output(comment)
            if not text:
                continue
            text = clip_text_for_llm(text, max_chars=comment_char_cap)
            remaining_chars = payload_char_budget - used_chars
            if remaining_chars <= 0:
                break
            if len(text) > remaining_chars:
                text = clip_text_for_llm(text, max_chars=remaining_chars)
            if not text:
                continue
            payload.append(
                {
                    "id": f"s{idx + 1:04d}",
                    "weight": round(float(comment.weight), 3),
                    "text": text,
                }
            )
            used_chars += len(text)
        if not payload:
            return error_payload, llm_disabled, True

        retries_left = min(1, max(0, int(self.settings.position_title_retry_count)))
        retry_reason: str | None = None
        for attempt in range(retries_left + 1):
            retry_note = f"Retry reason: {retry_reason}\n" if retry_reason else ""
            prompt = (
                "You name one position/narrative inside a cluster of YouTube comments.\n"
                "Return strict JSON only in Russian.\n"
                "JSON schema:\n"
                "{\n"
                '  "title": "3-10 words title",\n'
                '  "summary": "one sentence",\n'
                '  "markers": ["marker 1", "marker 2"]\n'
                "}\n"
                "Rules:\n"
                "- title must be 3-10 words;\n"
                "- title must express exactly one claim/stance;\n"
                "- no slash '/', no semicolon ';', avoid multi-part title with colon;\n"
                "- no generic words (video, comments, discussion, misc);\n"
                "- markers must be 2-5 short arguments.\n"
                f"Cluster title context: {cluster_title}\n"
                f"Fallback title: {fallback_title}\n"
                f"{retry_note}"
                f"Comments sample ({len(payload)}):\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            )
            try:
                data = request_llm_json(
                    llm_provider,
                    prompt,
                    task="position_naming",
                    estimated_out_tokens=220,
                    max_output_tokens=min(self.settings.openai_max_output_tokens, 420),
                )
            except (BudgetExceededError, ExternalServiceError) as exc:
                message = str(exc).lower()
                if (
                    isinstance(exc, BudgetExceededError)
                    or "429" in message
                    or "rate-limit" in message
                    or "rate limit" in message
                ):
                    llm_disabled = True
                    if diagnostics is not None:
                        _mark_llm_disabled(
                            diagnostics,
                            reason=(
                                "budget_exceeded"
                                if isinstance(exc, BudgetExceededError)
                                else "rate_limited"
                            ),
                        )
                elif diagnostics is not None:
                    _mark_llm_disabled(diagnostics, reason="external_service_error")
                if diagnostics is not None:
                    diagnostics["llm_position_fail_count"] = (
                        diagnostics.get("llm_position_fail_count", 0) + 1
                    )
                self.logger.warning(
                    "Position naming failed via %s: %s", llm_provider.provider_name, exc
                )
                return error_payload, llm_disabled, True

            title_candidate = (
                sanitize_cluster_title(str(data.get("title", "") or ""), sample_comments)
                if data
                else ""
            )
            if not title_candidate:
                if attempt < retries_left:
                    retry_reason = "title violated naming constraints"
                    continue
                return error_payload, llm_disabled, True
            if not position_title_single_claim_passed(title_candidate):
                if attempt < retries_left:
                    retry_reason = "title includes more than one claim"
                    continue
                return error_payload, llm_disabled, True

            summary = " ".join(str(data.get("summary", "") or "").split()).strip() if data else ""
            if not summary:
                summary = fallback_summary
            raw_markers = data.get("markers") if isinstance(data, dict) else []
            markers: list[str] = []
            if isinstance(raw_markers, list):
                for marker in raw_markers:
                    cleaned = " ".join(str(marker or "").split()).strip(" -:;,.")
                    if not cleaned:
                        continue
                    if cleaned.lower() in {item.lower() for item in markers}:
                        continue
                    markers.append(cleaned)
                    if len(markers) >= 5:
                        break
            if len(markers) < 2:
                markers = fallback_markers[:5]
            return (
                {
                    "title": title_candidate,
                    "summary": summary,
                    "markers": markers[:5],
                },
                llm_disabled,
                False,
            )
        return error_payload, llm_disabled, True

    # ------------------------------------------------------------------
    # Naming sample selection
    # ------------------------------------------------------------------

    def build_position_naming_sample(
        self,
        *,
        llm_provider: LLMProvider,
        members: list[int],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
        rank_cluster_member_indices: RankClusterMemberIndicesFn,
    ) -> list[ProcessedComment]:
        """Select a representative sample of comments for LLM position naming.

        Parameters
        ----------
        llm_provider:
            LLM provider instance (used to choose sample-size heuristics).
        members:
            Global indices of comments in the position group.
        comments:
            Full list of processed comments.
        vectors:
            Embedding vectors aligned with *comments*.
        rank_cluster_member_indices:
            Callable that ranks member indices by relevance within a cluster.

        Returns
        -------
        list[ProcessedComment]
            The selected sample comments.
        """
        if not members:
            return []
        local_cluster = ClusterDraft(
            cluster_key="position",
            member_indices=list(members),
            representative_indices=list(members[: min(4, len(members))]),
            centroid=self.compute_group_centroid(members, vectors),
            size_count=len(members),
            share_pct=100.0,
            weighted_share=100.0,
            is_emerging=False,
        )
        ranked = rank_cluster_member_indices(local_cluster, comments, vectors)
        if not ranked:
            ranked = list(members)
        if isinstance(llm_provider, OpenAIChatProvider):
            min_target = max(1, int(self.settings.position_llm_sample_min_openai))
            max_target = max(min_target, int(self.settings.position_llm_sample_max_openai))
            adaptive_target = int(round(float(np.sqrt(len(members))) * 3.0))
            target = max(min_target, min(max_target, adaptive_target))
        else:
            target = min(
                max(40, int(self.settings.position_llm_sample_max)),
                max(
                    int(self.settings.position_llm_sample_min),
                    int(round(float(np.sqrt(len(members))) * 8)),
                ),
            )
        if len(ranked) <= target:
            return [comments[idx] for idx in ranked if 0 <= idx < len(comments)]
        step = max(1, len(ranked) // target)
        selected = [ranked[pos] for pos in range(0, len(ranked), step)][:target]
        return [comments[idx] for idx in selected if 0 <= idx < len(comments)]

    # ------------------------------------------------------------------
    # Fallback title generation
    # ------------------------------------------------------------------

    def fallback_position_title(
        self,
        comments: list[ProcessedComment],
        *,
        cluster_title: str,
    ) -> str:
        """Generate a fallback position title from comment keywords.

        Parameters
        ----------
        comments:
            Comments belonging to the position group.
        cluster_title:
            Parent cluster title used as a last resort.

        Returns
        -------
        str
            A cleaned position title string.
        """
        texts = [comment_text_for_output(comment) for comment in comments]
        keywords = extract_salient_comment_keywords(texts, max_keywords=4)
        if len(keywords) >= 2:
            candidate = f"Мнение аудитории о {keywords[0]} и {keywords[1]}"
        elif keywords:
            candidate = f"Позиция аудитории по вопросу {keywords[0]}"
        else:
            candidate = f"Позиция аудитории по теме {cluster_title}"
        sanitized = sanitize_cluster_title(candidate, comments)
        if sanitized:
            return sanitized
        sanitized_cluster = sanitize_cluster_title(cluster_title, comments)
        if sanitized_cluster:
            return sanitized_cluster
        return "Позиция аудитории по теме"

    def fallback_single_claim_position_title(
        self,
        comments: list[ProcessedComment],
        *,
        cluster_title: str,
    ) -> str:
        """Generate a fallback title guaranteed to pass the single-claim check.

        Parameters
        ----------
        comments:
            Comments belonging to the position group.
        cluster_title:
            Parent cluster title used as a last resort.

        Returns
        -------
        str
            A single-claim position title.
        """
        title = self.fallback_position_title(comments, cluster_title=cluster_title)
        if position_title_single_claim_passed(title):
            return title
        candidate = sanitize_cluster_title("Позиция аудитории по теме", comments)
        if candidate and position_title_single_claim_passed(candidate):
            return candidate
        return "Позиция аудитории по теме"

    # ------------------------------------------------------------------
    # Coherence estimation
    # ------------------------------------------------------------------

    def estimate_position_coherence(
        self,
        *,
        members: list[int],
        comments: list[ProcessedComment],
        vectors: list[list[float]],
    ) -> float:
        """Estimate the coherence of a position group via weighted cosine similarity.

        Parameters
        ----------
        members:
            Global indices of comments in the position group.
        comments:
            Full list of processed comments.
        vectors:
            Embedding vectors aligned with *comments*.

        Returns
        -------
        float
            Coherence score in ``[0.0, 1.0]``.
        """
        if not members:
            return 0.0
        centroid = self.compute_group_centroid(members, vectors)
        if not centroid:
            return 0.0
        score = 0.0
        weight_sum = 0.0
        for idx in members:
            sim = _cosine_similarity(vectors[idx], centroid)
            weight = max(0.1, float(comments[idx].weight))
            score += sim * weight
            weight_sum += weight
        if weight_sum <= 0:
            return 0.0
        return max(0.0, min(1.0, float(score / weight_sum)))
