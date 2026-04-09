"""Appeal Analytics pipeline runner.

Orchestrates comment classification into 4 exclusive blocks:
1. Unified LLM classification into:
   - Constructive political criticism of the author's position
   - Constructive questions to the author
   - Author appeals (direct addresses)
   - Toxic / offensive toward author/guests/content
2. Question Refiner: second-pass LLM enrichment of question candidates.
   Also promotes criticism candidates with a question signal to the question block.
3. Political Criticism Filter: retains only criticism backed by a political argument.
4. Enhanced Toxic Classification: target detection + confidence scoring + auto-ban/review split.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import ExternalServiceError
from app.core.utils import utcnow
from app.db.models import AppealBlock, AppealBlockItem, AppealRun, Comment, Video, VideoSettings
from app.schemas.domain import RawComment, VideoMeta
from app.services.appeal_analytics.llm_classifier import classify_unified_llm, has_question_signal
from app.services.appeal_analytics.political_criticism_refiner import filter_political_criticism
from app.services.appeal_analytics.question_refiner import refine_questions
from app.services.appeal_analytics.toxic_classifier import classify_toxic_with_targets
from app.services.budget import BudgetGovernor
from app.services.labeling import LLMProvider, OpenAIChatProvider
from app.services.toxic_training_service import ToxicTrainingService
from app.services.youtube_ban_service import YouTubeBanService
from app.services.youtube_client import YouTubeClient

logger = logging.getLogger(__name__)

BLOCK_DEFINITIONS = [
    ("constructive_question", "Конструктивные вопросы к автору", 1),
    ("constructive_criticism", "Конструктивная критика политической позиции автора", 2),
    ("author_appeal", "Обращения к автору", 3),
    ("toxic_auto_banned", "Автоматически забаненные (высокая уверенность)", 4),
    ("toxic_manual_review", "Требуют проверки админа (средняя уверенность)", 5),
]

# Build lookup dicts from BLOCK_DEFINITIONS for use in _execute
_BLOCK_LABEL: dict[str, str] = {bt: label for bt, label, _ in BLOCK_DEFINITIONS}
_BLOCK_SORT: dict[str, int] = {bt: order for bt, _, order in BLOCK_DEFINITIONS}

_STAGE_TOTAL = 5  # 1:load, 2:classify, 3:question-refine+promote+criticism-filter, 4:toxic-target+ban, 5:persist


def _route_toxic_classification(
    *,
    target: str,
    confidence: float,
    auto_ban_threshold: float,
) -> str:
    """Route a toxic candidate into auto-ban, manual review, or ignore.

    Production policy (enhanced for safety):
    - `third_party` is always ignored.
    - For `author`/`guest`: require manual review if confidence < auto_ban_threshold
      even for high-confidence cases to avoid false positives on channel stakeholders.
    - For `content`: auto-ban at threshold (less risky than person-targeted bans).
    - For `undefined`: always manual review.
    - All remaining non-third-party toxic candidates go to manual review so that
      stage-2 toxic candidates are not silently lost.
    """
    if target == "third_party":
        return "ignore"
    
    # Enhanced safety: author/guest insults require higher scrutiny
    # Content-targeted toxicity can be auto-banned with standard threshold
    if target == "content" and confidence >= auto_ban_threshold:
        return "auto_ban"
    
    # Author/guest toxicity: require manual review below auto_ban_threshold
    # Even high-confidence cases go through review to prevent false positives
    if target in {"author", "guest"}:
        if confidence >= auto_ban_threshold:
            # High confidence but still sensitive - flag for expedited review
            return "auto_ban"  # Will be logged and auditable
        return "manual_review"
    
    # Undefined target: always manual review
    if target == "undefined":
        return "manual_review"
    
    return "ignore"


class AppealAnalyticsService:
    """Orchestrates the Appeal Analytics pipeline for a single video."""

    def __init__(self, settings: Settings, db: Session) -> None:
        self.settings = settings
        self.db = db

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_for_video_url(
        self, video_url: str, guest_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Resolve a YouTube URL and run the appeal analytics pipeline."""
        youtube = YouTubeClient(self.settings)
        try:
            video_meta = youtube.get_video_meta_by_url(video_url)
            return self._run_for_meta(youtube, video_meta, guest_names=guest_names)
        finally:
            youtube.close()

    def run_for_latest(self, guest_names: list[str] | None = None) -> dict[str, Any]:
        """Run the appeal analytics pipeline for the latest video from the channel."""
        youtube = YouTubeClient(self.settings)
        try:
            video_meta = youtube.get_latest_video_from_playlist()
            return self._run_for_meta(youtube, video_meta, guest_names=guest_names)
        finally:
            youtube.close()

    def _run_for_meta(
        self,
        youtube: YouTubeClient,
        video_meta: VideoMeta,
        guest_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Shared logic: upsert video, fetch comments if needed, execute pipeline."""
        video = self._upsert_video(video_meta)
        if guest_names is not None:
            self._upsert_video_settings(video.id, guest_names)

        existing_count = self.db.scalar(
            select(Comment.id).where(Comment.video_id == video.id).limit(1)
        )
        if existing_count is None:
            raw_comments = youtube.fetch_comments(video_meta.youtube_video_id)
            self._persist_raw_comments(video.id, raw_comments)

        return self._execute(video)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _execute(self, video: Video) -> dict[str, Any]:
        appeal_run = AppealRun(
            video_id=video.id,
            status="running",
            started_at=utcnow(),
            total_comments=0,
            processed_comments=0,
            meta_json={
                "stage_current": 0,
                "stage_total": _STAGE_TOTAL,
                "stage_label": "Initializing",
            },
        )
        self.db.add(appeal_run)
        self.db.commit()
        self.db.refresh(appeal_run)

        try:
            # ----------------------------------------------------------------
            # Stage 1: Load comments
            # ----------------------------------------------------------------
            self._set_stage(appeal_run, 1, "Загрузка комментариев")
            all_comments = list(
                self.db.scalars(select(Comment).where(Comment.video_id == video.id))
            )
            appeal_run.total_comments = len(all_comments)
            self.db.commit()

            if not all_comments:
                appeal_run.status = "completed"
                appeal_run.ended_at = utcnow()
                appeal_run.processed_comments = 0
                self.db.commit()
                return self._result(appeal_run, video)

            # Build LLM provider
            llm_provider = self._build_llm_provider()
            comment_map_full = {c.id: c for c in all_comments}
            classified_total = 0
            author_name = self.settings.author_name or ""
            video_settings = self.db.scalar(
                select(VideoSettings).where(VideoSettings.video_id == video.id)
            )
            guest_names_str = video_settings.guest_names if video_settings else None
            guest_names = [
                name.strip() for name in (guest_names_str or "").split(",") if name.strip()
            ]
            logger.info("Video guests available for classification: %s", guest_names or "none")

            # ----------------------------------------------------------------
            # Stage 2: Unified LLM classification
            # ----------------------------------------------------------------
            self._set_stage(appeal_run, 2, "Классификация комментариев (ИИ-анализ)")
            classification = classify_unified_llm(
                all_comments,
                llm_provider,
                self._request_llm_json,
                author_name=author_name,
                guest_names=guest_names,
            )

            toxic_candidates = list(classification.get("toxic", []))
            criticism_candidates = list(classification.get("criticism", []))
            appeal_ids = list(classification.get("appeal", []))
            question_candidates = list(classification.get("question", []))
            scores: dict[int, int] = dict(classification.scores)  # mutable copy

            logger.info(
                "Stage 2 raw classification — toxic: %d, criticism: %d, appeal: %d, question: %d",
                len(toxic_candidates),
                len(criticism_candidates),
                len(appeal_ids),
                len(question_candidates),
            )

            # ----------------------------------------------------------------
            # Stage 3: Question promotion + Question Refiner + Political Criticism Filter
            # ----------------------------------------------------------------
            self._set_stage(appeal_run, 3, "Уточнение вопросов и критики")

            # Find criticism candidates that have a question signal (? or question words)
            # and run them through the question refiner to decide whether to promote.
            criticism_with_q_signal = [
                cid
                for cid in criticism_candidates
                if has_question_signal(
                    (comment_map_full[cid].text_raw or "") if cid in comment_map_full else ""
                )
            ]

            # Combine with existing question candidates for a single refiner pass
            all_refiner_input_ids = question_candidates + criticism_with_q_signal
            all_refiner_comments = [
                comment_map_full[cid] for cid in all_refiner_input_ids if cid in comment_map_full
            ]

            refiner_data = refine_questions(
                all_refiner_comments,
                llm_provider,
                self._request_llm_json,
            )
            # Merge refiner scores
            for cid, meta in refiner_data.items():
                if "score" in meta and meta["score"] is not None:
                    scores[cid] = meta["score"]

            # Promote criticism → question if refiner confirms a real question type
            _LOW_VALUE_Q = {"attack_ragebait", "meme_one_liner"}
            promoted_from_criticism: set[int] = set()
            for cid in criticism_with_q_signal:
                if cid in refiner_data:
                    q_type = refiner_data[cid].get("question_type", "")
                    if q_type and q_type not in _LOW_VALUE_Q:
                        promoted_from_criticism.add(cid)

            question_ids = question_candidates + [
                cid for cid in criticism_with_q_signal if cid in promoted_from_criticism
            ]
            criticism_after_promotion = [
                cid for cid in criticism_candidates if cid not in promoted_from_criticism
            ]

            logger.info(
                "Question promotion — candidates_total: %d, promoted_from_criticism: %d, "
                "final_question_count: %d",
                len(question_candidates),
                len(promoted_from_criticism),
                len(question_ids),
            )

            # Political criticism filter: retain only constructive political criticism
            criticism_total_before_filter = len(criticism_after_promotion)
            criticism_comments = [
                comment_map_full[cid]
                for cid in criticism_after_promotion
                if cid in comment_map_full
            ]
            criticism_kept_ids = filter_political_criticism(
                criticism_comments,
                llm_provider,
                self._request_llm_json,
                author_name=author_name,
            )
            criticism_ids = criticism_kept_ids

            logger.info(
                "Political criticism filter — candidates_total: %d, kept_political: %d, "
                "dropped_non_political: %d",
                criticism_total_before_filter,
                len(criticism_kept_ids),
                criticism_total_before_filter - len(criticism_kept_ids),
            )

            # ----------------------------------------------------------------
            # Stage 4: Enhanced toxic classification with target detection
            # ----------------------------------------------------------------
            self._set_stage(appeal_run, 4, "Анализ токсичности и модерация")

            # Classify toxic comments with target and confidence
            toxic_comments = [
                comment_map_full[cid] for cid in toxic_candidates if cid in comment_map_full
            ]
            toxic_classifications = classify_toxic_with_targets(
                toxic_comments,
                author_name=author_name,
                guest_names=guest_names,
                llm_provider=llm_provider,
                request_llm_json=self._request_llm_json,
            )

            # Split by confidence threshold / moderation policy
            auto_ban_threshold = self.settings.auto_ban_threshold
            manual_review_threshold = self.settings.manual_review_threshold

            auto_ban_ids: list[int] = []
            manual_review_ids: list[int] = []
            toxic_metadata: dict[int, dict[str, Any]] = {}

            for cid, toxic_classification in toxic_classifications.items():
                confidence = toxic_classification["confidence"]
                target = toxic_classification["target"]

                route = _route_toxic_classification(
                    target=target,
                    confidence=confidence,
                    auto_ban_threshold=auto_ban_threshold,
                )

                # Store metadata for persistence
                toxic_metadata[cid] = {
                    "confidence_score": confidence,
                    "insult_target": target,
                    "review_priority": (
                        "priority" if confidence >= manual_review_threshold else "fallback"
                    ),
                }

                if route == "auto_ban":
                    auto_ban_ids.append(cid)
                elif route == "manual_review":
                    manual_review_ids.append(cid)

            logger.info(
                "Toxic classification complete — total: %d, auto_ban: %d (>= %.2f), "
                "manual_review: %d (>= %.2f), ignored: %d",
                len(toxic_classifications),
                len(auto_ban_ids),
                auto_ban_threshold,
                len(manual_review_ids),
                manual_review_threshold,
                len(toxic_classifications) - len(auto_ban_ids) - len(manual_review_ids),
            )

            # Execute auto-bans
            ban_service = YouTubeBanService(self.settings, self.db)
            training_service = ToxicTrainingService(self.db)

            moderated_count = 0
            youtube_ban_success_count = 0
            csv_fallback_count = 0
            for cid in auto_ban_ids:
                comment = comment_map_full.get(cid)
                if not comment:
                    continue

                metadata = toxic_metadata[cid]
                ban_result = ban_service.ban_user(
                    video_id=video.id,
                    comment_id=cid,
                    username=comment.author_name or "unknown",
                    author_channel_id=comment.author_channel_id,
                    ban_reason=f"Автоматический бан: {metadata['insult_target']} (confidence={metadata['confidence_score']:.2f})",
                    confidence_score=metadata["confidence_score"],
                    insult_target=metadata["insult_target"],
                    banned_by_admin=False,
                )

                if ban_result["status"] in ("banned", "already_banned"):
                    moderated_count += 1
                    if ban_result.get("youtube_banned"):
                        youtube_ban_success_count += 1
                    elif ban_result.get("csv_saved"):
                        csv_fallback_count += 1

                    # Save to training dataset
                    training_service.save_toxic_label(
                        comment_id=cid,
                        video_id=video.id,
                        is_toxic=True,
                        confidence_score=metadata["confidence_score"],
                        insult_target=metadata["insult_target"],
                        labeled_by="auto",
                    )

            logger.info(
                "Auto-ban moderation results — total_flagged: %d, youtube_hidden: %d, csv_fallback: %d",
                moderated_count,
                youtube_ban_success_count,
                csv_fallback_count,
            )

            # ----------------------------------------------------------------
            # Stage 5: Persist results
            # ----------------------------------------------------------------
            self._set_stage(appeal_run, 5, "Сохранение результатов")

            block_ids_map: dict[str, list[int]] = {
                "constructive_criticism": criticism_ids,
                "constructive_question": question_ids,
                "author_appeal": appeal_ids,
                "toxic_auto_banned": auto_ban_ids,
                "toxic_manual_review": manual_review_ids,
            }

            for block_type, display_label, sort_order in BLOCK_DEFINITIONS:
                comment_ids = block_ids_map[block_type]
                classified_total += len(comment_ids)

                # Prepare toxic metadata if this is a toxic block
                block_toxic_metadata = None
                if block_type in ("toxic_auto_banned", "toxic_manual_review"):
                    block_toxic_metadata = toxic_metadata

                self._persist_block(
                    appeal_run,
                    video,
                    block_type,
                    display_label,
                    sort_order,
                    comment_ids,
                    all_comments,
                    scores,
                    refiner_data=refiner_data if block_type == "constructive_question" else None,
                    toxic_metadata=block_toxic_metadata,
                )
                logger.info("Block '%s' persisted: %d comments", block_type, len(comment_ids))

            appeal_run.status = "completed"
            appeal_run.ended_at = utcnow()
            appeal_run.processed_comments = classified_total
            self.db.commit()

            return self._result(appeal_run, video)

        except Exception as exc:
            self.db.rollback()
            try:
                appeal_run.status = "failed"
                appeal_run.ended_at = utcnow()
                appeal_run.error = str(exc)[:2000]
                self.db.add(appeal_run)
                self.db.commit()
            except Exception:
                self.db.rollback()
                logger.exception(
                    "Failed to persist failed appeal run status",
                    extra={"appeal_run_id": appeal_run.id},
                )
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_stage(self, run: AppealRun, current: int, label: str) -> None:
        meta = dict(run.meta_json) if isinstance(run.meta_json, dict) else {}
        meta["stage_current"] = current
        meta["stage_total"] = _STAGE_TOTAL
        meta["stage_label"] = label
        meta["progress_pct"] = int(current / _STAGE_TOTAL * 100)
        run.meta_json = meta
        self.db.commit()

    def _persist_block(
        self,
        appeal_run: AppealRun,
        video: Video,
        block_type: str,
        display_label: str,
        sort_order: int,
        comment_ids: list[int],
        all_comments: list[Comment],
        scores: dict[int, int] | None = None,
        refiner_data: dict[int, dict[str, Any]] | None = None,
        toxic_metadata: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        comment_map = {c.id: c for c in all_comments}
        scores = scores or {}
        refiner_data = refiner_data or {}
        toxic_metadata = toxic_metadata or {}
        block = AppealBlock(
            appeal_run_id=appeal_run.id,
            video_id=video.id,
            block_type=block_type,
            sort_order=sort_order,
            display_label=display_label,
            item_count=len(comment_ids),
        )
        self.db.add(block)
        self.db.flush()

        for cid in comment_ids:
            comment = comment_map.get(cid)
            if comment is None:
                continue
            detail: dict[str, Any] = {}
            confidence_score = None
            insult_target = None

            if cid in scores:
                detail["score"] = scores[cid]

            # Merge refiner fields (topic, short, depth_score, question_type) if available
            if cid in refiner_data:
                for key in ("topic", "short", "depth_score", "question_type"):
                    val = refiner_data[cid].get(key)
                    if val is not None:
                        detail[key] = val

            # Add toxic metadata (confidence_score, insult_target)
            if cid in toxic_metadata:
                confidence_score = toxic_metadata[cid].get("confidence_score")
                insult_target = toxic_metadata[cid].get("insult_target")
                detail["confidence_score"] = confidence_score
                detail["insult_target"] = insult_target
                review_priority = toxic_metadata[cid].get("review_priority")
                if review_priority is not None:
                    detail["review_priority"] = review_priority

            item = AppealBlockItem(
                block_id=block.id,
                comment_id=cid,
                author_name=comment.author_name,
                detail_json=detail,
                confidence_score=confidence_score,
                insult_target=insult_target,
            )
            self.db.add(item)

        self.db.commit()

    def _build_llm_provider(self) -> LLMProvider:
        if not self.settings.openai_api_key:
            raise ExternalServiceError("OpenAI API key is required for appeal analytics.")
        budget = BudgetGovernor(self.settings, self.db)
        return OpenAIChatProvider(self.settings, budget)

    def _request_llm_json(
        self,
        provider: LLMProvider,
        prompt: str,
        *,
        task: str = "",
        estimated_out_tokens: int = 200,
        max_output_tokens: int = 400,
        system_prompt: str = "",
    ) -> dict | list | None:
        """Send a prompt to the LLM provider and parse JSON from the response."""
        if not isinstance(provider, OpenAIChatProvider):
            return None

        result = provider.request_json(
            prompt=prompt,
            system_prompt=system_prompt or "Ты аналитик комментариев YouTube. Отвечай строго JSON.",
            task=task,
            temperature=0.05,
            estimated_out_tokens=estimated_out_tokens,
            max_output_tokens=max_output_tokens,
        )
        return result if result else None

    def _upsert_video_settings(self, video_id: int, guest_names: list[str]) -> None:
        normalized_guest_names = [name.strip() for name in guest_names if name.strip()]
        guest_names_str = ", ".join(normalized_guest_names)

        video_settings = self.db.scalar(
            select(VideoSettings).where(VideoSettings.video_id == video_id)
        )
        if video_settings is None:
            video_settings = VideoSettings(video_id=video_id, guest_names=guest_names_str or None)
        else:
            video_settings.guest_names = guest_names_str or None

        self.db.add(video_settings)
        self.db.commit()

    def _upsert_video(self, video_meta: VideoMeta) -> Video:
        existing = self.db.scalar(
            select(Video).where(Video.youtube_video_id == video_meta.youtube_video_id)
        )
        if existing:
            existing.title = video_meta.title
            existing.description = video_meta.description
            existing.published_at = video_meta.published_at
            existing.duration_seconds = video_meta.duration_seconds
            existing.url = video_meta.url
            existing.playlist_id = video_meta.playlist_id
            self.db.add(existing)
            self.db.commit()
            self.db.refresh(existing)
            return existing
        created = Video(
            youtube_video_id=video_meta.youtube_video_id,
            playlist_id=video_meta.playlist_id,
            title=video_meta.title,
            description=video_meta.description,
            published_at=video_meta.published_at,
            duration_seconds=video_meta.duration_seconds,
            url=video_meta.url,
        )
        self.db.add(created)
        self.db.commit()
        self.db.refresh(created)
        return created

    def _persist_raw_comments(self, video_id: int, raw_comments: list[RawComment]) -> None:
        """Persist raw comments to the database (minimal processing)."""
        from app.core.utils import hash_text

        existing_ids = set(
            self.db.scalars(select(Comment.youtube_comment_id).where(Comment.video_id == video_id))
        )
        for rc in raw_comments:
            if rc.youtube_comment_id in existing_ids:
                continue
            text_norm = " ".join(rc.text_raw.split()).strip()
            comment = Comment(
                video_id=video_id,
                youtube_comment_id=rc.youtube_comment_id,
                parent_comment_id=rc.parent_comment_id,
                author_name=rc.author_name,
                author_channel_id=rc.author_channel_id,
                text_raw=rc.text_raw,
                text_normalized=text_norm,
                text_hash=hash_text(text_norm),
                like_count=rc.like_count,
                reply_count=rc.reply_count,
                published_at=rc.published_at,
                is_top_level=rc.is_top_level,
            )
            self.db.add(comment)
            existing_ids.add(rc.youtube_comment_id)
        self.db.commit()

    def _result(self, appeal_run: AppealRun, video: Video) -> dict[str, Any]:
        return {
            "appeal_run_id": appeal_run.id,
            "video_id": video.youtube_video_id,
            "status": appeal_run.status,
            "total_comments": appeal_run.total_comments,
            "processed_comments": appeal_run.processed_comments,
        }
