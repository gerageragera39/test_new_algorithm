"""Service for saving toxic training data for future model fine-tuning.

Records labeled toxic/non-toxic comments to build a training dataset
that can later be used to train a custom toxic comment classifier.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.core.utils import utcnow
from app.db.models import Comment, ToxicTrainingData

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ToxicTrainingService:
    """Service for recording toxic comment labels for model training."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def save_toxic_label(
        self,
        comment_id: int,
        video_id: int,
        is_toxic: bool,
        confidence_score: float,
        insult_target: str | None,
        labeled_by: str,  # 'auto' or 'admin'
    ) -> ToxicTrainingData:
        """Save a labeled toxic/non-toxic comment for training.

        Args:
            comment_id: Internal comment ID
            video_id: Internal video ID
            is_toxic: True if toxic, False if not
            confidence_score: LLM confidence (0.0-1.0)
            insult_target: Target of insult (author/guest/content/undefined/third_party)
            labeled_by: 'auto' for automatic, 'admin' for manual review

        Returns:
            ToxicTrainingData record
        """
        # Get comment text and author
        comment = self.db.get(Comment, comment_id)
        if not comment:
            raise ValueError(f"Comment {comment_id} not found")

        # Create training record
        training_data = ToxicTrainingData(
            comment_id=comment_id,
            video_id=video_id,
            text=comment.text_raw or "",
            author_name=comment.author_name,
            is_toxic=is_toxic,
            confidence_score=confidence_score,
            insult_target=insult_target,
            labeled_by=labeled_by,
            labeled_at=utcnow(),
        )
        self.db.add(training_data)
        self.db.commit()
        self.db.refresh(training_data)

        logger.debug(
            "Saved training data: comment_id=%d, is_toxic=%s, target=%s, labeled_by=%s",
            comment_id,
            is_toxic,
            insult_target or "unknown",
            labeled_by,
        )

        return training_data

    def save_batch_toxic_labels(
        self,
        labels: list[dict],
    ) -> list[ToxicTrainingData]:
        """Save multiple toxic labels in batch.

        Args:
            labels: List of dicts with keys: comment_id, video_id, is_toxic,
                    confidence_score, insult_target, labeled_by

        Returns:
            List of created ToxicTrainingData records
        """
        records = []
        for label_data in labels:
            try:
                record = self.save_toxic_label(
                    comment_id=label_data["comment_id"],
                    video_id=label_data["video_id"],
                    is_toxic=label_data["is_toxic"],
                    confidence_score=label_data["confidence_score"],
                    insult_target=label_data.get("insult_target"),
                    labeled_by=label_data["labeled_by"],
                )
                records.append(record)
            except Exception as exc:
                logger.error(
                    "Failed to save training label for comment %s: %s",
                    label_data.get("comment_id"),
                    exc,
                )

        logger.info("Saved %d toxic training labels", len(records))
        return records

    def get_training_stats(self) -> dict:
        """Get statistics about the training dataset."""
        from sqlalchemy import func, select

        # Total count
        total = self.db.scalar(select(func.count(ToxicTrainingData.id))) or 0

        # Toxic vs non-toxic
        toxic_count = (
            self.db.scalar(
                select(func.count(ToxicTrainingData.id)).where(ToxicTrainingData.is_toxic)
            )
            or 0
        )

        # By source
        auto_count = (
            self.db.scalar(
                select(func.count(ToxicTrainingData.id)).where(
                    ToxicTrainingData.labeled_by == "auto"
                )
            )
            or 0
        )

        admin_count = (
            self.db.scalar(
                select(func.count(ToxicTrainingData.id)).where(
                    ToxicTrainingData.labeled_by == "admin"
                )
            )
            or 0
        )

        # By target
        target_counts = {}
        targets = ["author", "guest", "content", "undefined", "third_party"]
        for target in targets:
            count = (
                self.db.scalar(
                    select(func.count(ToxicTrainingData.id)).where(
                        ToxicTrainingData.insult_target == target
                    )
                )
                or 0
            )
            if count > 0:
                target_counts[target] = count

        return {
            "total": total,
            "toxic": toxic_count,
            "non_toxic": total - toxic_count,
            "auto_labeled": auto_count,
            "admin_labeled": admin_count,
            "by_target": target_counts,
        }
