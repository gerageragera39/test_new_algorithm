"""Domain data transfer objects used across pipeline stages.

Defines Pydantic models for video metadata, raw and processed comments,
cluster drafts, labeling results, topic summaries, and the daily briefing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


@dataclass
class EpisodeTopic:
    """A single editorial topic extracted from the episode transcript."""

    title: str
    summary: str

    def as_prompt_line(self) -> str:
        return f"{self.title}: {self.summary}"


@dataclass
class EpisodeContext:
    """Aggregated episode context including topics and transcript text."""

    source: str
    topics: list[EpisodeTopic] = field(default_factory=list)
    transcript_text: str = ""
    transcript_language: str | None = None

    @property
    def topic_lines(self) -> list[str]:
        return [topic.as_prompt_line() for topic in self.topics]


class VideoMeta(BaseModel):
    """Metadata for a single YouTube video as fetched from the API."""

    youtube_video_id: str
    playlist_id: str
    title: str
    description: str | None = None
    published_at: datetime
    duration_seconds: int | None = None
    url: str


class RawComment(BaseModel):
    """A comment as originally retrieved from YouTube before any processing."""

    youtube_comment_id: str
    parent_comment_id: str | None = None
    author_name: str | None = None
    author_channel_id: str | None = None
    text_raw: str
    like_count: int = 0
    reply_count: int = 0
    published_at: datetime
    is_top_level: bool = True


class ProcessedComment(BaseModel):
    """A comment after normalization, hashing, weighting, and moderation."""

    youtube_comment_id: str
    parent_comment_id: str | None = None
    author_name: str | None = None
    author_channel_id: str | None = None
    text_raw: str
    text_normalized: str
    text_hash: str
    language: str | None = None
    like_count: int = 0
    reply_count: int = 0
    published_at: datetime
    weight: float = 1.0
    is_top_level: bool = True
    is_filtered: bool = False
    filter_reason: str | None = None
    moderation_action: Literal["keep", "flag", "drop"] = "keep"
    moderation_reason: str | None = None
    moderation_source: Literal["rule", "llm", "fallback"] = "rule"
    moderation_score: float | None = None


class ClusterMember(BaseModel):
    """Association of a comment index to a cluster with a similarity score."""

    comment_index: int
    score: float
    is_representative: bool = False


class ClusterDraft(BaseModel):
    """Preliminary cluster data produced by the clustering algorithm before labeling."""

    cluster_key: str
    member_indices: list[int] = Field(default_factory=list)
    representative_indices: list[int] = Field(default_factory=list)
    centroid: list[float] | None = None
    size_count: int
    share_pct: float
    weighted_share: float
    is_emerging: bool = False
    assignment_confidence: float = 0.0
    ambiguous_member_count: int = 0
    ambiguous_share_pct: float = 0.0


class ClusterLabelResult(BaseModel):
    """LLM-generated label, description, and sentiment for a cluster."""

    label: str
    description: str
    author_actions: list[str]
    sentiment: Literal["positive", "neutral", "negative"]
    emotion_tags: list[str] = Field(default_factory=list)
    intent_distribution: dict[str, int] = Field(default_factory=dict)
    representative_quotes: list[str] = Field(default_factory=list)


class TopicPosition(BaseModel):
    """A distinct opinion or stance identified within a topic cluster."""

    key: str
    title: str
    summary: str
    markers: list[str] = Field(default_factory=list)
    prototype: str
    count: int
    pct: float
    weighted_count: float
    weighted_pct: float
    comments: list[str] = Field(default_factory=list)
    is_undetermined: bool = False
    is_author_disagreement: bool = False
    coherence_score: float = 0.0
    single_claim_passed: bool = True


class TopicSummary(BaseModel):
    """Full summary of a topic cluster including labels, positions, and statistics."""

    cluster_key: str
    label: str
    description: str
    author_actions: list[str]
    sentiment: str
    emotion_tags: list[str]
    intent_distribution: dict[str, int]
    representative_quotes: list[str]
    question_comments: list[str] = Field(default_factory=list)
    positions: list[TopicPosition] = Field(default_factory=list)
    size_count: int
    share_pct: float
    weighted_share: float
    is_emerging: bool = False
    source: Literal["episode_topic", "comment_topic"] = "comment_topic"
    coherence_score: float = 0.0
    centroid: list[float] | None = None
    assignment_confidence: float = 0.0
    ambiguous_share_pct: float = 0.0
    soft_assignment_notes: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    """Structured author action derived from a topic cluster."""

    topic_cluster_key: str
    topic_label: str
    share_pct: float = 0.0
    priority: int = 0
    action: str
    key_criticism: str = ""
    key_question: str = ""


class TopicTrendPoint(BaseModel):
    """One point on a topic history chart."""

    video_id: str
    video_title: str
    published_at: datetime
    share_pct: float
    weighted_share: float
    matched_topic_label: str = ""
    similarity: float = 0.0
    is_current: bool = False


class TopicTrendSeries(BaseModel):
    """Historical evolution of a current topic across previous reports."""

    cluster_key: str
    topic_label: str
    summary: str = ""
    points: list[TopicTrendPoint] = Field(default_factory=list)


class DailyBriefing(BaseModel):
    """Top-level briefing for a video containing topics, actions, risks, and quotes."""

    video_id: str
    video_title: str
    published_at: datetime
    mode: str
    executive_summary: str
    top_topics: list[TopicSummary]
    actions_for_tomorrow: list[str]
    action_items: list[ActionItem] = Field(default_factory=list)
    misunderstandings_and_controversies: list[str]
    audience_requests_and_questions: list[str]
    risks_and_toxicity: list[str]
    representative_quotes: list[str]
    author_disagreement_comments: list[str] = Field(default_factory=list)
    trend_vs_previous: list[str] = Field(default_factory=list)
    topic_trends: list[TopicTrendSeries] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
