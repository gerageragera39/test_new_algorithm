# Pipelines: Technical Specification

This document describes the two production pipelines implemented in this repository:

1. **Topic Intelligence Pipeline** (`DailyRunService`)
2. **Appeal Analytics Pipeline** (`AppealAnalyticsService`)

It reflects the current codebase in `app/services/**`.

## 1) Topic Intelligence Pipeline

Primary orchestrator: `app/services/pipeline/runner.py` (`DailyRunService`)

Execution stages recorded in `runs.meta_json`:

| # | Stage Key | Runtime Label (typical) |
|---|---|---|
| 1 | `context` | Loading previous report context |
| 2 | `comments_fetch` | Fetching YouTube comments |
| 3 | `preprocess` | Preprocessing comments |
| 4 | `comments_persist` | Persisting processed comments |
| 5 | `embeddings` | Building embeddings for processed comments |
| 6 | `clustering` | Clustering comment embeddings |
| 7 | `episode_match` | Skipped: transcription removed |
| 8 | `labeling` | Labeling clusters with LLM/fallback provider |
| 9 | `clusters_persist` | Persisting clusters and cluster items |
| 10 | `briefing` | Building daily briefing and trend deltas |
| 11 | `report_export` | Rendering and saving report artifacts |

## Stage 1: Context

Current behavior:

- Loads previous report topics for trend comparison.
- Builds a comment-only `EpisodeContext` (`source="comments_only"`).
- Transcript-derived topics are not generated in the active runtime path.

## Stage 2: YouTube Comment Fetch

Component: `app/services/youtube_client.py`

Features:

- playlist-driven latest video resolution,
- URL-to-video-ID parsing for explicit runs,
- rate limiting + retries,
- hybrid comment fetch:
  - time-sorted sample,
  - relevance-sorted sample (optional),
  - deterministic merge by comment ID with time-first priority.

Output model: `RawComment`.

## Stage 3: Preprocessing and Moderation

Component: `app/services/preprocessing.py`

Pipeline:

1. normalization (`normalize_text`, URL stripping + whitespace collapse),
2. filtering (`empty`, `too_short`, `noise`, `low_signal`, `duplicate`),
3. rule moderation:
   - link spam patterns,
   - profanity/toxicity handling policy,
   - borderline scoring,
4. optional LLM borderline moderation (`runner._apply_llm_borderline_moderation`),
5. weight calculation (likes, replies, length, recency),
6. duplicate-volume boost on canonical comments.

Output model: `ProcessedComment`.

## Stage 4: Comment Persistence

Comments are upserted into `comments` table with moderation fields:

- `is_filtered`, `filter_reason`,
- `moderation_action`, `moderation_reason`, `moderation_source`, `moderation_score`.

Run-level moderation counters are written into `runs.meta_json`.

## Stage 5: Embeddings

Component: `app/services/embeddings.py`

Providers:

- Local: `LocalSentenceTransformerProvider`
  - default model: `intfloat/multilingual-e5-large`
  - task-aware text preparation for topic clustering
  - E5 prefix handling (`query: `)
  - optional instruction prefix for instruction-aware local models
  - normalized vectors
- OpenAI: `OpenAIEmbeddingProvider`
  - model: `text-embedding-3-small` by default
  - normalized vectors for parity with local mode

Caching:

- DB cache: `embedding_cache` table
- filesystem cache: `data/cache/embeddings/...`

## Stage 6: Clustering

Component: `app/services/clustering.py`

Algorithm:

- optional PCA reduction before clustering for larger batches,
- adaptive HDBSCAN parameter candidates by dataset size,
- candidate scoring by coverage + silhouette + expected cluster count,
- soft assignment for borderline / noise comments using confidence + centroid similarity,
- acceptance by cluster count + noise ratio thresholds,
- KMeans fallback when HDBSCAN is insufficient,
- optional large-noise split into multiple emerging groups,
- optional large-cluster split,
- optional cluster merge pass,
- cluster count cap with overflow into `emerging_other`.

Output model: `ClusterDraft`.

## Stage 7: Episode Match (Compatibility Stage)

Status in current runtime:

- Stage key exists and progress is tracked.
- Label is set to `Skipped: transcription removed`.
- No transcript topic matching is executed.

Notes:

- Legacy transcript-related settings still exist in config for compatibility.
- `transcript_matcher.py` has been removed. Stage is an empty compatibility stub.

## Stage 8: Labeling and Position Extraction

Components:

- `app/services/labeling.py` (`OpenAIChatProvider`, `NoLLMFallbackProvider`)
- `app/services/pipeline/cluster_enricher.py`
- `app/services/pipeline/position_extractor.py`

Process:

1. cluster-level labeling:
   - label,
   - grounded description,
   - sentiment/emotions/intents,
   - representative quotes,
   - author actions.
2. position extraction inside cluster:
   - primary: single-call LLM position identification,
   - fallback: embedding subclustering + LLM position naming.
3. postprocessing:
   - uncertain topic handling,
   - quote dedupe/cross-topic cleanup,
   - optional topic merge.
4. quality metrics and degraded-run evaluation:
   - fallback title rates,
   - undetermined share,
   - LLM failure counters.

## Stage 9: Cluster Persistence

Persists:

- `clusters` records,
- `cluster_items` links to comments.

Each cluster stores sentiment, intents, actions, representative quotes, and centroid.

## Stage 10: Briefing

Components:

- `app/services/briefing.py`
- `app/services/pipeline/report_builder.py`

Produces `DailyBriefing`:

- executive summary,
- ranked top topics,
- concrete actions for the next episode,
- misunderstandings / controversies that need clarification,
- audience requests and FAQ-style questions,
- risk / toxicity notes,
- representative quotes,
- disagreement comments (position-level first, regex fallback second),
- trend vs previous report by centroid similarity,
- run metadata diagnostics.

Optional: additional OpenAI polish call for `executive_summary` when enabled.

Post-position cluster renaming is now conditional: the extra LLM call is only used
for weak/generic topic labels, reducing cost without changing already grounded topics.

## Stage 11: Export

Component: `app/services/exporter.py`

Artifacts:

- Markdown report: `data/reports/YYYY-MM-DD/{video_id}.md`
- HTML report: `data/reports/YYYY-MM-DD/{video_id}.html`
- Diagnostics JSON: `data/reports/YYYY-MM-DD/{video_id}.cluster_diagnostics.json` (if enabled)
- DB report row in `reports` (`content_markdown`, `content_html`, `structured_json`)

Run is finalized as `completed` or `failed`.

## 2) Appeal Analytics Pipeline

Primary orchestrator: `app/services/appeal_analytics/runner.py` (`AppealAnalyticsService`)

Execution stages recorded in `appeal_runs.meta_json`:

| # | Stage | Label |
|---|---|---|
| 1 | load | Loading comments |
| 2 | classify | LLM classification + Question Refiner |
| 3 | refine | Question refinement + Criticism filter + Promotion |
| 4 | toxic | Toxic classification + Auto-ban + Manual review |
| 5 | persist | Persisting results |

## Stage A1: Comment Source

- Reuses `comments` already stored for the video.
- If missing, fetches from YouTube and persists minimally (`_persist_raw_comments`).

## Stage A2: Unified LLM Classification

Component: `app/services/appeal_analytics/llm_classifier.py`

**Single-pass batch classification** into:

- `toxic`: insults/offensive toward author/guests/content
- `criticism`: constructive criticism of author's position
- `question`: genuine questions to the author
- `appeal`: direct requests/suggestions to the author
- `skip` (ignored for block persistence)

Important behavior:

- classification is explicitly author-directed (comments about third parties should go to `skip`),
- on LLM/budget errors, classifier falls back to heuristic regex scoring per batch,
- **partial LLM coverage**: if the LLM omits some comment numbers from the response, the missing entries are automatically filled by heuristic fallback — no comment is silently lost,
- comments are clipped via **head+tail strategy** (first ~180 words + `[...]` + last ~40 words) to preserve both opening context and trailing questions,
- priority at category overlap: `question > criticism > appeal > toxic > skip`,
- comments with both criticism and a question are classified as `question`.

## Stage A3: Question Refiner + Criticism Filter + Promotion

Components: `question_refiner.py`, `political_criticism_refiner.py`

**Question Refiner (second pass)**:

All `question` candidates are enriched via a second LLM call that assigns:
- `topic`: geopolitical/thematic label
- `score`: quality 1–10 (becomes the primary sort key for `constructive_question`)
- `short`: Russian-language 5–12 word summary
- `depth_score`: analytical depth 0–10
- `question_type`: one of `myth_claim`, `fact_check`, `analysis_why_how`, `prediction_what_next`, `strategy_advice`, `clarification_needed`, `attack_ragebait`, `meme_one_liner`

**Promotion from Criticism:**
- Criticism comments with question signals (?, interrogative words) go through Question Refiner
- If confirmed as real question (not `attack_ragebait` or `meme_one_liner`), promoted to `question` block

**Political Criticism Filter:**
- Retains only criticism backed by political argument
- Drops non-political complaints (production quality, audio, etc.)

**Local Question Mark Check:**
- Question candidates without "?" in text moved back to criticism
- Prevents argumentative comments from polluting question block

## Stage A4: Enhanced Toxic Classification + Auto-Ban + Manual Review

Components: `toxic_classifier.py`, `youtube_ban_service.py`, `toxic_training_service.py`

**Toxic Classification Flow:**

1. **Unified toxic candidates**: Stage A2 provides broad toxic candidates.
2. **LLM Target Detection**: Classify every toxic candidate into:
   - `author`: insults to channel author
   - `guest`: insults to video guests (from `video_settings.guest_names`)
   - `content`: insults to video/channel content
   - `undefined`: insults to unspecified persons
   - `third_party`: insults to politicians/public figures → **IGNORED**
3. **Confidence Scoring**: LLM assigns 0.0-1.0 confidence
4. **Split Decision**:
   - **Auto-ban**: `target in (author, guest, content)` AND `confidence >= AUTO_BAN_THRESHOLD` (0.85)
   - **Manual review**: every remaining non-`third_party` toxic candidate not auto-banned
   - **Ignored**: `target == third_party`

**CRITICAL**:
- `third_party` insults are **NEVER** auto-banned, regardless of confidence.
- `undefined` is review-only. It never goes to auto-ban.
- Low-confidence non-`third_party` toxic candidates are still retained in manual review to avoid silent recall loss.

**Auto-Ban Execution:**
- Real YouTube hide-user action via OAuth2 API (if configured)
- Fallback to CSV export if OAuth not available
- Record in `banned_users` table
- Save to `toxic_training_data` for ML training
- Uses author channel ID as the primary stable identity when available
- Duplicate suppression and review exclusion are channel-wide, matching YouTube semantics

**Manual Review Queue:**
- Sorted by confidence, but includes lower-confidence non-`third_party` cases as fallback review items
- Excludes already hidden/banned users channel-wide
- Admin can review/ban from UI

## Stage A5: Persistence Model

Tables:

- `appeal_runs`
- `appeal_blocks`
- `appeal_block_items`

Persisted output blocks:

1. `constructive_criticism`: Political criticism backed by argument
2. `constructive_question`: Questions enriched with topic/score/depth
3. `author_appeal`: Direct requests/suggestions
4. `toxic_auto_banned`: High-confidence toxic (auto-banned)
5. `toxic_manual_review`: Medium-confidence toxic (needs admin review)

API read model:

- `/appeal/{video_id}` returns block summaries and comments
- `/appeal/{video_id}/toxic-review` returns manual review queue (excludes banned users)
- `POST /appeal/ban-user` executes manual ban from admin panel
- For `toxic_auto_banned`, items grouped by author
- For `toxic_manual_review`, items displayed with ban action in UI

## Triggers and Scheduling

Synchronous API triggers:

- `POST /run/latest?sync=true`
- `POST /run/video?sync=true`
- `POST /appeal/run?sync=true`

Async triggers:

- same endpoints without `sync=true` dispatch Celery tasks.

Scheduled trigger:

- `run_scheduled_latest_task` runs every minute via Celery Beat,
- if runtime schedule matches and not already triggered today, it dispatches:
  - `run_latest_task` (Topic Intelligence),
  - `run_appeal_analytics_task` (Appeal Analytics).

## Cost and Quota Behavior

Usage tracking is active:

- every OpenAI call is persisted in `budget_usage`,
- `/budget` exposes daily cost and token details.

Blocking behavior:

- hard budget/call-limit enforcement is currently disabled in this build (tracking-only mode).

## Failure and Recovery

Both pipelines:

- create run records at start,
- update progress metadata per stage,
- persist partial diagnostics where possible,
- mark run as `failed` with truncated error details on exceptions.

Topic pipeline includes additional degraded quality flags in `runs.meta_json` and report metadata.
