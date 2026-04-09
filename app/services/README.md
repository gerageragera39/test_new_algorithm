# app/services/

Business logic layer for both backend pipelines and supporting services.

## Top-Level Services

| File | Main Classes/Functions | Role |
|---|---|---|
| `youtube_client.py` | `YouTubeClient` | YouTube API metadata + comments fetch |
| `preprocessing.py` | `CommentPreprocessor` | Filtering, normalization, moderation, weighting |
| `embeddings.py` | `EmbeddingService`, providers, cache store | Embedding generation + DB/filesystem cache |
| `clustering.py` | `ClusteringService` | HDBSCAN/KMeans clustering and cluster shaping |
| `labeling.py` | `LLMProvider`, `OpenAIChatProvider`, `NoLLMFallbackProvider` | Cluster-level LLM analysis and fallback normalization |
| `briefing.py` | `BriefingService` | Builds `DailyBriefing` skeleton and trend summary |
| `exporter.py` | `ReportExporter` | Markdown/HTML rendering and report persistence |
| `budget.py` | `BudgetGovernor` | OpenAI usage estimation and DB usage recording |
| `runtime_settings.py` | `RuntimeSettingsStore` | Mutable runtime settings persisted to JSON |
| `moderation_llm.py` | moderation decision parsing | Parses strict moderation JSON decisions |
| `openai_compat.py` | helper funcs | GPT model compatibility helpers for API params |
| `openai_endpoint.py` | endpoint guards | OpenAI endpoint mode/allow-list checks |

## Pipeline Packages

## `pipeline/` (Topic Intelligence)

Main orchestrator:

- `pipeline/runner.py` (`DailyRunService`)

Supporting modules:

| File | Responsibility |
|---|---|
| `cluster_enricher.py` | Topic labeling orchestration, postprocessing, dedupe/merge |
| `position_extractor.py` | Position extraction inside topic clusters |
| `report_builder.py` | Disagreement extraction and optional briefing polish |
| `quality_metrics.py` | Quality counters, degraded evaluation, diagnostics payload |
| `text_utils.py` | Shared text/token helpers and title sanitization |
| `constants.py` | Regex/constants used by pipeline internals |

Runtime note:

- Stage `episode_match` is currently a compatibility stage and is explicitly skipped.

## `appeal_analytics/` (Appeal Analytics)

Main orchestrator:

- `appeal_analytics/runner.py` (`AppealAnalyticsService`)

Supporting modules:

| File | Responsibility |
|---|---|
| `llm_classifier.py` | Unified single-pass LLM classifier with heuristic fallback |
| `spam_detector.py` | Duplicate-based author spam detection |
| `toxic_detector.py` | Toxicity helper logic |
| `author_appeal_detector.py` | Author-address detection helpers |
| `runner.py` | End-to-end appeal pipeline execution and persistence |

## Provider Behavior

Current production behavior:

- Topic and appeal LLM logic run through OpenAI provider classes.
- Heuristic fallback logic exists for resilience when LLM calls fail.
- Budget/call usage is tracked, while hard blockers are disabled in current build.
