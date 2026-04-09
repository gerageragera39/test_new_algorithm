# Topic Clustering Optimization Plan

## Goal

Make the topic intelligence pipeline significantly more reliable for YouTube videos with **300-600 comments** by default and **up to 1000-2000 comments** in heavier runs, while avoiding unnecessary comment loss.

The optimized design keeps **HDBSCAN** as the primary clustering algorithm, but changes the surrounding pipeline so that clustering quality depends less on fragile raw-density behavior in the original embedding space.

## Implemented architecture

### 1. Topic-focused embedding preparation

Embeddings are now requested with an explicit **task mode**:

- `topic` — main subject / semantic topic of the comment

The embedding service supports task-aware text preparation:

- E5-family models continue to use `query:` prefixing.
- instruction-aware models (Qwen / GTE / instruct-like models) can receive a topic-clustering instruction automatically.
- cache keys are task-scoped, so future stance/position embeddings can coexist safely.

### 2. Dimensionality reduction before clustering

For larger batches, the pipeline reduces dense vectors with **PCA** before HDBSCAN.

Why PCA was chosen now:

- already available via `scikit-learn`
- no new dependency required
- stable for 300-2000 comments
- materially improves density estimation versus clustering directly in high-dimensional embedding space

Default behavior:

- enabled for batches above `cluster_reduction_min_comments`
- target dimension controlled by `cluster_reduction_target_dim`

### 3. Adaptive HDBSCAN parameters

HDBSCAN no longer relies on one static parameter pair.

The service now evaluates a small adaptive grid around the dataset size and scores candidates by:

- cluster coverage (`1 - noise_ratio`)
- silhouette (when meaningful)
- distance from an expected cluster-count range

It also tries several `cluster_selection_epsilon` values to reduce micro-fragmentation.

### 4. Soft assignment instead of hard dropping

A major failure mode for comment clustering is losing too many comments into `noise`.

The optimized pipeline now:

- reads HDBSCAN confidence (`probabilities_`)
- computes reduced-space centroid similarity
- softly assigns borderline/noise comments to the nearest good cluster when the semantic similarity is high enough
- marks uncertain assignments as **ambiguous** instead of pretending they are fully clean

This improves coverage without forcing every weak comment into a bad cluster.

### 5. Ambiguity-aware cluster quality

Each cluster now carries:

- average assignment confidence
- ambiguous member count
- ambiguous share percent

These metrics are propagated into topic summaries, report metadata, and diagnostics.

### 6. Better diagnostics

The report metadata and diagnostics JSON now capture:

- embedding provider/model/task
- original embedding dimension
- PCA reduction summary
- chosen clustering parameters
- average assignment confidence
- ambiguous-comment share

This makes tuning measurable rather than guess-based.

## Recommended local embedding models

### Best quality / modern recommendation

- `Qwen/Qwen3-Embedding-0.6B`

Why:

- strong multilingual semantic quality
- instruction-friendly
- practical for local use compared with larger frontier open models

### Strong alternative

- `BAAI/bge-m3`

Why:

- still very strong multilingual model
- good semantic retrieval quality
- especially useful if you later want hybrid dense + lexical workflows

### Conservative fallback

- `intfloat/multilingual-e5-large`

Why:

- robust and well-understood
- already integrated
- still competitive as a fallback, though not the most modern option

## Suggested operating profiles

### Balanced profile

- embedding model: `Qwen/Qwen3-Embedding-0.6B`
- reduction: PCA to 32 dims
- soft assignment: enabled
- cluster max count: 10-12

### Cost-safe fallback profile

- embedding model: `BAAI/bge-m3` or `intfloat/multilingual-e5-large`
- reduction: PCA to 24-32 dims
- soft assignment: enabled

### High-volume profile (1000-2000 comments)

- embedding model: `Qwen/Qwen3-Embedding-0.6B`
- reduction: PCA to 32-48 dims
- keep adaptive HDBSCAN enabled
- monitor ambiguous-comment share and fallback to stronger model when ambiguous share spikes

## Tuning checklist

When evaluating runs, pay attention to:

1. **Average assignment confidence** — higher is better.
2. **Ambiguous share** — too high means the embedding space or clustering params are weak.
3. **Noise ratio / emerging clusters** — too high means the model or density setup is underfitting.
4. **Duplicate-topic rate** — if many similar clusters survive, increase merge strictness or raise epsilon slightly.
5. **Low-coherence clusters** — inspect whether the embedding model is grouping by tone instead of subject.

## Next recommended improvements

The current implementation is a strong production upgrade without new dependencies. The next high-value steps are:

1. add an optional UMAP stage for clustering space (only if you explicitly allow a new dependency),
2. add a second embedding track for **stance/position** clustering inside each topic,
3. build a small manually-labeled benchmark set from your real channel comments,
4. run A/B comparisons between `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3`, and `multilingual-e5-large` on your own data.

## Files changed for this redesign

- `app/services/embeddings.py`
- `app/services/clustering.py`
- `app/services/pipeline/runner.py`
- `app/services/pipeline/cluster_enricher.py`
- `app/services/pipeline/quality_metrics.py`
- `app/services/exporter.py`
- `frontend/src/pages/ReportPage.tsx`
- `frontend/src/styles.css`
- `frontend/src/types/api.ts`
- mirrored `desktop/...` files
