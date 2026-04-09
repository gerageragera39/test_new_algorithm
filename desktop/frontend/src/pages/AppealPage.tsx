import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { ApiError, getAppealAnalytics, getToxicReview } from "../lib/api";
import type { AppealAnalyticsResponse, AppealAuthorGroup, AppealBlock, AppealBlockItem, ToxicReviewResponse } from "../types/api";
import { ToxicReviewPanel } from "../components/ToxicReviewPanel";

const BLOCK_COLORS: Record<string, string> = {
  constructive_criticism: "agree",
  constructive_question: "agree",
  author_appeal: "neutral",
  toxic_auto_banned: "danger",
  toxic_manual_review: "neutral",
};

const BLOCK_ORDER: string[] = [
  "constructive_question",
  "constructive_criticism",
  "author_appeal",
  "toxic_auto_banned",
  "toxic_manual_review",
];

const WORD_LIMIT = 40;

function ExpandableText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const words = text.split(" ");

  if (words.length <= WORD_LIMIT) {
    return <>{text}</>;
  }

  const preview = words.slice(0, WORD_LIMIT).join(" ");

  if (expanded) {
    return (
      <>
        {text}{" "}
        <button
          type="button"
          style={{ background: "none", border: "none", cursor: "pointer", padding: "0 2px", opacity: 0.45, fontSize: "0.85em" }}
          onClick={() => setExpanded(false)}
        >
          ↑
        </button>
      </>
    );
  }

  return (
    <>
      {preview}
      <button
        type="button"
        style={{ background: "none", border: "none", cursor: "pointer", padding: "0 2px", color: "var(--accent, #4f8ef7)", fontWeight: 600 }}
        onClick={() => setExpanded(true)}
        title="Показать полностью"
      >
        …
      </button>
    </>
  );
}

function AuthorChip({
  author,
}: {
  author: AppealAuthorGroup;
}) {
  const [expanded, setExpanded] = useState(false);

  function handleClick() {
    setExpanded(!expanded);
  }

  return (
    <div className="author-chip-wrap">
      <button type="button" className="author-chip" onClick={handleClick}>
        <strong>{author.author_name}</strong>
        <span className="author-chip-count">{author.comment_count}</span>
      </button>
      {expanded ? (
        <div className="author-comments">
          <ul className="quote-list expanded">
            {author.comments.map((c, i) => (
              <li key={`${c.comment_id}-${i}`}><ExpandableText text={c.text} /></li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

function AuthorGroupedBlock({ block }: { block: AppealBlock }) {
  return (
    <div className="appeal-authors-grid appeal-authors-columns">
      {block.authors.map((author) => (
        <AuthorChip key={author.author_name} author={author} />
      ))}
      {block.authors.length === 0 ? <p className="muted">Нет комментариев.</p> : null}
    </div>
  );
}

function ScoreBadge({ score }: { score: number | null }) {
  if (score == null) return null;
  const color = score >= 7 ? "agree" : score >= 4 ? "neutral" : "disagree";
  return (
    <span className={`chip ${color} score-badge`}>
      {score}/10
    </span>
  );
}

function FlatBlock({ block }: { block: AppealBlock }) {
  const [showAll, setShowAll] = useState(false);
  const previewLimit = 5;
  const items = block.items;
  const hasScores = block.block_type === "constructive_criticism" || block.block_type === "constructive_question";
  const visible = showAll ? items : items.slice(0, previewLimit);
  const remaining = items.length - previewLimit;

  return (
    <div>
      {items.length === 0 ? (
        <p className="muted">Нет комментариев.</p>
      ) : (
        <>
          <ul className="quote-list">
            {visible.map((item, i) => (
              <li key={`${item.comment_id}-${i}`}>
                {hasScores ? <ScoreBadge score={item.score} /> : null}
                <span className="muted" style={{ fontSize: "0.82rem" }}>
                  {item.author_name ?? "—"}:
                </span>{" "}
                <ExpandableText text={item.text} />
              </li>
            ))}
          </ul>
          {remaining > 0 ? (
            <button type="button" className="btn secondary inline-btn" onClick={() => setShowAll(!showAll)}>
              {showAll ? "Свернуть" : `Показать все (${items.length})`}
            </button>
          ) : null}
        </>
      )}
    </div>
  );
}

export function AppealPage() {
  const { videoId = "" } = useParams();
  const [data, setData] = useState<AppealAnalyticsResponse | null>(null);
  const [toxicReview, setToxicReview] = useState<ToxicReviewResponse | null>(null);
  const [toxicReviewError, setToxicReviewError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function load() {
      setIsLoading(true);
      try {
        const appealResponse = await getAppealAnalytics(videoId);
        
        // Try to load toxic review separately (may not exist)
        let toxicResponse: ToxicReviewResponse | null = null;
        let toxicError: string | null = null;
        
        try {
          toxicResponse = await getToxicReview(videoId);
        } catch (toxicErr) {
          // Distinguish between "no review queue" vs actual error
          const toxicMessage = toxicErr instanceof Error ? toxicErr.message : String(toxicErr);
          if (toxicErr instanceof ApiError && toxicErr.status === 404) {
            // No review queue is normal - leave toxicResponse as null
            toxicError = null;
          } else {
            // Real error loading toxic review
            toxicError = `Ошибка загрузки очереди модерации: ${toxicMessage}`;
          }
        }
        
        if (isMounted) {
          setData(appealResponse);
          setToxicReview(toxicResponse);
          setToxicReviewError(toxicError);
          setError(null);
        }
      } catch (loadError) {
        if (isMounted) {
          const message = loadError instanceof Error ? loadError.message : "Не удалось загрузить отчет";
          setError(message);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    if (videoId) {
      void load();
    } else {
      setError("Отсутствует ID видео");
      setIsLoading(false);
    }

    return () => {
      isMounted = false;
    };
  }, [videoId]);

  async function handleBanComplete() {
    // Reload toxic review after ban
    if (videoId) {
      try {
        const toxicResponse = await getToxicReview(videoId);
        setToxicReview(toxicResponse);
        setToxicReviewError(null);
      } catch (toxicErr) {
        const toxicMessage = toxicErr instanceof Error ? toxicErr.message : String(toxicErr);
        if (toxicErr instanceof ApiError && toxicErr.status === 404) {
          // No more items in review queue
          setToxicReview(null);
          setToxicReviewError(null);
        } else {
          setToxicReviewError(`Ошибка загрузки очереди модерации: ${toxicMessage}`);
        }
      }
    }
  }

  return (
    <>
      {isLoading ? <section className="card">Загрузка аналитики обращений...</section> : null}
      {error ? (
        <section className="card">
          <div className="notice error">{error}</div>
        </section>
      ) : null}

      {data ? (
        <>
          <section className="card accent">
            <div className="section-head compact">
              <div>
                <p className="eyebrow">Appeal Analytics</p>
                <h1>{data.video_title}</h1>
              </div>
              <a
                href={`https://www.youtube.com/watch?v=${data.video_id}`}
                target="_blank"
                rel="noreferrer"
                className="text-link strong"
              >
                Открыть видео на YouTube
              </a>
            </div>
            <div className="stats-grid" style={{ marginTop: 12 }}>
              <article className="stat">
                <span>Всего</span>
                <strong>{data.total_comments}</strong>
                <small>комментариев</small>
              </article>
              <article className="stat">
                <span>Классифицировано</span>
                <strong>{data.classified_comments}</strong>
                <small>комментариев</small>
              </article>
              <article className="stat">
                <span>Блоков</span>
                <strong>{data.blocks.length}</strong>
                <small>категорий</small>
              </article>
            </div>
          </section>

          {[...data.blocks]
            .sort((a, b) => {
              const ai = BLOCK_ORDER.indexOf(a.block_type);
              const bi = BLOCK_ORDER.indexOf(b.block_type);
              return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
            })
            .map((block) => {
            const colorClass = BLOCK_COLORS[block.block_type] ?? "neutral";
            const isAuthorGrouped = block.block_type === "toxic_auto_banned";
            const isToxicReview = block.block_type === "toxic_manual_review";

            return (
              <section className="card" key={block.block_type}>
                <div className="section-head compact">
                  <h2>
                    {block.display_label}
                    <span className={`chip ${colorClass}`} style={{ marginLeft: 8, verticalAlign: "middle" }}>
                      {block.item_count}
                    </span>
                  </h2>
                </div>

                {isToxicReview ? (
                  toxicReviewError ? (
                    <div className="notice error" style={{ marginTop: 12 }}>
                      {toxicReviewError}
                    </div>
                  ) : toxicReview && toxicReview.items.length > 0 ? (
                    <ToxicReviewPanel
                      videoId={videoId}
                      items={toxicReview.items}
                      onBanComplete={handleBanComplete}
                    />
                  ) : (
                    <p className="muted" style={{ marginTop: 12 }}>Нет комментариев для ручной проверки.</p>
                  )
                ) : isAuthorGrouped ? (
                  <AuthorGroupedBlock block={block} />
                ) : (
                  <FlatBlock block={block} />
                )}
              </section>
            );
          })}

          <section className="card">
            <div className="section-head compact">
              <Link to="/videos" className="text-link">К списку видео</Link>
            </div>
          </section>
        </>
      ) : null}
    </>
  );
}
