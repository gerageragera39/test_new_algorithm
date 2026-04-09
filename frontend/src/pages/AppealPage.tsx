import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { ApiError, getAppealAnalytics, getToxicReview, unbanUser } from "../lib/api";
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
  onUnbanComplete,
}: {
  author: AppealAuthorGroup;
  onUnbanComplete?: (bannedUserId: number) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [isUnbanning, setIsUnbanning] = useState(false);
  const [unbanError, setUnbanError] = useState<string | null>(null);
  const canUnban = author.banned_user_id != null;

  function handleClick() {
    setExpanded(!expanded);
  }

  async function handleUnban() {
    if (!author.banned_user_id || isUnbanning) {
      return;
    }
    setIsUnbanning(true);
    setUnbanError(null);
    try {
      const result = await unbanUser(author.banned_user_id);
      if (result.status !== "unbanned") {
        throw new Error(result.youtube_error || "Не удалось снять бан.");
      }
      onUnbanComplete?.(author.banned_user_id);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Не удалось снять бан.";
      setUnbanError(message);
    } finally {
      setIsUnbanning(false);
    }
  }

  return (
    <div className="author-chip-wrap">
      <button type="button" className="author-chip" onClick={handleClick}>
        <strong>{author.author_name}</strong>
        <span className="author-chip-count">{author.comment_count}</span>
      </button>
      {expanded ? (
        <div className="author-comments">
          {canUnban ? (
            <div style={{ marginBottom: 10, display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
              <button
                type="button"
                className="btn secondary inline-btn"
                onClick={() => void handleUnban()}
                disabled={isUnbanning}
              >
                {isUnbanning ? "Снимаю бан..." : "Разбанить"}
              </button>
              {author.youtube_banned ? (
                <span className="muted" style={{ fontSize: "0.85rem" }}>
                  Попытаемся восстановить доступ и на YouTube.
                </span>
              ) : null}
            </div>
          ) : null}
          {unbanError ? <div className="notice error">{unbanError}</div> : null}
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

function AuthorGroupedBlock({
  block,
  onUnbanComplete,
}: {
  block: AppealBlock;
  onUnbanComplete?: (bannedUserId: number) => void;
}) {
  return (
    <div className="appeal-authors-grid appeal-authors-columns">
      {block.authors.map((author) => (
        <AuthorChip
          key={`${author.author_channel_id ?? author.author_name}-${author.banned_user_id ?? "no-ban"}`}
          author={author}
          onUnbanComplete={onUnbanComplete}
        />
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

  const loadPageData = useCallback(async () => {
    if (!videoId) {
      setError("Отсутствует ID видео");
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    try {
      const appealResponse = await getAppealAnalytics(videoId);

      let toxicResponse: ToxicReviewResponse | null = null;
      let toxicError: string | null = null;

      try {
        toxicResponse = await getToxicReview(videoId);
      } catch (toxicErr) {
        const toxicMessage = toxicErr instanceof Error ? toxicErr.message : String(toxicErr);
        if (toxicErr instanceof ApiError && toxicErr.status === 404) {
          toxicError = null;
        } else {
          toxicError = `Ошибка загрузки очереди модерации: ${toxicMessage}`;
        }
      }

      setData(appealResponse);
      setToxicReview(toxicResponse);
      setToxicReviewError(toxicError);
      setError(null);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "Не удалось загрузить отчет";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [videoId]);

  useEffect(() => {
    if (videoId) {
      void loadPageData();
    } else {
      setError("Отсутствует ID видео");
      setIsLoading(false);
    }
  }, [loadPageData, videoId]);

  async function handleBanComplete() {
    if (!videoId) return;
    try {
      const toxicResponse = await getToxicReview(videoId);
      setToxicReview(toxicResponse);
      setToxicReviewError(null);
    } catch (toxicErr) {
      const toxicMessage = toxicErr instanceof Error ? toxicErr.message : String(toxicErr);
      if (toxicErr instanceof ApiError && toxicErr.status === 404) {
        setToxicReview(null);
        setToxicReviewError(null);
      } else {
        setToxicReviewError(`Ошибка загрузки очереди модерации: ${toxicMessage}`);
      }
    }
  }

  async function handleUnbanComplete(bannedUserId: number) {
    setData((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        blocks: prev.blocks.map((block) =>
          block.block_type !== "toxic_auto_banned"
            ? block
            : {
                ...block,
                authors: block.authors.map((author) =>
                  author.banned_user_id === bannedUserId
                    ? {
                        ...author,
                        banned_user_id: null,
                        is_banned_active: false,
                        youtube_banned: false,
                      }
                    : author
                ),
              }
        ),
      };
    });
    await handleBanComplete();
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
                  <AuthorGroupedBlock block={block} onUnbanComplete={handleUnbanComplete} />
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
