import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getReportDetail } from "../lib/api";
import { formatPercent } from "../lib/format";
import type {
  ReportDetailResponse,
  TopicSummary,
  TopicTrendPoint,
  TopicTrendSeries,
} from "../types/api";

function getTopicComments(topic: TopicSummary, detail: ReportDetailResponse): string[] {
  return detail.topic_comments[topic.cluster_key] ?? topic.representative_quotes;
}

function getTopicQuestionComments(topic: TopicSummary, detail: ReportDetailResponse): string[] {
  return detail.topic_question_comments[topic.cluster_key] ?? topic.question_comments;
}

function getOrderedPositions(topic: TopicSummary) {
  return [...(topic.positions ?? [])].sort((left, right) => {
    if (left.is_undetermined !== right.is_undetermined) {
      return Number(left.is_undetermined) - Number(right.is_undetermined);
    }
    return right.count - left.count;
  });
}

function formatShortDate(value: string) {
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
  }).format(new Date(value));
}

function renderQuoteList(items: string[], previewLimit: number, summaryLabel: string) {
  if (items.length === 0) {
    return <p className="muted">Нет комментариев.</p>;
  }

  const head = items.slice(0, previewLimit);
  const tail = items.slice(previewLimit);

  return (
    <>
      <ul className="quote-list">
        {head.map((item, index) => (
          <li key={`${index}-${item.slice(0, 24)}`}>{item}</li>
        ))}
      </ul>
      {tail.length > 0 ? (
        <details className="expandable">
          <summary>{summaryLabel.replace("{count}", String(tail.length))}</summary>
          <ul className="quote-list expanded">
            {tail.map((item, index) => (
              <li key={`tail-${index}-${item.slice(0, 24)}`}>{item}</li>
            ))}
          </ul>
        </details>
      ) : null}
    </>
  );
}

function renderBulletSection(
  title: string,
  items: string[],
  emptyMessage: string,
  options?: { hideWhenEmpty?: boolean }
) {
  if (options?.hideWhenEmpty && items.length === 0) {
    return null;
  }

  return (
    <section className="card">
      <h2>{title}</h2>
      {items.length > 0 ? (
        <ul className="quote-list expanded">
          {items.map((item, index) => (
            <li key={`${title}-${index}-${item.slice(0, 24)}`}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="muted">{emptyMessage}</p>
      )}
    </section>
  );
}

function buildSparklinePath(points: TopicTrendPoint[], width: number, height: number) {
  if (points.length === 0) {
    return "";
  }
  const maxValue = Math.max(...points.map((point) => point.share_pct), 1);
  const stepX = points.length > 1 ? width / (points.length - 1) : width;
  return points
    .map((point, index) => {
      const x = index * stepX;
      const y = height - (point.share_pct / maxValue) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function TrendSparkline({ series }: { series: TopicTrendSeries }) {
  const width = 180;
  const height = 44;
  const path = buildSparklinePath(series.points, width, height);
  const maxValue = Math.max(...series.points.map((point) => point.share_pct), 1);

  return (
    <div className="trend-chart-wrap compact">
      <svg viewBox={`0 0 ${width} ${height + 16}`} className="trend-chart compact" role="img" aria-label={series.topic_label}>
        <path d={path} className="trend-chart-line" />
        {series.points.map((point, index) => {
          const x = series.points.length > 1 ? (index * width) / (series.points.length - 1) : width / 2;
          const y = height - (point.share_pct / maxValue) * height;
          return (
            <g key={`${series.cluster_key}-${point.video_id}-${index}`}>
              <circle cx={x} cy={y} r={point.is_current ? 4 : 3} className={point.is_current ? "trend-dot current" : "trend-dot"} />
              <text x={x} y={height + 12} textAnchor="middle" className="trend-axis-label">
                {formatShortDate(point.published_at)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function ReportPage() {
  const { videoId = "" } = useParams();

  const [data, setData] = useState<ReportDetailResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function load() {
      setIsLoading(true);
      try {
        const response = await getReportDetail(videoId);
        if (isMounted) {
          setData(response);
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

  const briefing = data?.report.briefing;
  const trendMap = useMemo(
    () => new Map((data?.topic_trends ?? []).map((series) => [series.cluster_key, series])),
    [data?.topic_trends]
  );

  function scrollToTopic(clusterKey: string) {
    const node = document.getElementById(`topic-${clusterKey}`);
    if (node) {
      node.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  return (
    <>
      {isLoading ? <section className="card">Загрузка отчета...</section> : null}
      {error ? <section className="card"><div className="notice error">{error}</div></section> : null}

      {briefing && data ? (
        <>
          <section className="card accent">
            <div className="section-head compact">
              <div>
                <p className="eyebrow">Report Overview</p>
                <h1>{briefing.video_title}</h1>
              </div>
              <a
                href={`https://www.youtube.com/watch?v=${briefing.video_id}`}
                target="_blank"
                rel="noreferrer"
                className="text-link strong"
              >
                Открыть видео на YouTube
              </a>
            </div>
            <p className="summary">{briefing.executive_summary}</p>
          </section>

          <section className="card">
            <h2>Ключевые моменты для {data.author_name}</h2>
            <div className="topics-col">
              {briefing.top_topics.length > 0 ? (
                briefing.top_topics.map((topic, index) => (
                  <article className="topic-item full" key={`key-${topic.cluster_key}`}>
                    <h3>
                      {index + 1}. {topic.label}
                      <span className="muted"> — {formatPercent(topic.share_pct)} от всех комментариев</span>
                    </h3>
                    <p>{topic.description}</p>
                  </article>
                ))
              ) : (
                <p className="muted">Нет тематических кластеров для отображения.</p>
              )}
            </div>
          </section>

          <section className="card">
            <div className="section-head compact">
              <div>
                <h2>Что сделать в следующем выпуске</h2>
                <p className="muted">Название темы кликабельно и ведет к нужному кластеру в разделе «Топ-темы».</p>
              </div>
            </div>
            {briefing.action_items.length > 0 ? (
              <div className="action-grid">
                {briefing.action_items.map((item) => (
                  <article className="action-card detailed" key={`${item.topic_cluster_key}-${item.priority}`}>
                    <div className="action-header">
                      <button type="button" className="topic-jump" onClick={() => scrollToTopic(item.topic_cluster_key)}>
                        {item.topic_label}
                      </button>
                      {item.share_pct > 0 ? <span className="action-badge">{formatPercent(item.share_pct)}</span> : null}
                    </div>
                    {item.action ? <p className="mt">{item.action}</p> : null}
                    <section className="action-focus">
                      <p className="action-label">Главный вопрос аудитории</p>
                      <p>{item.key_question || "Явного повторяющегося вопроса по теме не найдено."}</p>
                    </section>
                  </article>
                ))}
              </div>
            ) : (
              <p className="muted">Конкретные действия не были сформированы.</p>
            )}
          </section>

          {renderBulletSection(
            "Где аудитория спорит или не понимает подачу",
            briefing.misunderstandings_and_controversies,
            "Крупных зон непонимания не выявлено.",
            { hideWhenEmpty: true }
          )}

          {renderBulletSection(
            "Риски и токсичность",
            briefing.risks_and_toxicity,
            "Существенных рисков эскалации не найдено.",
            { hideWhenEmpty: true }
          )}

          {data.topic_trends.length > 0 ? (
            <section className="card">
              <h2>Динамика относительно прошлых выпусков</h2>
              <div className="trend-grid">
                {data.topic_trends.map((series) => (
                  <article className="topic-item trend-card compact" key={series.cluster_key}>
                    <div className="section-head compact">
                      <div>
                        <h3>{series.topic_label}</h3>
                        <p className="muted">{series.summary}</p>
                      </div>
                      <button type="button" className="text-link topic-inline-link" onClick={() => scrollToTopic(series.cluster_key)}>
                        Открыть кластер
                      </button>
                    </div>
                    <TrendSparkline series={series} />
                    <div className="trend-summary-row">
                      <span>
                        Сейчас: <strong>{formatPercent(series.points[series.points.length - 1]?.share_pct ?? 0)}</strong>
                      </span>
                      <span className="muted">
                        {series.points
                          .slice()
                          .reverse()
                          .find((point) => !point.is_current)?.matched_topic_label || "Похожая тема не найдена"}
                      </span>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          ) : (
            renderBulletSection(
              "Динамика относительно прошлого выпуска",
              briefing.trend_vs_previous,
              "Сравнение с предыдущим выпуском недоступно."
            )
          )}

          <section className="card" id="top-topics">
            <h2>Топ-темы</h2>
            <div className="topics-col">
              {briefing.top_topics.map((topic, index) => {
                const comments = getTopicComments(topic, data);
                const questionComments = getTopicQuestionComments(topic, data);
                const positions = getOrderedPositions(topic);
                const hasPositions = positions.length > 0;
                const trend = trendMap.get(topic.cluster_key);

                return (
                  <article className="topic-item full" key={topic.cluster_key} id={`topic-${topic.cluster_key}`}>
                    <h3>
                      {index + 1}. {topic.label}
                    </h3>
                    <p className="muted">
                      Размер: {topic.size_count} | Доля: {formatPercent(topic.share_pct)}
                    </p>
                    <div className="chips">
                      <span className="chip neutral">Confidence {topic.assignment_confidence.toFixed(2)}</span>
                      {topic.ambiguous_share_pct > 0 ? (
                        <span className="chip disagree">Ambiguous {topic.ambiguous_share_pct.toFixed(1)}%</span>
                      ) : null}
                    </div>
                    <p>
                      <strong>Настроение:</strong> {topic.sentiment}
                      {topic.emotion_tags.length > 0 ? ` (${topic.emotion_tags.join(", ")})` : ""}
                    </p>
                    {trend ? <p className="muted"><strong>История темы:</strong> {trend.summary}</p> : null}
                    {topic.soft_assignment_notes.length > 0 ? (
                      <ul className="quote-list compact">
                        {topic.soft_assignment_notes.map((note, noteIndex) => (
                          <li key={`${topic.cluster_key}-note-${noteIndex}`}>{note}</li>
                        ))}
                      </ul>
                    ) : null}

                    {hasPositions ? (
                      <>
                        <p className="muted mt"><strong>Позиции в кластере:</strong> {positions.length}</p>
                        {positions.map((position) => (
                          <details className="expandable" key={`${topic.cluster_key}-${position.key}`}>
                            <summary>
                              <b>{position.title}</b> — {formatPercent(position.pct)} ({position.count}/{topic.size_count})
                            </summary>
                            {position.summary ? <p className="muted">{position.summary}</p> : null}
                            {position.comments.length > 0 ? (
                              <ul className="quote-list expanded">
                                {position.comments.map((comment, commentIndex) => (
                                  <li key={`${position.key}-${commentIndex}-${comment.slice(0, 24)}`}>{comment}</li>
                                ))}
                              </ul>
                            ) : (
                              <p className="muted">Нет комментариев.</p>
                            )}
                          </details>
                        ))}
                      </>
                    ) : (
                      <>
                        <p className="muted mt"><strong>Комментарии по теме:</strong> {comments.length}</p>
                        {renderQuoteList(comments, 3, "Показать все комментарии (еще {count})")}

                        <p className="muted mt"><strong>Вопросы по теме:</strong> {questionComments.length}</p>
                        {renderQuoteList(questionComments, 1, "Показать все вопросы (еще {count})")}
                      </>
                    )}
                  </article>
                );
              })}
            </div>
          </section>

          <section className="card">
            <div className="section-head compact">
              <h2>Комментарии с несогласием с позицией {data.author_name}</h2>
              <Link to="/videos" className="text-link">К списку видео</Link>
            </div>

            {briefing.author_disagreement_comments.length > 0 ? (
              <details className="expandable">
                <summary>
                  Показать все комментарии ({briefing.author_disagreement_comments.length})
                </summary>
                <ul className="quote-list expanded">
                  {briefing.author_disagreement_comments.map((comment, index) => (
                    <li key={`${index}-${comment.slice(0, 24)}`}>{comment}</li>
                  ))}
                </ul>
              </details>
            ) : (
              <p className="muted">Явных комментариев с несогласием не найдено.</p>
            )}
          </section>
        </>
      ) : null}
    </>
  );
}
