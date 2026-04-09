import { type ChangeEvent, type FormEvent, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  ApiError,
  getBudget,
  getHealth,
  getLatestReport,
  getQueueSnapshot,
  runAppealAnalytics,
  runLatest,
  runVideo,
} from "../lib/api";
import { formatDateTime, formatMoney, formatPercent } from "../lib/format";
import type {
  BudgetUsageResponse,
  HealthResponse,
  ReportResponse,
  QueueSnapshotResponse,
} from "../types/api";
import { StatusPill } from "../components/StatusPill";

function describeQueuePayload(payload: Record<string, unknown>): string {
  const rawUrl = payload.video_url ?? payload.url ?? payload.videoUrl;
  if (typeof rawUrl === "string" && rawUrl.trim()) {
    return rawUrl.trim();
  }
  const rawId = payload.video_id ?? payload.videoId;
  if (typeof rawId === "string" && rawId.trim()) {
    return `Video ID: ${rawId.trim()}`;
  }
  return "Без дополнительных данных";
}

/**
 * Main dashboard page. Displays health status, the latest report summary,
 * OpenAI budget overview, and controls to trigger pipeline runs.
 */
export function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [budget, setBudget] = useState<BudgetUsageResponse | null>(null);
  const [queueSnapshot, setQueueSnapshot] = useState<QueueSnapshotResponse | null>(null);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [isQueueLoading, setIsQueueLoading] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [videoUrl, setVideoUrl] = useState("");
  const [skipFiltering, setSkipFiltering] = useState(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [appealUrl, setAppealUrl] = useState("");
  const [appealMessage, setAppealMessage] = useState<string | null>(null);
  const [appealError, setAppealError] = useState<string | null>(null);
  const [isAppealSubmitting, setIsAppealSubmitting] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function load() {
      setIsLoading(true);
      try {
        const healthData = await getHealth();
        if (!isMounted) {
          return;
        }
        setHealth(healthData);

        try {
          const savedFilter = window.localStorage.getItem("dashboard.skipFiltering");
          if (savedFilter === "true") {
            setSkipFiltering(true);
          }
        } catch {
          // Ignore browser storage errors.
        }

        try {
          const latest = await getLatestReport();
          if (isMounted) {
            setReport(latest);
          }
        } catch (error) {
          if (error instanceof ApiError && error.status === 404) {
            if (isMounted) {
              setReport(null);
            }
          } else {
            throw error;
          }
        }

        const budgetData = await getBudget();
        if (isMounted) {
          setBudget(budgetData);
        }
      } catch (error) {
        if (isMounted) {
          const message = error instanceof Error ? error.message : "Не удалось загрузить дашборд";
          setActionError(message);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    void load();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    let isMounted = true;

    async function loadQueue() {
      setIsQueueLoading(true);
      try {
        const snapshot = await getQueueSnapshot();
        if (!isMounted) {
          return;
        }
        setQueueSnapshot(snapshot);
        setQueueError(null);
      } catch (error) {
        if (!isMounted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Не удалось загрузить очередь";
        setQueueError(message);
      } finally {
        if (isMounted) {
          setIsQueueLoading(false);
        }
      }
    }

    void loadQueue();
    const interval = window.setInterval(() => {
      void loadQueue();
    }, 5000);

    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  const topTopics = useMemo(() => report?.briefing.top_topics.slice(0, 4) ?? [], [report]);

  function handleSkipFilteringChange(event: ChangeEvent<HTMLInputElement>) {
    const nextValue = event.target.checked;
    setSkipFiltering(nextValue);
    try {
      window.localStorage.setItem("dashboard.skipFiltering", String(nextValue));
    } catch {
      // Ignore browser storage errors.
    }
  }

  async function handleRunLatest() {
    setIsSubmitting(true);
    setActionError(null);
    setActionMessage(null);

    try {
      const response = await runLatest({ skipFiltering });
      setActionMessage(`Задача запущена: ${response.task_id}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Ошибка запуска";
      setActionError(message);
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleRunVideo(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!videoUrl.trim()) {
      setActionError("Введите ссылку на видео.");
      return;
    }

    setIsSubmitting(true);
    setActionError(null);
    setActionMessage(null);

    try {
      const response = await runVideo(videoUrl.trim(), { skipFiltering });
      setActionMessage(`Задача по видео запущена: ${response.task_id}`);
      setVideoUrl("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Ошибка запуска";
      setActionError(message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <>
      <section className="card hero">
        <div className="section-head">
          <div>
            <p className="eyebrow">Editorial Intelligence</p>
            <h1>Панель управления анализом комментариев</h1>
          </div>
          <div className="mode-chip">OpenAI: {health?.openai_endpoint_host ?? "..."}</div>
        </div>
        <p className="lead">
          Запускайте pipeline по последнему видео или по ссылке. Все результаты, ключевые категории,
          комментарии и бюджет доступны в одном интерфейсе.
        </p>

        <div className="toggle-row">
          <label htmlFor="toggle-skip-filtering" className="toggle-control">
            <input
              id="toggle-skip-filtering"
              type="checkbox"
              checked={skipFiltering}
              onChange={handleSkipFilteringChange}
              disabled={isSubmitting}
            />
            <span>Отключить фильтрацию комментариев</span>
          </label>
          <p className="muted toggle-hint">
            При включении все комментарии попадут в анализ без предварительной фильтрации.
          </p>
        </div>

        <div className="actions-row">
          <button type="button" className="btn primary" disabled={isSubmitting} onClick={handleRunLatest}>
            {isSubmitting ? "Запуск..." : "Запустить анализ последнего видео"}
          </button>
        </div>

        <form className="video-form" onSubmit={handleRunVideo}>
          <label htmlFor="video-url" className="field-label">
            Анализ по ссылке YouTube
          </label>
          <div className="field-grid">
            <input
              id="video-url"
              type="url"
              value={videoUrl}
              onChange={(event) => setVideoUrl(event.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              required
            />
            <button type="submit" className="btn secondary" disabled={isSubmitting}>
              Запустить
            </button>
          </div>
        </form>

        {actionMessage && <div className="notice success">{actionMessage}</div>}
        {actionError && <div className="notice error">{actionError}</div>}
      </section>

      <section className="card hero">
        <div className="section-head">
          <div>
            <p className="eyebrow">Comment Classification</p>
            <h2>Аналитика обращений</h2>
          </div>
        </div>
        <p className="lead">
          Классификация комментариев по категориям: токсичные, спам,
          конструктивная критика, обращения к автору и конструктивные вопросы.
        </p>

        <div className="actions-row">
          <button
            type="button"
            className="btn primary"
            disabled={isAppealSubmitting}
            onClick={async () => {
              setIsAppealSubmitting(true);
              setAppealError(null);
              setAppealMessage(null);
              try {
                const response = await runAppealAnalytics();
                setAppealMessage(`Аналитика обращений запущена: ${response.task_id}`);
              } catch (err) {
                const message = err instanceof Error ? err.message : "Ошибка запуска";
                setAppealError(message);
              } finally {
                setIsAppealSubmitting(false);
              }
            }}
          >
            {isAppealSubmitting ? "Запуск..." : "Запустить анализ последнего видео"}
          </button>
        </div>

        <form
          className="video-form"
          onSubmit={async (event) => {
            event.preventDefault();
            if (!appealUrl.trim()) {
              setAppealError("Введите ссылку на видео.");
              return;
            }
            setIsAppealSubmitting(true);
            setAppealError(null);
            setAppealMessage(null);
            try {
              const response = await runAppealAnalytics(appealUrl.trim());
              setAppealMessage(`Аналитика обращений запущена: ${response.task_id}`);
              setAppealUrl("");
            } catch (err) {
              const message = err instanceof Error ? err.message : "Ошибка запуска";
              setAppealError(message);
            } finally {
              setIsAppealSubmitting(false);
            }
          }}
        >
          <label htmlFor="appeal-url" className="field-label">
            Ссылка на видео YouTube
          </label>
          <div className="field-grid">
            <input
              id="appeal-url"
              type="url"
              value={appealUrl}
              onChange={(e) => setAppealUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              required
            />
            <button type="submit" className="btn primary" disabled={isAppealSubmitting}>
              {isAppealSubmitting ? "Запуск..." : "Запустить аналитику"}
            </button>
          </div>
        </form>

        {appealMessage && <div className="notice success">{appealMessage}</div>}
        {appealError && <div className="notice error">{appealError}</div>}
      </section>

      <section className="card queue-card">
        <div className="section-head compact">
          <div>
            <p className="eyebrow">Local queue</p>
            <h2>Очередь задач</h2>
          </div>
          <p className="muted">В очереди: {queueSnapshot?.queued.length ?? 0}</p>
        </div>

        {queueError ? <div className="notice error">{queueError}</div> : null}
        {isQueueLoading && !queueSnapshot ? (
          <p className="muted">Загрузка очереди...</p>
        ) : (
          <div className="queue-grid">
            <div className="queue-column">
              <h3>Выполняется сейчас</h3>
              {queueSnapshot?.current ? (
                <div className="queue-list-item">
                  <div className="queue-meta">
                    <strong>{queueSnapshot.current.kind}</strong>
                    <StatusPill status={queueSnapshot.current.status} label={queueSnapshot.current.status || "-"} />
                  </div>
                  <p className="muted mono">{queueSnapshot.current.id}</p>
                  <p className="muted">
                    Начато:{" "}
                    {queueSnapshot.current.started_at
                      ? formatDateTime(queueSnapshot.current.started_at)
                      : "ожидается"}
                  </p>
                  <p className="queue-detail">{describeQueuePayload(queueSnapshot.current.payload)}</p>
                </div>
              ) : (
                <p className="muted">Сейчас нет активных задач</p>
              )}
            </div>

            <div className="queue-column">
              <h3>Ожидают</h3>
              {queueSnapshot?.queued.length ? (
                <ul className="queue-list">
                  {queueSnapshot.queued.map((job) => (
                    <li className="queue-list-item" key={job.id}>
                      <div className="queue-meta">
                        <strong>{job.kind}</strong>
                        <StatusPill status={job.status} label={job.status || "queued"} />
                      </div>
                      <p className="muted mono">{job.id}</p>
                      <p className="muted">Добавлено: {formatDateTime(job.created_at)}</p>
                      <p className="queue-detail">{describeQueuePayload(job.payload)}</p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">Очередь пуста</p>
              )}
            </div>
          </div>
        )}
      </section>

      {isLoading ? <section className="card">Загрузка данных...</section> : null}

      {!isLoading && report ? (
        <section className="card">
          <div className="section-head compact">
            <h2>Последний отчет</h2>
            <Link to={`/reports/${report.video_id}`} className="text-link strong">
              Открыть полный отчет
            </Link>
          </div>
          <p className="muted">Видео ID: {report.video_id}</p>
          <p className="summary">{report.briefing.executive_summary}</p>

          <div className="topics-grid">
            {topTopics.map((topic, index) => (
              <article className="topic-item" key={topic.cluster_key}>
                <h3>
                  {index + 1}. {topic.label}
                </h3>
                <p className="muted">
                  Доля {formatPercent(topic.share_pct)} | Взвешенная {formatPercent(topic.weighted_share)}
                </p>
                <p>{topic.description}</p>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {!isLoading && !report ? (
        <section className="card">
          <h2>Отчетов пока нет</h2>
          <p className="muted">После запуска анализа здесь появится последний briefing по выпуску.</p>
        </section>
      ) : null}

      {!isLoading && budget ? (
        <section className="card accent">
          <div className="section-head compact">
            <h2>OpenAI бюджет</h2>
            <Link to="/budget" className="text-link strong">
              Детализация
            </Link>
          </div>
          <div className="stats-grid">
            <article className="stat">
              <span>Потрачено сегодня</span>
              <strong>{formatMoney(budget.spent_usd)}</strong>
            </article>
            <article className="stat">
              <span>Токены сегодня</span>
              <strong>{budget.tokens_used}</strong>
            </article>
          </div>
        </section>
      ) : null}

    </>
  );
}
