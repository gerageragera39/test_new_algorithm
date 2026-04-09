import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { getVideosStatuses } from "../lib/api";
import { formatDateTime } from "../lib/format";
import type { VideoStatusRow } from "../types/api";
import { StatusPill } from "../components/StatusPill";

/**
 * Videos list page. Displays all tracked videos with their pipeline processing
 * statuses, progress bars, and links to reports. Auto-refreshes every 5 seconds.
 */
export function VideosPage() {
  const [items, setItems] = useState<VideoStatusRow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function load() {
      try {
        const rows = await getVideosStatuses();
        if (isMounted) {
          setItems(rows);
          setError(null);
        }
      } catch (loadError) {
        if (isMounted) {
          const message = loadError instanceof Error ? loadError.message : "Ошибка загрузки статусов";
          setError(message);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    }

    void load();
    const interval = window.setInterval(() => {
      void load();
    }, 5000);

    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  const runningCount = useMemo(() => items.filter((item) => item.run_status === "running").length, [items]);

  return (
    <section className="card">
      <div className="section-head compact">
        <div>
          <p className="eyebrow">Pipeline Monitor</p>
          <h1>Видео и статусы обработки</h1>
        </div>
        <p className="muted">В работе: {runningCount}</p>
      </div>

      {error ? <div className="notice error">{error}</div> : null}
      {isLoading ? <p className="muted">Загрузка...</p> : null}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Дата</th>
              <th>Видео</th>
              <th>Статус</th>
              <th>Анализ тем</th>
              <th>Обращения</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.video_id}>
                <td className="mono">{formatDateTime(item.published_at)}</td>
                <td>
                  <a href={`https://www.youtube.com/watch?v=${item.youtube_video_id}`} target="_blank" rel="noreferrer">
                    {item.title}
                  </a>
                  <div className="muted mono">{item.youtube_video_id}</div>
                </td>
                <td>
                  <StatusPill status={item.run_status} label={item.run_status_text || "-"} />
                  {item.run_status === "running" && item.stage_total > 0 ? (
                    <div className="progress-row">
                      <progress value={item.progress_pct} max={100} />
                      <span>{item.progress_pct}%</span>
                    </div>
                  ) : null}
                </td>
                <td>
                  {item.has_report ? (
                    <Link to={`/reports/${item.youtube_video_id}`} className="text-link strong">
                      Открыть
                    </Link>
                  ) : (
                    <span className="muted">-</span>
                  )}
                </td>
                <td>
                  {item.has_appeal_report ? (
                    <Link to={`/appeal/${item.youtube_video_id}`} className="text-link strong">
                      Открыть
                    </Link>
                  ) : item.appeal_run_status === "running" ? (
                    <StatusPill status="running" label="Анализ..." />
                  ) : (
                    <span className="muted">-</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
