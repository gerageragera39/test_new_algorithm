import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";

import { completeSetup, getSetupStatus } from "../lib/api";
import type { SetupStatusResponse } from "../types/api";

export function AppShell() {
  const [setupStatus, setSetupStatus] = useState<SetupStatusResponse | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(true);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [openaiKey, setOpenaiKey] = useState("");
  const [youtubeKey, setYoutubeKey] = useState("");
  const [playlistId, setPlaylistId] = useState("");
  const [isSavingSetup, setIsSavingSetup] = useState(false);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [setupMessage, setSetupMessage] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  const loadStatus = useCallback(async () => {
    setIsLoadingStatus(true);
    setStatusError(null);
    try {
      const status = await getSetupStatus();
      if (isMountedRef.current) {
        setSetupStatus(status);
      }
    } catch (error) {
      if (isMountedRef.current) {
        const message = error instanceof Error ? error.message : "Не удалось загрузить статус настройки.";
        setStatusError(message);
      }
    } finally {
      if (isMountedRef.current) {
        setIsLoadingStatus(false);
      }
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    void loadStatus();
    return () => {
      isMountedRef.current = false;
    };
  }, [loadStatus]);

  const isSetupReady = Boolean(setupStatus?.is_configured);
  const statusRows = [
    { id: "openai", label: "OpenAI API key", ready: setupStatus?.has_openai_api_key ?? false },
    { id: "youtube", label: "YouTube API key", ready: setupStatus?.has_youtube_api_key ?? false },
    { id: "playlist", label: "YouTube playlist ID (можно добавить позже)", ready: setupStatus?.has_playlist_id ?? false },
  ];

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedOpenai = openaiKey.trim();
    const trimmedYoutube = youtubeKey.trim();
    if (!trimmedOpenai || !trimmedYoutube) {
      setSetupError("Оба ключа обязательны для запуска.");
      return;
    }
    setIsSavingSetup(true);
    setSetupError(null);
    setSetupMessage(null);
    try {
      const nextStatus = await completeSetup({
        openai_api_key: trimmedOpenai,
        youtube_api_key: trimmedYoutube,
        youtube_playlist_id: playlistId.trim() || undefined,
      });
      if (!isMountedRef.current) {
        return;
      }
      setSetupStatus(nextStatus);
      setSetupMessage("Ключи сохранены. Система готова.");
      setOpenaiKey("");
      setYoutubeKey("");
      setPlaylistId("");
    } catch (error) {
      if (isMountedRef.current) {
        const message = error instanceof Error ? error.message : "Не удалось сохранить ключи.";
        setSetupError(message);
      }
    } finally {
      if (isMountedRef.current) {
        setIsSavingSetup(false);
      }
    }
  }

  const canSubmit = Boolean(openaiKey.trim() && youtubeKey.trim()) && !isSavingSetup;

  return (
    <div className="app-root">
      <div className="bg-ambient" aria-hidden="true" />
      <header className="topbar">
        <div className="container topbar-inner">
          <NavLink to="/" className="brand">
            <span className="brand-mark" aria-hidden="true" />
            <span className="brand-text">
              <span className="brand-title">YouTube Intel</span>
              <span className="brand-subtitle">Made by Herman Dihtenko</span>
            </span>
          </NavLink>
          {isSetupReady ? (
            <nav className="topnav" aria-label="Навигация">
              <NavLink to="/" end>
                Главная
              </NavLink>
              <NavLink to="/videos">Видео</NavLink>
              <NavLink to="/budget">Dashboard</NavLink>
            </nav>
          ) : null}
        </div>
      </header>

      <main className="page">
        <div className="container">
          {isSetupReady ? (
            <Outlet />
          ) : isLoadingStatus ? (
            <section className="card">
              <p className="muted">Проверка конфигурации...</p>
            </section>
          ) : (
            <section className="card setup-card">
              <div className="section-head">
                <div>
                  <p className="eyebrow">Первичная настройка</p>
                  <h1>Введите OpenAI key, YouTube key и Playlist ID</h1>
                </div>
              </div>
              <p className="lead">
                Перед запуском аналитики сохраните ключи OpenAI, YouTube и при необходимости Playlist ID. Секреты
                зашифрованы и хранятся локально.
              </p>

              {statusError ? (
                <div className="notice error">
                  <p>{statusError}</p>
                  <button type="button" className="btn secondary inline-btn" onClick={() => void loadStatus()}>
                    Повторить
                  </button>
                </div>
              ) : null}
              {setupMessage ? <div className="notice success">{setupMessage}</div> : null}

              <ul className="setup-status-list">
                {statusRows.map((row) => (
                  <li key={row.id}>
                    <span className={`setup-status-dot ${row.ready ? "setup-status-dot-ready" : ""}`} />
                    <span>{row.ready ? `${row.label} добавлен` : `${row.label} ожидается`}</span>
                  </li>
                ))}
              </ul>

              <p className="muted setup-path">
                Конфигурация: {setupStatus?.runtime_env_path ?? "ожидается..."}
              </p>

              <form className="setup-form" onSubmit={handleSubmit}>
                <label className="field-label" htmlFor="openai-key">
                  OpenAI API key
                </label>
                <input
                  id="openai-key"
                  type="password"
                  autoComplete="new-password"
                  value={openaiKey}
                  onChange={(event) => setOpenaiKey(event.target.value)}
                  placeholder="sk-..."
                />

                <label className="field-label" htmlFor="youtube-key">
                  YouTube API key
                </label>
                <input
                  id="youtube-key"
                  type="password"
                  autoComplete="new-password"
                  value={youtubeKey}
                  onChange={(event) => setYoutubeKey(event.target.value)}
                  placeholder="AIza{...}"
                />

                <label className="field-label" htmlFor="playlist-id">
                  YouTube Playlist ID
                </label>
                <input
                  id="playlist-id"
                  type="text"
                  value={playlistId}
                  onChange={(event) => setPlaylistId(event.target.value)}
                  placeholder="PL..."
                />

                <div className="actions-row">
                  <button type="submit" className="btn primary" disabled={!canSubmit}>
                    {isSavingSetup ? "Сохранение..." : "Сохранить данные"}
                  </button>
                </div>
              </form>
              {setupError ? <div className="notice error">{setupError}</div> : null}

              <p className="muted setup-note">
                Секреты сохраняются локально и доступны только вам. После успешной настройки приложение откроет
                полноценный интерфейс.
              </p>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
