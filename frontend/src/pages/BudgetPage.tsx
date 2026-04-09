import { useEffect, useMemo, useState } from "react";

import {
  getBudget,
  getRuntimeSettings,
  getSetupStatus,
  updateRuntimeSettings,
  updateSetup,
} from "../lib/api";
import { formatDateTime, formatMoney } from "../lib/format";
import type {
  BudgetEntry,
  BudgetUsageResponse,
  RuntimeSettingsResponse,
  RuntimeSettingsUpdateRequest,
  SetupStatusResponse,
  SetupUpdateRequest,
} from "../types/api";

/** Local form state for runtime settings. */
interface RuntimeFormState {
  beat_enabled: boolean;
  beat_time_kyiv: string;
  author_name: string;
  openai_chat_model: string;
  embedding_mode: string;
  local_embedding_model: string;
  cluster_max_count: number;
  max_comments_per_video: number;
  youtube_include_replies: boolean;
  openai_enable_polish_call: boolean;
}

interface SetupFormState {
  youtube_playlist_id: string;
  youtube_oauth_client_id: string;
  youtube_oauth_client_secret: string;
  youtube_oauth_refresh_token: string;
}

interface AggregatedSpendRow {
  key: string;
  label: string;
  pipeline: string;
  provider: string;
  models: string[];
  cost: number;
  input: number;
  output: number;
  calls: number;
}

function describeBudgetTask(task: string, provider: string): { label: string; pipeline: string } {
  const normalizedTask = (task || "").trim().toLowerCase();
  const normalizedProvider = (provider || "").trim().toLowerCase();

  if (normalizedProvider === "openai_embedding" || normalizedTask === "comment_embeddings") {
    return { label: "Эмбеддинги комментариев", pipeline: "Topic Intelligence" };
  }

  switch (normalizedTask) {
    case "cluster_labeling":
      return { label: "LLM-маркировка тем", pipeline: "Topic Intelligence" };
    case "cluster_title_naming":
      return { label: "LLM-название темы", pipeline: "Topic Intelligence" };
    case "position_naming":
      return { label: "LLM-название позиции аудитории", pipeline: "Topic Intelligence" };
    case "moderation_borderline":
      return { label: "Пограничная модерация", pipeline: "Moderation" };
    case "briefing_polish":
      return { label: "Полировка итогового отчёта", pipeline: "Topic Intelligence" };
    case "appeal_unified":
      return { label: "Единая appeal-классификация", pipeline: "Appeal Analytics" };
    case "question_refiner":
      return { label: "Уточнение конструктивных вопросов", pipeline: "Appeal Analytics" };
    case "toxic_target_classification":
      return { label: "Определение цели оскорбления", pipeline: "Appeal Analytics" };
    case "json_generation":
      return { label: "Генерация JSON-ответа", pipeline: "Service Utility" };
    default:
      return {
        label: normalizedTask ? normalizedTask.replace(/_/g, " ") : "Неизвестный этап",
        pipeline: normalizedProvider === "openai_chat" ? "OpenAI Chat" : "Service Utility"
      };
  }
}

function aggregateSpend(
  entries: BudgetEntry[] | undefined,
  keyBuilder: (entry: BudgetEntry) => string,
  labelBuilder: (entry: BudgetEntry) => { label: string; pipeline: string }
): AggregatedSpendRow[] {
  if (!entries || entries.length === 0) {
    return [];
  }

  const grouped = new Map<string, AggregatedSpendRow>();
  for (const entry of entries) {
    const key = keyBuilder(entry);
    const descriptor = labelBuilder(entry);
    const current = grouped.get(key) ?? {
      key,
      label: descriptor.label,
      pipeline: descriptor.pipeline,
      provider: entry.provider,
      models: [],
      cost: 0,
      input: 0,
      output: 0,
      calls: 0
    };

    current.cost += entry.estimated_cost_usd;
    current.input += entry.tokens_input;
    current.output += entry.tokens_output;
    current.calls += 1;
    if (!current.models.includes(entry.model)) {
      current.models.push(entry.model);
      current.models.sort((a, b) => a.localeCompare(b));
    }
    grouped.set(key, current);
  }

  return [...grouped.values()].sort((a, b) => b.cost - a.cost || a.label.localeCompare(b.label));
}

/** Convert a RuntimeSettingsResponse into a form-friendly state. */
function toFormState(state: RuntimeSettingsResponse): RuntimeFormState {
  return {
    beat_enabled: state.beat_enabled,
    beat_time_kyiv: state.beat_time_kyiv,
    author_name: state.author_name,
    openai_chat_model: state.openai_chat_model,
    embedding_mode: state.embedding_mode,
    local_embedding_model: state.local_embedding_model,
    cluster_max_count: state.cluster_max_count,
    max_comments_per_video: state.max_comments_per_video,
    youtube_include_replies: state.youtube_include_replies,
    openai_enable_polish_call: state.openai_enable_polish_call,
  };
}

function emptySetupForm(): SetupFormState {
  return {
    youtube_playlist_id: "",
    youtube_oauth_client_id: "",
    youtube_oauth_client_secret: "",
    youtube_oauth_refresh_token: "",
  };
}

/**
 * Budget and runtime settings page. Shows OpenAI spend summary,
 * daily call log table, and editable scheduler controls.
 */
export function BudgetPage() {
  const [snapshot, setSnapshot] = useState<BudgetUsageResponse | null>(null);
  const [runtime, setRuntime] = useState<RuntimeSettingsResponse | null>(null);
  const [setupStatus, setSetupStatus] = useState<SetupStatusResponse | null>(null);
  const [formState, setFormState] = useState<RuntimeFormState | null>(null);
  const [setupForm, setSetupForm] = useState<SetupFormState>(emptySetupForm);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isSavingSetup, setIsSavingSetup] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const stageSpendRows = useMemo(
    () =>
      aggregateSpend(
        snapshot?.entries,
        (entry) => `${entry.provider}::${entry.task || "unknown"}`,
        (entry) => describeBudgetTask(entry.task, entry.provider)
      ),
    [snapshot]
  );

  const modelSpendRows = useMemo(
    () =>
      aggregateSpend(
        snapshot?.entries,
        (entry) => `${entry.provider}::${entry.model}`,
        (entry) => ({
          label: entry.model,
          pipeline: entry.provider === "openai_embedding" ? "Embeddings" : "OpenAI Chat"
        })
      ),
    [snapshot]
  );

  const pipelineSpendRows = useMemo(
    () =>
      aggregateSpend(
        snapshot?.entries,
        (entry) => describeBudgetTask(entry.task, entry.provider).pipeline,
        (entry) => {
          const descriptor = describeBudgetTask(entry.task, entry.provider);
          return { label: descriptor.pipeline, pipeline: "Pipeline Summary" };
        }
      ),
    [snapshot]
  );

  const chatSpend = useMemo(
    () =>
      snapshot?.entries
        .filter((entry) => entry.provider === "openai_chat")
        .reduce((sum, entry) => sum + entry.estimated_cost_usd, 0) ?? 0,
    [snapshot]
  );

  const embeddingSpend = useMemo(
    () =>
      snapshot?.entries
        .filter((entry) => entry.provider === "openai_embedding")
        .reduce((sum, entry) => sum + entry.estimated_cost_usd, 0) ?? 0,
    [snapshot]
  );

  const topStage = stageSpendRows[0] ?? null;

  useEffect(() => {
    let isMounted = true;

    async function load() {
      try {
        const [budgetData, runtimeData] = await Promise.all([getBudget(), getRuntimeSettings()]);
        let setupData: SetupStatusResponse | null = null;
        try {
          setupData = await getSetupStatus();
        } catch {
          // `/app/setup/*` endpoints exist only in desktop runtime.
          setupData = null;
        }
        if (isMounted) {
          setSnapshot(budgetData);
          setRuntime(runtimeData);
          setSetupStatus(setupData);
          setFormState(toFormState(runtimeData));
          setError(null);
        }
      } catch (loadError) {
        if (isMounted) {
          const message = loadError instanceof Error ? loadError.message : "Ошибка загрузки настроек";
          setError(message);
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

  async function handleSave() {
    if (!formState) {
      return;
    }
    setIsSaving(true);
    setSaveMessage(null);
    setError(null);

    const payload: RuntimeSettingsUpdateRequest = {
      beat_enabled: formState.beat_enabled,
      beat_time_kyiv: formState.beat_time_kyiv,
      author_name: formState.author_name,
      openai_chat_model: formState.openai_chat_model,
      embedding_mode: formState.embedding_mode,
      local_embedding_model: formState.local_embedding_model,
      cluster_max_count: formState.cluster_max_count,
      max_comments_per_video: formState.max_comments_per_video,
      youtube_include_replies: formState.youtube_include_replies,
      openai_enable_polish_call: formState.openai_enable_polish_call,
    };

    try {
      const updated = await updateRuntimeSettings(payload);
      setRuntime(updated);
      setFormState(toFormState(updated));
      setSaveMessage("Настройки сохранены.");
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "Ошибка сохранения";
      setError(message);
    } finally {
      setIsSaving(false);
    }
  }

  async function handleSaveSetupSecrets() {
    const payload: SetupUpdateRequest = {
      youtube_playlist_id: setupForm.youtube_playlist_id.trim() || undefined,
      youtube_oauth_client_id: setupForm.youtube_oauth_client_id.trim() || undefined,
      youtube_oauth_client_secret: setupForm.youtube_oauth_client_secret.trim() || undefined,
      youtube_oauth_refresh_token: setupForm.youtube_oauth_refresh_token.trim() || undefined,
    };
    if (Object.values(payload).every((value) => value == null || value === "")) {
      setSaveMessage("Нет новых OAuth/playlist значений для сохранения.");
      return;
    }

    setIsSavingSetup(true);
    setError(null);
    setSaveMessage(null);
    try {
      const updated = await updateSetup(payload);
      setSetupStatus(updated);
      setSetupForm(emptySetupForm());
      setSaveMessage("OAuth/playlist параметры сохранены.");
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "Ошибка сохранения OAuth";
      setError(message);
    } finally {
      setIsSavingSetup(false);
    }
  }

  function updateForm(patch: Partial<RuntimeFormState>) {
    setFormState((prev) => (prev ? { ...prev, ...patch } : prev));
  }

  function updateSetupForm(patch: Partial<SetupFormState>) {
    setSetupForm((prev) => ({ ...prev, ...patch }));
  }

  return (
    <>
      <section className="card accent">
        <div className="section-head compact">
          <div>
            <p className="eyebrow">Runtime Controls</p>
            <h1>Dashboard</h1>
          </div>
          <div className="mode-chip">Часовой пояс: Europe/Kyiv</div>
        </div>

        {error ? <div className="notice error">{error}</div> : null}
        {saveMessage ? <div className="notice success">{saveMessage}</div> : null}
        {isLoading ? <p className="muted">Загрузка...</p> : null}

        {formState ? (
          <div className="settings-grid">
            <label className="settings-field checkbox-field">
              <input
                type="checkbox"
                checked={formState.beat_enabled}
                onChange={(event) => updateForm({ beat_enabled: event.target.checked })}
              />
              <span>Ежедневный автозапуск через beat</span>
            </label>

            <label className="settings-field">
              <span>Время запуска (Киев)</span>
              <input
                type="time"
                value={formState.beat_time_kyiv}
                onChange={(event) => updateForm({ beat_time_kyiv: event.target.value })}
              />
            </label>

            <label className="settings-field">
              <span>Имя автора канала</span>
              <input
                type="text"
                value={formState.author_name}
                onChange={(event) => updateForm({ author_name: event.target.value })}
                placeholder="Введите имя автора"
              />
            </label>

            <label className="settings-field">
              <span>Модель OpenAI</span>
              <select
                value={formState.openai_chat_model}
                onChange={(event) => updateForm({ openai_chat_model: event.target.value })}
              >
                <option value="gpt-4o-mini">gpt-4o-mini (быстрая, дешёвая)</option>
                <option value="gpt-4o">gpt-4o (мощная)</option>
                <option value="gpt-5-mini">gpt-5-mini (новая, баланс)</option>
                <option value="gpt-5.2">gpt-5.2 (самая мощная)</option>
              </select>
            </label>

            <label className="settings-field">
              <span>Режим эмбеддингов</span>
              <select
                value={formState.embedding_mode}
                onChange={(event) => updateForm({ embedding_mode: event.target.value })}
              >
                <option value="local">Локальный (бесплатно)</option>
                <option value="openai">OpenAI (платный)</option>
              </select>
            </label>

            {formState.embedding_mode === "local" ? (
              <label className="settings-field">
                <span>Локальная embedding-модель</span>
                <select
                  value={formState.local_embedding_model}
                  onChange={(event) => updateForm({ local_embedding_model: event.target.value })}
                >
                  <option value="Qwen/Qwen3-Embedding-0.6B">Qwen/Qwen3-Embedding-0.6B</option>
                  <option value="BAAI/bge-m3">BAAI/bge-m3 (рекомендуется)</option>
                  <option value="intfloat/multilingual-e5-large">intfloat/multilingual-e5-large</option>
                </select>
              </label>
            ) : null}

            <label className="settings-field">
              <span>Макс. тем (0 = без лимита)</span>
              <input
                type="number"
                min={0}
                max={50}
                value={formState.cluster_max_count}
                onChange={(event) => updateForm({ cluster_max_count: Number(event.target.value) })}
              />
            </label>

            <label className="settings-field">
              <span>Макс. комментариев на видео</span>
              <input
                type="number"
                min={50}
                max={10000}
                step={50}
                value={formState.max_comments_per_video}
                onChange={(event) => updateForm({ max_comments_per_video: Number(event.target.value) })}
              />
            </label>

            <label className="settings-field checkbox-field">
              <input
                type="checkbox"
                checked={formState.youtube_include_replies}
                onChange={(event) => updateForm({ youtube_include_replies: event.target.checked })}
              />
              <span>Включать ответы на комментарии</span>
            </label>

            <label className="settings-field checkbox-field">
              <input
                type="checkbox"
                checked={formState.openai_enable_polish_call}
                onChange={(event) => updateForm({ openai_enable_polish_call: event.target.checked })}
              />
              <span>Полировка итогового отчёта (доп. вызов LLM)</span>
            </label>

          </div>
        ) : null}

        <div className="actions-row">
          <button type="button" className="btn primary" onClick={handleSave} disabled={isSaving || !formState}>
            {isSaving ? "Сохранение..." : "Сохранить настройки"}
          </button>
        </div>

        {runtime?.updated_at ? (
          <p className="muted mt">Последнее обновление: {formatDateTime(runtime.updated_at)}</p>
        ) : null}
      </section>

      {setupStatus ? (
      <section className="card">
        <div className="section-head compact">
          <div>
            <p className="eyebrow">YouTube Moderation OAuth</p>
            <h2>Desktop secrets и moderation credentials</h2>
          </div>
        </div>

        <p className="muted">
          Эти поля не обязательны. Но если они заполнены, desktop-приложение сможет
          делать channel-level ban/unban через YouTube OAuth.
        </p>

        <div className="settings-grid">
          <label className="settings-field">
            <span>YOUTUBE_PLAYLIST_ID</span>
            <input
              type="text"
              value={setupForm.youtube_playlist_id}
              onChange={(event) => updateSetupForm({ youtube_playlist_id: event.target.value })}
              placeholder={setupStatus?.has_playlist_id ? "Уже сохранён" : "PL..."}
            />
          </label>

          <label className="settings-field">
            <span>YOUTUBE_OAUTH_CLIENT_ID</span>
            <input
              type="password"
              autoComplete="new-password"
              value={setupForm.youtube_oauth_client_id}
              onChange={(event) =>
                updateSetupForm({ youtube_oauth_client_id: event.target.value })
              }
              placeholder={
                setupStatus?.has_youtube_oauth_client_id ? "Уже сохранён" : "Необязательно"
              }
            />
          </label>

          <label className="settings-field">
            <span>YOUTUBE_OAUTH_CLIENT_SECRET</span>
            <input
              type="password"
              autoComplete="new-password"
              value={setupForm.youtube_oauth_client_secret}
              onChange={(event) =>
                updateSetupForm({ youtube_oauth_client_secret: event.target.value })
              }
              placeholder={
                setupStatus?.has_youtube_oauth_client_secret ? "Уже сохранён" : "Необязательно"
              }
            />
          </label>

          <label className="settings-field">
            <span>YOUTUBE_OAUTH_REFRESH_TOKEN</span>
            <textarea
              value={setupForm.youtube_oauth_refresh_token}
              onChange={(event) =>
                updateSetupForm({ youtube_oauth_refresh_token: event.target.value })
              }
              placeholder={
                setupStatus?.has_youtube_oauth_refresh_token
                  ? "Уже сохранён"
                  : "Необязательно"
              }
              rows={4}
            />
          </label>
        </div>

        <div className="actions-row">
          <button
            type="button"
            className="btn secondary"
            onClick={handleSaveSetupSecrets}
            disabled={isSavingSetup}
          >
            {isSavingSetup ? "Сохранение..." : "Сохранить OAuth / playlist"}
          </button>
        </div>
      </section>
      ) : null}

      {snapshot ? (
        <section className="card accent">
          <div className="section-head compact">
            <div>
              <p className="eyebrow">OpenAI Cost Control</p>
              <h2>Бюджет и токены</h2>
            </div>
            <div className="mode-chip">Дата: {snapshot.usage_date ?? "-"}</div>
          </div>

          <div className="stats-grid">
            <article className="stat">
              <span>Потрачено сегодня</span>
              <strong>{formatMoney(snapshot.spent_usd)}</strong>
              <small>все OpenAI-вызовы за день</small>
            </article>
            <article className="stat">
              <span>Токены сегодня</span>
              <strong>{snapshot.tokens_used}</strong>
              <small>input + output</small>
            </article>
            <article className="stat">
              <span>Chat models</span>
              <strong>{formatMoney(chatSpend)}</strong>
              <small>вызовы chat/completions</small>
            </article>
            <article className="stat">
              <span>Embeddings</span>
              <strong>{formatMoney(embeddingSpend)}</strong>
              <small>векторизация комментариев</small>
            </article>
          </div>
        </section>
      ) : null}

      {pipelineSpendRows.length > 0 ? (
        <section className="card">
          <div className="section-head compact">
            <div>
              <p className="eyebrow">Management View</p>
              <h2>Сводка по пайплайнам</h2>
            </div>
            {topStage ? (
              <div className="mode-chip">
                Самый дорогой этап: {topStage.label} · {formatMoney(topStage.cost, 6)}
              </div>
            ) : null}
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Пайплайн</th>
                  <th>Модели</th>
                  <th>Вызовов</th>
                  <th>Вход</th>
                  <th>Выход</th>
                  <th>Сумма</th>
                </tr>
              </thead>
              <tbody>
                {pipelineSpendRows.map((row) => (
                  <tr key={row.key}>
                    <td>{row.label}</td>
                    <td>{row.models.join(", ") || "—"}</td>
                    <td>{row.calls}</td>
                    <td>{row.input}</td>
                    <td>{row.output}</td>
                    <td>{formatMoney(row.cost, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {stageSpendRows.length > 0 ? (
        <section className="card">
          <div className="section-head compact">
            <div>
              <p className="eyebrow">Primary Breakdown</p>
              <h2>Расход по этапам</h2>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Этап</th>
                  <th>Пайплайн</th>
                  <th>Модели</th>
                  <th>Вызовов</th>
                  <th>Вход</th>
                  <th>Выход</th>
                  <th>Сумма</th>
                </tr>
              </thead>
              <tbody>
                {stageSpendRows.map((row) => (
                  <tr key={row.key}>
                    <td>{row.label}</td>
                    <td>{row.pipeline}</td>
                    <td>{row.models.join(", ") || "—"}</td>
                    <td>{row.calls}</td>
                    <td>{row.input}</td>
                    <td>{row.output}</td>
                    <td>{formatMoney(row.cost, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {modelSpendRows.length > 0 ? (
        <section className="card">
          <div className="section-head compact">
            <div>
              <p className="eyebrow">Secondary Breakdown</p>
              <h2>Расход по моделям</h2>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Провайдер</th>
                  <th>Модель</th>
                  <th>Вызовов</th>
                  <th>Вход</th>
                  <th>Выход</th>
                  <th>Сумма</th>
                </tr>
              </thead>
              <tbody>
                {modelSpendRows.map((row) => (
                  <tr key={row.key}>
                    <td>{row.provider}</td>
                    <td>{row.label}</td>
                    <td>{row.calls}</td>
                    <td>{row.input}</td>
                    <td>{row.output}</td>
                    <td>{formatMoney(row.cost, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="card">
        <details>
          <summary className="text-link strong" style={{ cursor: "pointer" }}>
            Показать технический лог отдельных запросов
          </summary>
          <p className="muted mt" style={{ marginBottom: 12 }}>
            Этот блок нужен для отладки. Основной ориентир для отчётности — таблица расходов по этапам выше.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Провайдер</th>
                  <th>Модель</th>
                  <th>Задача</th>
                  <th>Вход</th>
                  <th>Выход</th>
                  <th>Стоимость</th>
                  <th>Время</th>
                </tr>
              </thead>
              <tbody>
                {snapshot?.entries.map((row, index) => (
                  <tr key={`${row.created_at}-${index}`}>
                    <td>{row.provider}</td>
                    <td>{row.model}</td>
                    <td>{row.task}</td>
                    <td>{row.tokens_input}</td>
                    <td>{row.tokens_output}</td>
                    <td>{formatMoney(row.estimated_cost_usd, 6)}</td>
                    <td>{formatDateTime(row.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>
    </>
  );
}
