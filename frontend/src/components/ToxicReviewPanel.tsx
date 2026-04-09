import { useState } from "react";
import { banUser } from "../lib/api";
import type { ToxicReviewItem } from "../types/api";

interface ToxicReviewPanelProps {
  videoId: string;
  items: ToxicReviewItem[];
  onBanComplete?: () => void;
}

function ConfidenceBadge({ score }: { score: number }) {
  const percentage = Math.round(score * 100);
  const color = score >= 0.75 ? "danger" : score >= 0.6 ? "neutral" : "agree";
  
  return (
    <span className={`chip ${color}`} style={{ fontSize: "0.85rem", marginLeft: 4 }}>
      {percentage}%
    </span>
  );
}

function TargetBadge({ target }: { target: string | null }) {
  if (!target) return null;
  
  const labels: Record<string, string> = {
    author: "Автор",
    guest: "Гость",
    content: "Контент",
    undefined: "Неопр.",
  };
  
  const colors: Record<string, string> = {
    author: "danger",
    guest: "danger",
    content: "danger",
    undefined: "neutral",
  };
  
  return (
    <span className={`chip ${colors[target] || "neutral"}`} style={{ fontSize: "0.75rem", marginLeft: 4 }}>
      {labels[target] || target}
    </span>
  );
}

function ExpandableText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  const WORD_LIMIT = 30;
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

export function ToxicReviewPanel({ videoId, items, onBanComplete }: ToxicReviewPanelProps) {
  const [banning, setBanning] = useState<Set<number>>(new Set());
  const [banned, setBanned] = useState<Set<number>>(new Set());
  const [error, setError] = useState<string | null>(null);

  async function handleBan(item: ToxicReviewItem) {
    if (banning.has(item.comment_id) || banned.has(item.comment_id)) {
      return;
    }

    setBanning(prev => new Set(prev).add(item.comment_id));
    setError(null);

    try {
      const result = await banUser(
        videoId,
        item.comment_id,
        item.author_name || "unknown",
        `Ручной бан: ${item.insult_target || "undefined"}`
      );

      setBanned(prev => new Set(prev).add(item.comment_id));
      
      if (onBanComplete) {
        onBanComplete();
      }

      // Show success message briefly
      setTimeout(() => {
        setError(null);
      }, 3000);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Ошибка бана";
      setError(`Не удалось забанить ${item.author_name}: ${message}`);
    } finally {
      setBanning(prev => {
        const next = new Set(prev);
        next.delete(item.comment_id);
        return next;
      });
    }
  }

  if (items.length === 0) {
    return (
      <div className="notice info" style={{ marginTop: 12 }}>
        Нет комментариев, требующих ручной проверки.
      </div>
    );
  }

  return (
    <div>
      {error && (
        <div className="notice error" style={{ marginBottom: 12 }}>
          {error}
        </div>
      )}

      <ul className="quote-list" style={{ marginTop: 12 }}>
        {items.map((item) => {
          const isBanning = banning.has(item.comment_id);
          const isBanned = banned.has(item.comment_id);

          return (
            <li key={item.comment_id} style={{ opacity: isBanned ? 0.5 : 1 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ marginBottom: 4 }}>
                    <span className="muted" style={{ fontSize: "0.85rem" }}>
                      {item.author_name || "—"}
                    </span>
                    <ConfidenceBadge score={item.confidence_score} />
                    <TargetBadge target={item.insult_target} />
                  </div>
                  <ExpandableText text={item.text} />
                  {isBanned && (
                    <div style={{ marginTop: 6, fontSize: "0.85rem", color: "var(--success, #22c55e)" }}>
                      ✓ Забанен
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  className="btn danger"
                  style={{ flexShrink: 0, padding: "6px 12px", fontSize: "0.85rem" }}
                  onClick={() => handleBan(item)}
                  disabled={isBanning || isBanned}
                >
                  {isBanning ? "Баню..." : isBanned ? "Забанен" : "Забанить"}
                </button>
              </div>
            </li>
          );
        })}
      </ul>

      <div className="notice neutral" style={{ marginTop: 16, fontSize: "0.9rem" }}>
        <strong>Совет:</strong> В manual review попадают и низкоуверенные случаи. Проверяйте контекст: часть комментариев может оказаться резкой критикой или спорным сарказмом, а не прямым оскорблением.
      </div>
    </div>
  );
}
