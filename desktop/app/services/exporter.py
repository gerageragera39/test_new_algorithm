"""Report export service for Markdown, HTML, and diagnostic output.

Converts DailyBriefing objects into Markdown and HTML reports, persists
them to disk organized by date, and exports cluster diagnostic data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import markdown

from app.core.config import Settings
from app.schemas.domain import ActionItem, DailyBriefing, TopicSummary


class ReportExporter:
    """Exports analysis results as Markdown and HTML report files.

    Handles formatting of briefing data into human-readable reports and
    persists them to the configured reports directory.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def to_markdown(self, briefing: DailyBriefing) -> str:
        """Convert a DailyBriefing into a formatted Markdown string.

        Args:
            briefing: The briefing data to render.

        Returns:
            Complete Markdown document as a string.
        """
        lines: list[str] = []
        lines.append(f"# Брифинг по комментариям: {briefing.video_title}")
        lines.append("")
        lines.append(f"- Видео: https://www.youtube.com/watch?v={briefing.video_id}")
        lines.append(f"- Дата видео: {briefing.published_at.isoformat()}")
        lines.append(f"- Режим: `{briefing.mode}`")

        context_source = briefing.metadata.get("episode_context_source")
        if isinstance(context_source, str) and context_source:
            lines.append(f"- Контекст выпуска: `{context_source}`")
        episode_topics = briefing.metadata.get("episode_topics")
        if isinstance(episode_topics, list) and episode_topics:
            preview = ", ".join(str(item) for item in episode_topics[:5])
            lines.append(f"- Темы выпуска (контекст): {preview}")
        provider_used = briefing.metadata.get("llm_provider_used")
        if isinstance(provider_used, str) and provider_used:
            lines.append(f"- LLM провайдер: `{provider_used}`")

        lines.append("")
        lines.append("## 📝 Краткое резюме")
        lines.append(briefing.executive_summary)
        
        # ПРИОРИТЕТ 1: Действия для автора (самое важное)
        self._append_action_items(lines, briefing.action_items)
        
        # ПРИОРИТЕТ 2: Критичные проблемы (риски и непонимание)
        self._append_list_section(
            lines,
            "## ⚠️ Риски и токсичность",
            briefing.risks_and_toxicity,
            empty_message="Существенных рисков эскалации не найдено.",
        )
        self._append_list_section(
            lines,
            "## ❓ Где аудитория спорит или не понимает подачу",
            briefing.misunderstandings_and_controversies,
            empty_message="Крупных зон непонимания не выявлено.",
        )
        
        # ПРИОРИТЕТ 3: Критика и несогласие
        lines.append("")
        lines.append(f"## 🔍 Комментарии с несогласием с позицией {self.settings.author_name}")
        if briefing.author_disagreement_comments:
            for comment in briefing.author_disagreement_comments:
                lines.append(f"- {comment}")
        else:
            lines.append("- Явных комментариев с несогласием не найдено.")
        
        # ПРИОРИТЕТ 4: Детальная статистика по темам
        lines.append("")
        lines.append("## 📊 Топ-темы (детальная разбивка)")
        lines.append("")
        for idx, topic in enumerate(briefing.top_topics, start=1):
            lines.extend(self._topic_block(idx, topic))
        
        # ПРИОРИТЕТ 5: Контекст и тренды (второстепенное)
        self._append_list_section(
            lines,
            "## 📈 Динамика относительно прошлого выпуска",
            briefing.trend_vs_previous,
            empty_message="Сравнение с предыдущим выпуском недоступно.",
        )
        lines.append("")
        lines.append(f"_Сформировано: {datetime.now(UTC).isoformat()}_")
        return "\n".join(lines).strip()

    def to_html(self, markdown_content: str) -> str:
        """Convert a Markdown string to HTML using Python-Markdown.

        Args:
            markdown_content: Raw Markdown text to convert.

        Returns:
            Rendered HTML string.
        """
        return markdown.markdown(markdown_content, extensions=["extra", "nl2br", "tables"])

    def persist(self, video_id: str, markdown_content: str, html_content: str) -> tuple[Path, Path]:
        """Write Markdown and HTML reports to disk under the reports directory.

        Args:
            video_id: YouTube video ID used as the filename stem.
            markdown_content: Rendered Markdown content to save.
            html_content: Rendered HTML content to save.

        Returns:
            A tuple of (md_path, html_path) pointing to the written files.
        """
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        base = self.settings.reports_dir / day
        base.mkdir(parents=True, exist_ok=True)
        md_path = base / f"{video_id}.md"
        html_path = base / f"{video_id}.html"
        md_path.write_text(markdown_content, encoding="utf-8")
        html_path.write_text(html_content, encoding="utf-8")
        return md_path, html_path

    def persist_cluster_diagnostics(self, video_id: str, diagnostics: dict[str, Any]) -> Path:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        base = self.settings.reports_dir / day
        base.mkdir(parents=True, exist_ok=True)
        diagnostics_path = base / f"{video_id}.cluster_diagnostics.json"
        payload = json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n"
        diagnostics_path.write_text(payload, encoding="utf-8")
        return diagnostics_path

    def _topic_block(self, index: int, topic: TopicSummary) -> list[str]:
        lines: list[str] = []
        lines.append(f"### {index}. {topic.label}")
        lines.append(f"- Размер: {topic.size_count} комментариев ({topic.share_pct:.1f}%)")
        emotion = ", ".join(topic.emotion_tags) if topic.emotion_tags else "-"
        lines.append(f"- Настроение: {topic.sentiment} ({emotion})")
        lines.append(f"- Средняя уверенность кластера: {topic.assignment_confidence:.2f}")
        lines.append(f"- Пограничные комментарии: {topic.ambiguous_share_pct:.1f}%")
        if topic.soft_assignment_notes:
            lines.append("- Soft-assignment заметки:")
            for note in topic.soft_assignment_notes:
                lines.append(f"  - {note}")
        lines.append("- Позиции в кластере:")
        if topic.positions:
            ordered_positions = sorted(
                topic.positions, key=lambda item: (item.is_undetermined, -item.count)
            )
            for position in ordered_positions:
                lines.append(
                    f"  <details><summary><b>{position.title}</b> — {position.pct:.1f}% "
                    f"({position.count}/{topic.size_count})</summary>"
                )
                if position.summary:
                    lines.append(f"  {position.summary}")
                if position.markers:
                    lines.append("  Маркерные аргументы:")
                    for marker in position.markers:
                        lines.append(f"  - {marker}")
                for comment in position.comments:
                    lines.append(f"  - {comment}")
                lines.append("  </details>")
        else:
            fallback_comments = list(topic.representative_quotes) + list(topic.question_comments)
            lines.append(f"  <details><summary>Комментарии - {len(fallback_comments)}</summary>")
            for comment in fallback_comments:
                lines.append(f"  - {comment}")
            lines.append("  </details>")
        return lines

    @staticmethod
    def _append_list_section(
        lines: list[str],
        title: str,
        items: list[str],
        *,
        empty_message: str,
    ) -> None:
        lines.append("")
        lines.append(title)
        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append(f"- {empty_message}")

    @staticmethod
    def _append_action_items(lines: list[str], items: list[ActionItem]) -> None:
        lines.append("")
        lines.append("## 🎯 Что сделать в следующем выпуске")
        if not items:
            lines.append("- Конкретные действия не были сформированы.")
            return

        for index, item in enumerate(items, start=1):
            lines.append(f"### {index}. {item.topic_label}")
            lines.append(f"- Приоритет: {item.priority}")
            lines.append(f"- Доля темы: {item.share_pct:.1f}%")
            if item.action:
                lines.append(f"- Что сделать: {item.action}")
            if item.key_question:
                lines.append(f"- Главный вопрос аудитории: «{item.key_question}»")
            lines.append("")
