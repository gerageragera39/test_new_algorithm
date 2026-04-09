from __future__ import annotations

from app.api.routes import _prioritize_question_comments


def test_prioritize_question_comments_filters_offensive_items() -> None:
    questions = [
        "Ты дебил или что?",
        "Почему в отчете нет ссылок на источник?",
        "Как вы выбрали этот кластер?",
    ]
    ranked = _prioritize_question_comments(questions)
    assert ranked
    assert all("дебил" not in item.lower() for item in ranked)


def test_prioritize_question_comments_keeps_best_question_first() -> None:
    questions = [
        "Что?",
        "Почему эта категория получила высокий вес и какие факторы учитывались?",
        "Когда будет следующий разбор этой темы?",
    ]
    ranked = _prioritize_question_comments(questions)
    assert ranked[0].startswith("Почему")


def test_prioritize_question_comments_deduplicates() -> None:
    questions = [
        "Почему нет ссылки на источник?",
        "Почему нет ссылки на источник?",
        "Почему нет ссылки на источник?",
    ]
    ranked = _prioritize_question_comments(questions)
    assert len(ranked) == 1
