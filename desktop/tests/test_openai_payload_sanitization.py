from __future__ import annotations


def test_sanitize_openai_text_removes_control_chars_and_surrogates() -> None:
    from app.services.labeling import _sanitize_openai_text

    raw = "hello\u0000world\ud83dline"
    sanitized = _sanitize_openai_text(raw)

    assert "\u0000" not in sanitized
    assert "\ud83d" not in sanitized
    assert "hello" in sanitized
    assert "world" in sanitized
