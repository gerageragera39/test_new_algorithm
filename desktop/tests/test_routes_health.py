from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api import routes


def test_health_includes_openai_endpoint_metadata(test_settings) -> None:
    settings = test_settings.model_copy(
        update={
            "mode": "openai",
            "openai_base_url": "https://api.openai.com/v1",
        }
    )

    response = routes.health(settings=settings)

    assert response.status == "ok"
    assert response.openai_endpoint_host == "api.openai.com"
    assert response.openai_endpoint_mode == "official"


def test_load_spa_index_html_reads_from_configured_resource_path(tmp_path: Path, monkeypatch) -> None:
    spa_index = tmp_path / "frontend" / "dist" / "index.html"
    spa_index.parent.mkdir(parents=True, exist_ok=True)
    spa_index.write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(routes, "_SPA_INDEX_FILE", spa_index)

    assert routes._load_spa_index_html() == "<html>ok</html>"


def test_load_spa_index_html_returns_503_when_frontend_missing(monkeypatch) -> None:
    monkeypatch.setattr(routes, "_SPA_INDEX_FILE", Path("missing/spa/index.html"))

    with pytest.raises(HTTPException) as exc_info:
        routes._load_spa_index_html()

    assert exc_info.value.status_code == 503
