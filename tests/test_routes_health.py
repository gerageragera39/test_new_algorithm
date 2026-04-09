from __future__ import annotations

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
