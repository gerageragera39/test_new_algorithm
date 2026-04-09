from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.base import Base


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = testing_session_local()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture()
def test_settings() -> Settings:
    cache_dir = Path("data/test_cache")
    data_dir = Path("data/test_data")
    raw_dir = Path("data/test_raw")
    reports_dir = Path("data/test_reports")
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        dry_run=True,
        dry_run_fixture_path=Path("tests/fixtures/mock_youtube.json"),
        author_name="Сергей",
        youtube_playlist_id="PL_mock_playlist",
        youtube_api_key="mock-key",
        database_url="sqlite+pysqlite:///:memory:",
        cache_dir=cache_dir,
        data_dir=data_dir,
        raw_dir=raw_dir,
        reports_dir=reports_dir,
        cluster_min_size=2,
        cluster_min_samples=1,
        cluster_kmeans_fallback_enabled=True,
        max_representatives_per_cluster=3,
    )
