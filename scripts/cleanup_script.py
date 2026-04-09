#!/usr/bin/env python3
"""Fully reset local state: drop all DB tables (including Alembic state) and delete data dir."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine

from app.core.config import Settings
from app.db import models  # noqa: F401  # keep model imports for URL/config parity


def _clear_database(engine: Engine) -> None:
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if not table_names:
        print("Database is already empty (no tables found).")
        return

    dialect = engine.dialect.name.lower()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    with engine.begin() as conn:
        if dialect == "sqlite":
            conn.execute(text("PRAGMA foreign_keys=OFF"))
        metadata.drop_all(bind=conn)
        if dialect == "sqlite":
            conn.execute(text("PRAGMA foreign_keys=ON"))
    print(f"Dropped {len(table_names)} table(s), including alembic migration state.")


def _delete_data_dir(data_dir: Path) -> None:
    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        return
    shutil.rmtree(data_dir)
    print(f"Deleted data directory: {data_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Completely clear the configured database and remove the data directory."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute destructive cleanup without interactive confirmation.",
    )
    args = parser.parse_args()

    settings = Settings()
    db_url = settings.database_url
    data_dir = Path(settings.data_dir).resolve()

    print("This operation is destructive.")
    print(f"Database URL: {db_url}")
    print(f"Data directory: {data_dir}")

    if not args.force:
        answer = input("Type 'YES' to continue: ").strip()
        if answer != "YES":
            print("Aborted.")
            return

    engine = create_engine(db_url)
    try:
        _clear_database(engine)
    finally:
        engine.dispose()

    _delete_data_dir(data_dir)
    print("Cleanup completed.")


if __name__ == "__main__":
    main()
