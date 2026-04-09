#!/bin/sh
set -e

echo "Waiting for database migrations..."
ATTEMPTS=0
until alembic upgrade head; do
  ATTEMPTS=$((ATTEMPTS+1))
  if [ "$ATTEMPTS" -ge 12 ]; then
    echo "Failed to run migrations after multiple attempts."
    exit 1
  fi
  sleep 3
done

echo "Starting FastAPI..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

