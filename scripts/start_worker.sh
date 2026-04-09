#!/bin/sh
set -e

echo "Starting Celery worker..."
exec celery -A app.workers.celery_app:celery_app worker --loglevel=INFO

