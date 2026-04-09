#!/bin/sh
set -e

echo "Starting Celery beat..."
exec celery -A app.workers.celery_app:celery_app beat --loglevel=INFO

