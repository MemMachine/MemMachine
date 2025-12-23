#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=./app
uvicorn app.main:app --reload --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}
