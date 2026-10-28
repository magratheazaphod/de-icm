#!/usr/bin/env bash
# Test, deploy to Fly, then smoke-test the live app.
# Needs DEICM_AUTH=user:pass (one of the AUTH_USERS on the server) for the authenticated checks.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --extra web pytest -q
fly deploy
DEICM_URL="${DEICM_URL:-https://de-icm.fly.dev}" uv run --extra web pytest -q -m deploy -rs tests/test_deploy.py
