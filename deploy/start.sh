#!/bin/sh
# Container entrypoint. Runs as root only long enough to give the app user the volume,
# then drops to it. With a backup bucket configured: restore the database if the volume
# is empty, then run the web server under Litestream so every write is replicated.
set -eu

run_as_app() { exec setpriv --reuid=app --regid=app --init-groups "$@"; }

if [ -n "${DEICM_DB:-}" ]; then
  dir=$(dirname "$DEICM_DB")
  mkdir -p "$dir"
  chown app:app "$dir"
fi

if [ -n "${DEICM_DB:-}" ] && [ -n "${BUCKET_NAME:-}" ]; then
  setpriv --reuid=app --regid=app --init-groups \
    litestream restore -config /etc/litestream.yml -if-db-not-exists -if-replica-exists "$DEICM_DB"
  run_as_app litestream replicate -config /etc/litestream.yml -exec deicm-web
fi

run_as_app deicm-web
