FROM python:3.12-slim

# uv resolves and installs from the lockfile: runtime deps plus the `web` extra, no dev group.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" PORT=8080

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY deicm ./deicm
RUN uv sync --frozen --no-dev --extra web

# Litestream streams the SQLite database to the backup bucket (deploy/litestream.yml).
COPY --from=litestream/litestream:0.3.13 /usr/local/bin/litestream /usr/local/bin/litestream
COPY deploy/litestream.yml /etc/litestream.yml
COPY deploy/start.sh /usr/local/bin/start.sh

# start.sh drops from root to this user once the data volume is owned by it.
RUN useradd --system --create-home --uid 10001 app

EXPOSE 8080
CMD ["start.sh"]
