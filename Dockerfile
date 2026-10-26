FROM python:3.12-slim

# uv resolves and installs from the lockfile: runtime deps plus the `web` extra, no dev group.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" PORT=8080

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY deicm ./deicm
RUN uv sync --frozen --no-dev --extra web

RUN useradd --system --create-home --uid 10001 app
USER app

EXPOSE 8080
CMD ["deicm-web"]
