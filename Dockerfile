# Convenience image for colleagues who want to run aibom-security without
# installing Python/uv locally. Day-to-day development happens via `uv`
# directly (see README) — this is a packaging wrapper around that, not a
# separate dev environment.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /app
COPY . .
RUN uv sync --locked --no-dev

# Same base distro + Python version as the builder — required so the venv's
# interpreter path matches (see https://docs.astral.sh/uv/guides/integration/docker/).
FROM python:3.12-slim-bookworm

RUN groupadd --system --gid 999 nonroot \
    && useradd --system --gid 999 --uid 999 --create-home nonroot

COPY --from=builder --chown=nonroot:nonroot /app /app
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app
USER nonroot

ENTRYPOINT ["aibom"]
