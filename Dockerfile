# Copyright (c) 2025 Joan Fabrégat <j@fabreg.at>
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, subject to the conditions in the full MIT License.
# The Software is provided "as is", without warranty of any kind.

# --- Builder Stage ---
FROM python:3.13-slim-bookworm AS builder

ARG DEPENDENCIES_GROUP

WORKDIR /app

# Install only necessary system dependencies and remove them afterward
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install uv and its dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN chmod +x /bin/uv /bin/uvx && \
    uv venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency specification and install production dependencies
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --group ${DEPENDENCIES_GROUP} --no-default-groups


# --- Final Image ---
FROM python:3.13-slim-bookworm
WORKDIR /app

ARG PORT=8000
ARG EMBEDDING_MODEL
ARG VERSION
ARG BUILD_ID
ARG COMMIT_SHA

# Prevent Python from writing bytecode files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure that Python outputs are sent directly to terminal without buffering
ENV PYTHONUNBUFFERED=1
ENV PORT=${PORT}
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV VERSION=${VERSION}
ENV BUILD_ID=${BUILD_ID}
ENV COMMIT_SHA=${COMMIT_SHA}

# Copy only the needed virtual environment from builder
COPY --from=builder /app/.venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy only necessary application files
COPY app/ ./app/

# Ensure a non-root user
RUN addgroup --system app && adduser --system --group --no-create-home app && \
    chown -R app:app /app
USER app

# Download the models
ENV HF_HOME=/app/cache
RUN mkdir -p /app/cache && chmod 777 /app/cache && \
    python -m app.download_model

# https://cloud.google.com/run/docs/tips/python#optimize_gunicorn
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn app.api:api --host 0.0.0.0 --port $PORT --workers 1 --log-level info --timeout-keep-alive 0"]