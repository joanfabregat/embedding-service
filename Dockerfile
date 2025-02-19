# --- Builder Stage ---
FROM python:3.12-slim-bookworm AS builder
WORKDIR /app

# Install only necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv and its dependencies
COPY --from=ghcr.io/astral-sh/uv:0.5.31 /uv /uvx /bin/
RUN uv venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy dependency specification and install production dependencies
# Install minimal transformers without models
COPY uv.lock pyproject.toml ./

# Predownload models to a specific directory
# This prevents downloading the entire model zoo
RUN mkdir -p /app/models

# Install dependencies
RUN uv sync --frozen --group prod && \
    # Clean cache files from pip
    find /app/.venv -name '*.pyc' -delete && \
    find /app/.venv -name '__pycache__' -delete && \
    # Remove unnecessary torch components
    rm -rf /app/.venv/lib/python3.12/site-packages/torch/test/ && \
    rm -rf /app/.venv/lib/python3.12/site-packages/torch/cuda/ && \
    # Remove transformers cache and unneeded model files
    rm -rf /app/.venv/lib/python3.12/site-packages/transformers/models/* && \
    rm -rf /root/.cache/huggingface

# Copy and run model download script
COPY download_models.py ./
ENV TRANSFORMERS_CACHE=/app/models
RUN python download_models.py


# --- Final Image ---
FROM python:3.12-slim-bookworm
WORKDIR /app

ARG PORT=80
ARG APP_ENV=production
ARG APP_VERSION
ARG APP_BUILD_ID
ARG APP_COMMIT_SHA

ENV PORT=${PORT}
ENV APP_ENV=${APP_ENV}
ENV APP_VERSION=${APP_VERSION}
ENV APP_BUILD_ID=${APP_BUILD_ID}
ENV APP_COMMIT_SHA=${APP_COMMIT_SHA}

# Copy only the needed virtual environment from builder
COPY --from=builder /app/.venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy pre-downloaded models
COPY --from=builder /app/models /app/models
ENV TRANSFORMERS_CACHE=/app/models

# Copy only necessary application files
COPY app/ ./app/
ENV PYTHONPATH=/app

# Set torch env vars to reduce memory usage
ENV PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

EXPOSE ${PORT}
USER nobody
CMD ["sh", "-c", "gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app.frontend.serve:app"]