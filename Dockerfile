FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    OFD1_DATA_DIR=/data \
    OFD1_SEED_DIR=/app/seed_data \
    OFD1_REFRESH_SECONDS=600

WORKDIR /app

# Install dependencies into the project virtualenv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application code and seed telemetry parquets
COPY app ./app
COPY data ./seed_data

# Create mount point for Kubernetes PVC and non-root runtime user
RUN useradd --create-home --uid 10001 caver \
    && mkdir -p /data \
    && chown -R caver:caver /app /data

USER caver

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
