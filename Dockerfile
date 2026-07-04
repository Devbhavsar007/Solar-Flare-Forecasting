# Multi-stage Dockerfile for Solar-Flare-Forecasting

# 1. Base stage
FROM python:3.11-slim as base
WORKDIR /app

# Create a non-root user
RUN groupadd -r solar && useradd -r -g solar solar

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories and set permissions
RUN mkdir -p /app/data /app/models /app/logs /app/configs && \
    chown -R solar:solar /app

# Switch to non-root user
USER solar

# 2. API stage
FROM base as api
COPY --chown=solar:solar src/ /app/src/
COPY --chown=solar:solar configs/ /app/configs/
COPY --chown=solar:solar scripts/ /app/scripts/
# (models directory is usually mounted as a volume in prod/staging, but we can copy the dir structure)
COPY --chown=solar:solar models/ /app/models/

EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 3. Worker stage
FROM base as worker
COPY --chown=solar:solar src/ /app/src/
COPY --chown=solar:solar configs/ /app/configs/
COPY --chown=solar:solar scripts/ /app/scripts/
COPY --chown=solar:solar models/ /app/models/

ENTRYPOINT ["python", "src/orchestration/scheduler.py"]
