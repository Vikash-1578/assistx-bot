FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY app/ ./app/
COPY .env.example ./.env.example

# Non-root user
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Data + logs directories
RUN mkdir -p /app/data /app/logs

EXPOSE 8080

CMD ["python", "-m", "app.main"]
