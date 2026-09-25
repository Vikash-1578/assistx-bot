# ============================================================
# Stage 1 — Build FreeLLMAPI from source
# ============================================================
FROM node:20-slim AS freellmapi-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 https://github.com/tashfeenahmed/freellmapi.git .
RUN npm ci && npm run build


# ============================================================
# Stage 2 — Final image (Python bot + Node + FreeLLMAPI)
# ============================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Node.js 20 + runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gcc \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy FreeLLMAPI build ----
WORKDIR /freellmapi
COPY --from=freellmapi-builder /build/dist ./dist
COPY --from=freellmapi-builder /build/node_modules ./node_modules
COPY --from=freellmapi-builder /build/package.json ./package.json

# ---- Install Python bot ----
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ ./app/
COPY start.sh .
RUN chmod +x start.sh

# Data + logs directories
RUN mkdir -p /app/data /app/logs

# Non-root user (optional — remove if permission issues)
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app /freellmapi && \
    mkdir -p /app/data/freellmapi && \
    chown -R botuser:botuser /app/data/freellmapi
USER botuser

ENV FREELLMAPI_DATA_DIR=/app/data/freellmapi

EXPOSE 8080

CMD ["./start.sh"]
