#!/bin/bash
set -e

# ---- FreeLLMAPI environment ----
export PORT=3001
export HOST_BIND=127.0.0.1
export NODE_ENV=production

# Generate encryption key if not set
if [ -z "$ENCRYPTION_KEY" ]; then
    export ENCRYPTION_KEY=$(node -e "console.log(require('crypto').randomBytes(32).toString('hex'))")
fi

# ---- Build FreeLLMAPI config from env vars ----
export FREEAPI_CONFIG_JSON=$(node -e "
const keys = [];
if (process.env.GROQ_API_KEY) keys.push({platform: 'groq', key: process.env.GROQ_API_KEY, label: 'groq', enabled: true});
if (process.env.GEMINI_API_KEY) keys.push({platform: 'google', key: process.env.GEMINI_API_KEY, label: 'gemini', enabled: true});
if (process.env.CEREBRAS_API_KEY) keys.push({platform: 'cerebras', key: process.env.CEREBRAS_API_KEY, label: 'cerebras', enabled: true});
if (process.env.OPENROUTER_API_KEY) keys.push({platform: 'openrouter', key: process.env.OPENROUTER_API_KEY, label: 'openrouter', enabled: true});
if (process.env.SAMBANOVA_API_KEY) keys.push({platform: 'sambanova', key: process.env.SAMBANOVA_API_KEY, label: 'sambanova', enabled: true});
if (process.env.NVIDIA_API_KEY) keys.push({platform: 'nvidia', key: process.env.NVIDIA_API_KEY, label: 'nvidia', enabled: true});
console.log(JSON.stringify({
  admin: {email: 'admin@local', password: 'changeme12345'},
  keys: keys,
  routing: {strategy: 'balanced'}
}));
")

echo "[start.sh] FreeLLMAPI configured with $(echo $FREEAPI_CONFIG_JSON | node -e 'let d=\"\"; process.stdin.on(\"data\",c=>d+=c); process.stdin.on(\"end\",()=>console.log(JSON.parse(d).keys.length))') provider keys"

# ---- Start FreeLLMAPI ----
echo "[start.sh] Starting FreeLLMAPI on port 3001..."
cd /freellmapi
node dist/server.js > /tmp/freellmapi.log 2>&1 &
FREELMAPI_PID=$!
echo "[start.sh] FreeLLMAPI started with PID $FREELMAPI_PID"

# Wait for readiness
echo "[start.sh] Waiting for FreeLLMAPI..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:3001/ > /dev/null 2>&1; then
        echo "[start.sh] FreeLLMAPI is ready!"
        break
    fi
    sleep 1
done

# ---- Start Telegram bot ----
echo "[start.sh] Starting Telegram bot..."
cd /app
exec python -m app.main
