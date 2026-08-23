#!/bin/sh
set -eu

# Platforms like Cloud Run only publish one port from one container and
# inject which port to listen on via $PORT (defaulting here to 8080, Cloud
# Run's own default, so it works unset too). Both processes run inside this
# single container: the FastAPI backend on an internal-only port, and
# Streamlit on $PORT, the one actually published.
PORT="${PORT:-8080}"

python -m src.startup_checks

uvicorn src.api:app --host 127.0.0.1 --port 8000 &

# Bounded wait for the API (including model pre-warming in its startup
# lifespan, see src/api.py) to come up before Streamlit starts serving
# traffic -- on a scale-to-zero platform this is what makes a cold start
# resolve during container startup instead of mid-conversation.
for i in $(seq 1 60); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

exec streamlit run medibot.py \
    --server.address=0.0.0.0 \
    --server.port="$PORT" \
    --server.headless=true \
    --browser.gatherUsageStats=false
