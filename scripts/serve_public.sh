#!/usr/bin/env bash
# serve_public.sh - Make the detector API reachable for the deployed website.
#
#   scripts/serve_public.sh <site>.vercel.app <your-domain>.ngrok-free.app
#
# Starts the API (python -m src.api) and an ngrok tunnel on your fixed ngrok
# address, and keeps the laptop awake (no sleep, closing the lid does not
# suspend) until you press Ctrl+C. Run it from anywhere; it finds the project.
set -euo pipefail

SITE="${1:?usage: scripts/serve_public.sh <site>.vercel.app <your-domain>.ngrok-free.app}"
DOMAIN="${2:?usage: scripts/serve_public.sh <site>.vercel.app <your-domain>.ngrok-free.app}"
SITE="${SITE#https://}"
DOMAIN="${DOMAIN#https://}"
cd "$(dirname "$0")/.."

command -v ngrok >/dev/null || { echo "ngrok is not installed: see 'Deploy' in README.md." >&2; exit 1; }

# Re-run this script inside systemd-inhibit, so the laptop stays awake while it runs.
if [[ -z "${DEEPFAKE_INHIBITED:-}" ]]; then
  exec env DEEPFAKE_INHIBITED=1 systemd-inhibit --what=sleep:idle:handle-lid-switch \
    --who="Deepfake Face Detector" --why="Serving the detector API" "$0" "$@"
fi

# The website (and its preview deployments) may call the API, plus local development.
export WEB_ORIGINS="https://${SITE},http://localhost:3000,http://127.0.0.1:3000"
export WEB_ORIGIN_REGEX='https://[a-z0-9-]+\.vercel\.app'

# The API runs in its own memory-capped group, so running out of RAM stops only the API.
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python -m src.api &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

echo "Starting the API ..."
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:8000/api/health >/dev/null && break
  sleep 1
done
curl -sf http://127.0.0.1:8000/api/health >/dev/null || { echo "The API did not start." >&2; exit 1; }

echo "API is up. Public address: https://${DOMAIN}  (website: https://${SITE})"
echo "Press Ctrl+C to stop."
ngrok http --url="${DOMAIN}" 8000 --log=stdout --log-level=warn
