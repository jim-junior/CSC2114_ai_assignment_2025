#!/usr/bin/env bash
set -euo pipefail

if [ -n "${NGROK_AUTHTOKEN:-}" ]; then
  ngrok config add-authtoken "$NGROK_AUTHTOKEN" || true
fi

# start app in background
/opt/venv/bin/uvicorn app:app --host 0.0.0.0 --port "${NGROK_PORT:-80}" --proxy-headers &

# foreground ngrok so container stays alive
exec ngrok http --region="${NGROK_REGION:-us}" "${NGROK_PORT:-80}" --log=stdout
