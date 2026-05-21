#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:-}"

if [[ -z "${PROMPT_FILE}" ]]; then
  echo "Usage: $0 /path/to/prompt.txt" >&2
  exit 1
fi

python examples/online_serving/text_to_video_web/app.py \
  --host 0.0.0.0 \
  --port 7862 \
  --omni-server http://127.0.0.1:8099 \
  --compare-omni-server http://179.45.3.3:9099 \
  --prompt-file "${PROMPT_FILE}"
