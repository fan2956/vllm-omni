#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="${1:-${SCRIPT_DIR}/prompts.txt}"

python examples/online_serving/text_to_video_web/app.py \
  --host 0.0.0.0 \
  --port 7862 \
  --omni-server http://127.0.0.1:8099 \
  --compare-omni-server http://179.45.3.3:9099 \
  --prompt-file "${PROMPT_FILE}"
