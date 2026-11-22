#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec bash scripts/nx_env.sh python -m a2w_deploy.sim --headless --demo \
  --seconds 60 --output ../logs/offline_manual "$@"
