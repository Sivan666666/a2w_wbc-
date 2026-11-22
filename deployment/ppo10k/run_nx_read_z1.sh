#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Receive SDK feedback only; does not start a controller or send commands.
exec bash scripts/nx_env.sh python -m a2w_deploy.read_z1 \
  --seconds 3 --output ../logs/z1_readonly_latest.json "$@"
