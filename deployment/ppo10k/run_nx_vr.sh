#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec bash scripts/nx_env.sh ./run_vr.sh --config deploy.nx.json "$@"
