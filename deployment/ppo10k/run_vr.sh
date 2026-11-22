#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
exec "${DEPLOY_PYTHON:-python3}" -m a2w_deploy.run --mode vr "$@"
