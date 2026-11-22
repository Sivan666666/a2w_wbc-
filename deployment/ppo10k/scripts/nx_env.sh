#!/usr/bin/env bash
set -euo pipefail
PPO10K_PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PPO10K_INSTALL=$(dirname "$PPO10K_PROJECT")
export PATH="$PPO10K_INSTALL/.venv/bin:$PATH"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$PPO10K_INSTALL/cache/xdg" TORCH_HOME="$PPO10K_INSTALL/cache/torch"
export LD_LIBRARY_PATH="$PPO10K_INSTALL/vendor/z1_sdk/lib:$PPO10K_INSTALL/vendor/unitree_sdk2/thirdparty/lib/aarch64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset PYTHONPATH
cd "$PPO10K_PROJECT"
exec "$@"
