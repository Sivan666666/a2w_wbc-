#!/usr/bin/env bash
set -euo pipefail
PPO10K_PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PPO10K_Z1_CONTROLLER="$(dirname "$PPO10K_PROJECT")/vendor/z1_controller"
export LD_LIBRARY_PATH="$PPO10K_Z1_CONTROLLER/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PPO10K_Z1_CONTROLLER/build"
exec ./z1_ctrl
