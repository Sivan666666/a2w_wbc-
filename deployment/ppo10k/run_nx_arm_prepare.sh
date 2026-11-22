#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Actuating arm-only cycle: official home -> preparation -> hold -> home -> PASSIVE.
# Reads the same preparation profile as keyboard/VR deployment. Never starts PPO.
ARM_LOG_DIR="${1:-../logs/arm_prepare_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ARM_LOG_DIR"
exec bash scripts/nx_env.sh ./build/z1_arm_cycle "$ARM_LOG_DIR" deploy.nx.json "${2:-10}"
