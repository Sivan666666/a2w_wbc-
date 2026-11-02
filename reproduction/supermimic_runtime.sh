#!/usr/bin/env bash
# Run INSIDE the existing isolated PS1 image, using dependencies read-only.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
source /mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab/scripts/repro_env.sh
export HOME="$root/.runtime/home"
export XDG_CACHE_HOME="$root/.runtime/cache/xdg"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_CONFIG_HOME="$HOME/.config"
export TMPDIR="$root/.runtime/tmp"
export CUDA_CACHE_PATH="$root/.runtime/cache/cuda"
export TORCH_HOME="$root/.runtime/cache/torch"
export HF_HOME="$root/.runtime/cache/huggingface"
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_CONFIG_HOME"
export PYTHONPATH="$root/source/LeggedManip_Lab:$root/scripts"
export WANDB_MODE=disabled
cd "$root"
exec python "$@"
