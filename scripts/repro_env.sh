#!/usr/bin/env bash
set -euo pipefail
export LM_ROOT=/mnt/sda/ps/leggedmanip-lab
export LM_RUNTIME="$LM_ROOT/runtime"
export HOME="$LM_RUNTIME/home"
export XDG_CACHE_HOME="$LM_RUNTIME/cache/xdg"
export XDG_DATA_HOME="$LM_RUNTIME/home/.local/share"
export XDG_CONFIG_HOME="$LM_RUNTIME/home/.config"
export PIP_CACHE_DIR="$LM_RUNTIME/cache/pip"
export TMPDIR="$LM_RUNTIME/tmp"
export CUDA_CACHE_PATH="$LM_RUNTIME/cache/cuda"
export TORCH_HOME="$LM_RUNTIME/cache/torch"
export HF_HOME="$LM_RUNTIME/cache/huggingface"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
export PATH="$LM_RUNTIME/envs/train/bin:$PATH"
export ISAACLAB_PATH="$LM_ROOT/IsaacLab_5_1"
cd "$LM_ROOT/LeggedManip_Lab"
