#!/usr/bin/env bash
set -euo pipefail
export WHEEL_ROOT=/data01/niuwa/sivan/workspace/leggedmanip_wheelv2_20260926
export HOME=/data01/niuwa/sivan
export VIRTUAL_ENV="$WHEEL_ROOT/runtime/env"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export ISAACLAB_PATH="$WHEEL_ROOT/IsaacLab"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONHOME PYTHONPATH
export XDG_CACHE_HOME="$HOME/.cache/leggedmanip_wheelv2_20260926/${WHEEL_RUN:-setup}"
export XDG_CONFIG_HOME="$WHEEL_ROOT/runtime/config/${WHEEL_RUN:-setup}"
export XDG_DATA_HOME="$WHEEL_ROOT/runtime/data/${WHEEL_RUN:-setup}"
export TMPDIR="$HOME/tmp/leggedmanip_wheelv2_20260926/${WHEEL_RUN:-setup}"
export UV_CACHE_DIR="$HOME/.cache/leggedmanip_wheelv2_20260926/uv"
export PIP_CACHE_DIR="$HOME/.cache/leggedmanip_wheelv2_20260926/pip"
export TORCH_HOME="$HOME/.cache/leggedmanip_wheelv2_20260926/torch"
export TORCH_EXTENSIONS_DIR="$HOME/.cache/leggedmanip_wheelv2_20260926/torch_extensions"
export OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$WHEEL_ROOT/runtime/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR"
cd "$WHEEL_ROOT/LeggedManip_Lab"
exec "$@"
