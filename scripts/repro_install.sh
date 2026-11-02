#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/repro_env.sh"
# Run after preparing the private venv and copying the Isaac Sim 5.1 binary packages.
# The upstream Sim/Lab distributions have conflicting optional web dependency pins;
# install the source packages without letting pip replace the validated runtime.
python -m pip install --no-deps --no-build-isolation \
  -e "$ISAACLAB_PATH/source/isaaclab" -e "$ISAACLAB_PATH/source/isaaclab_assets" \
  -e "$ISAACLAB_PATH/source/isaaclab_tasks" -e "$ISAACLAB_PATH/source/isaaclab_rl[rsl-rl]" \
  -e source/LeggedManip_Lab
python -m pip install -c scripts/repro_constraints.txt \
  'rsl-rl-lib==5.0.1' 'opencv-python==4.11.0.86' 'pin-pink==3.1.0' \
  'daqp==0.7.2' 'dex-retargeting==0.4.6' pycollada -r requirements.txt
# Keep the full audit, including unused Isaac Sim GUI/cloud extras and pin conflicts.
python -m pip check > "$LM_RUNTIME/logs/pip_check.txt" 2>&1 || true
python -m pip freeze > "$LM_RUNTIME/logs/pip_freeze.txt"
python scripts/prepare_a2w_asset.py --source "$LM_RUNTIME/assets/a2wz1_source" \
  --destination source/LeggedManip_Lab/LeggedManip_Lab/assets/a2w_z1
