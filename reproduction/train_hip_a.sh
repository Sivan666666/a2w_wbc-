#!/usr/bin/env bash
# Current selected recipe A; does not launch until explicitly invoked.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# != 1 ]]; then
  echo 'Usage: bash reproduction/train_hip_a.sh NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
output="$(realpath -m "$1")"
if [[ -e "$output" ]]; then
  echo "Output directory already exists: $output" >&2
  exit 2
fi
cd "$root"
python reproduction/relocate_assets.py --check
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root/source/LeggedManip_Lab${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
exec python reproduction/hip_a/train_hips.py \
  --scheme a --output "$output" --num_envs "${NUM_ENVS:-4096}" \
  --iterations "${ITERATIONS:-10000}" --seed 42 --headless --device cuda:0
