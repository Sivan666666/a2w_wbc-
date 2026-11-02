#!/usr/bin/env bash
# Portable publication wrapper; original A100 invocation is archived separately.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
variant="${1:-precision}"
case "$variant" in
  baseline) task=A2W-Z1-WBC-WheelV2;;
  precision) task=A2W-Z1-WBC-WheelV2-Precision;;
  *) echo 'Usage: bash reproduction/train.sh baseline|precision' >&2; exit 2;;
esac
cd "$root"
python reproduction/relocate_assets.py --check
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root/source/LeggedManip_Lab${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
# Use the externally selected isolated environment; never install packages here.
# Do not silently overwrite/reuse a run directory. Isaac Lab timestamps each run.
exec python scripts/rsl_rl/train.py \
  --task "$task" --num_envs "${NUM_ENVS:-4096}" --headless --device cuda:0 \
  --max_iterations "${ITERATIONS:-50000}" --seed 42 \
  --run_name "${RUN_NAME:-${variant}_fresh_seed42_50k}" --logger tensorboard \
  'agent.resume=false' 'agent.save_interval=1000' \
  'env.scene.terrain.visual_material=null' 'env.scene.sky_light.spawn.texture_file=null' \
  'env.commands.ee_pose.debug_vis=false' 'env.commands.base_velocity.debug_vis=false'
