#!/usr/bin/env bash
# One independent fresh PPO run. Does NOT resume smoke-test weights.
set -euo pipefail
variant="${1:?baseline or precision}"; gpu="${2:?physical GPU}"; mode="${3:-train}"
root=/data01/niuwa/sivan/workspace/leggedmanip_wheelv2_20260926
case "$variant" in
  baseline) task=A2W-Z1-WBC-WheelV2;;
  precision) task=A2W-Z1-WBC-WheelV2-Precision;;
  *) exit 2;;
esac
case "$mode" in
  smoke) envs=64; iterations=3;;
  train) envs=4096; iterations=50000;;
  *) exit 2;;
esac
export CUDA_VISIBLE_DEVICES="$gpu"
export WHEEL_RUN="${variant}_${mode}_seed42_20260926"
exec bash "$root/env.sh" python scripts/rsl_rl/train.py \
  --task "$task" --num_envs "$envs" --headless --device cuda:0 \
  --kit_args="--/log/file=/data01/niuwa/sivan/logs/leggedmanip_wheelv2_20260926/kit_${WHEEL_RUN}.log" \
  --max_iterations "$iterations" --seed 42 --run_name "$WHEEL_RUN" --logger tensorboard \
  'agent.resume=false' 'agent.save_interval=1000' \
  'env.scene.terrain.visual_material=null' 'env.scene.sky_light.spawn.texture_file=null' \
  'env.commands.ee_pose.debug_vis=false' 'env.commands.base_velocity.debug_vis=false'
