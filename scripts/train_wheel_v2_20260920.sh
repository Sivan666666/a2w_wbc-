#!/usr/bin/env bash
# Run inside this project's GPU-isolated container. Does not resume old weights.
set -euo pipefail
source /mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab/scripts/repro_env.sh
run=wheel_v2_seed42_20260920
audit="logs/launch_${run}"
mkdir -p "$audit/source"
cp source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/a2w_z1_wheel_v2/*.py "$audit/source/"
cp scripts/train_wheel_v2_20260920.sh "$audit/source/"
python scripts/rsl_rl/train.py \
  --task A2W-Z1-WBC-WheelV2 --num_envs 4096 --headless \
  --device cuda:0 --max_iterations 5000 --seed 42 \
  --run_name "$run" --logger tensorboard \
  'agent.resume=false' \
  'env.scene.terrain.visual_material=null' \
  'env.scene.sky_light.spawn.texture_file=null' \
  'env.commands.ee_pose.debug_vis=false' \
  'env.commands.base_velocity.debug_vis=false' \
  2>&1 | tee "$audit/train.log"
