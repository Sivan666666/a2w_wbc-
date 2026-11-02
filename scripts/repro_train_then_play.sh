#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/repro_env.sh"
task="${1:?task}"; experiment="${2:?experiment}"; gpu="${3:?gpu}"
export CUDA_VISIBLE_DEVICES="$gpu"
export MUJOCO_EGL_DEVICE_ID=0
run="repro_seed42_$(date +%Y%m%d_%H%M%S)"
python scripts/rsl_rl/train.py --task "$task" --num_envs 4096 --headless \
  --device cuda:0 --max_iterations 5000 --seed 42 --run_name "$run" --logger tensorboard \
  'env.scene.terrain.visual_material=null' 'env.scene.sky_light.spawn.texture_file=null' \
  'env.commands.ee_pose.debug_vis=false' 'env.commands.base_velocity.debug_vis=false'
python scripts/rsl_rl/play.py --task "${task}-Play" --num_envs 4 --headless --device cuda:0 \
  --load_run ".*${run}" --checkpoint 'model_.*.pt' --video --video_length 1500 \
  'env.scene.terrain.visual_material=null' 'env.scene.sky_light.spawn.texture_file=null'
run_dir=$(find "logs/rsl_rl/$experiment" -maxdepth 1 -type d -name "*${run}" | sort | tail -1)
test -n "$run_dir"
robot="${experiment%_wbc}"
python scripts/repro_mujoco.py --robot "$robot" --policy "$run_dir/exported/policy.pt" \
  --output "$run_dir/mujoco" --seconds 30
