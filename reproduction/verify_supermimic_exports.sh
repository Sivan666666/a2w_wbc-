#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
run="$root/reproduction/supermimic_runtime.sh"
bash "$run" scripts/eval_supermimic.py --actor "$root/runs/smoke_full/actor_0000256.jit" \
  --output "$root/runs/eval_bptt_smoke" --num_envs 64 --seconds 4 --headless --device cuda:0
bash "$run" scripts/eval_supermimic.py --actor "$root/runs/ppo_smoke02/actor_final.jit" \
  --output "$root/runs/eval_ppo_smoke" --num_envs 64 --seconds 4 --headless --device cuda:0
bash "$run" scripts/train_supermimic.py --num_envs 4096 --steps 320 --warmup 16 \
  --validate_every 32 --horizon_ramp 32 --resume "$root/runs/smoke_full/latest.pt" \
  --output "$root/runs/resume_smoke" --headless --device cuda:0
