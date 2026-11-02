#!/usr/bin/env bash
# One GPU, sequential jobs; never stop or reuse another user's process.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
seed="${SEED:-42}"
envs="${NUM_ENVS:-4096}"
iterations="${PPO_ITERATIONS:-5000}"
steps="$((iterations * 24))"
out="$root/runs/pair_seed${seed}"
mkdir "$out"
trap 'printf "FAILED at %s\n" "$(date -Is)" > "$out/FAILED.txt"' ERR
runtime="$root/reproduction/supermimic_runtime.sh"
common=(--num_envs "$envs" --headless --device cuda:0 --seed "$seed")
bash "$runtime" scripts/train_supermimic.py "${common[@]}" --steps "$steps" \
  --output "$out/bptt" > "$out/bptt.log" 2>&1
test -f "$out/bptt/complete.json"
bash "$runtime" scripts/train_supermimic_ppo_control.py "${common[@]}" --iterations "$iterations" \
  --initial "$out/bptt/checkpoint_0000000.pt" --output "$out/ppo" > "$out/ppo.log" 2>&1
test -f "$out/ppo/complete.json"
for method in bptt ppo; do
  if [[ "$method" == bptt ]]; then
    printf -v actor '%s/bptt/actor_%07d.jit' "$out" "$steps"
  else
    actor="$out/ppo/actor_final.jit"
  fi
  for eval_seed in 20261111 20261112 20261113; do
    bash "$runtime" scripts/eval_supermimic.py --actor "$actor" \
      --output "$out/eval_${method}_${eval_seed}" --seed "$eval_seed" \
      --num_envs 64 --seconds 200 --headless --device cuda:0 \
      > "$out/eval_${method}_${eval_seed}.log" 2>&1
  done
done
bash "$runtime" scripts/eval_supermimic_curve.py "$out" > "$out/learning_curves.log" 2>&1
bash "$runtime" scripts/summarize_supermimic.py "$out"
printf 'COMPLETE %s\n' "$(date -Is)" > "$out/COMPLETE.txt"
