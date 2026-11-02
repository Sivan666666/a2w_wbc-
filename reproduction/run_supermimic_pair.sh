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
# User superseded the paired campaign: only train BPTT. Keep the prefix above
# unchanged so the already-running shell can read this tail after its child exits.
# Do not launch PPO, paired evaluation, or the paired aggregate automatically.
printf 'BPTT_ONLY_COMPLETE %s\n' "$(date -Is)" > "$out/COMPLETE.txt"
