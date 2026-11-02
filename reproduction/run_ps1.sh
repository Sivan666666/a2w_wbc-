#!/usr/bin/env bash
set -euo pipefail
source /mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab/scripts/repro_env.sh
R=/mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab/logs/wheelv2_50k_video_20261001
export CUDA_VISIBLE_DEVICES=0
python "$R/record_50k.py" --variant baseline --checkpoint "$R/checkpoints/baseline/model_49999.pt" \
  --output "$R/baseline" --seed 20260916 --num_envs 1 --seconds 48 --headless --device cuda:0 \
  --kit_args="--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0" > "$R/baseline.log" 2>&1
python -c "import json; assert json.load(open('$R/baseline/status.json'))['stage']=='COMPLETE'"
python "$R/record_50k.py" --variant precision --checkpoint "$R/checkpoints/precision/model_49999.pt" \
  --output "$R/precision" --seed 20260916 --num_envs 1 --seconds 48 --headless --device cuda:0 \
  --replay_trace "$R/baseline/trace.npz" \
  --kit_args="--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0" > "$R/precision.log" 2>&1
python - "$R" <<'PY'
import sys,json,numpy as np
from pathlib import Path
r=Path(sys.argv[1])
for v in ['baseline','precision']:
    assert json.loads((r/v/'status.json').read_text())['stage']=='COMPLETE'
a=np.load(r/'baseline/trace.npz');b=np.load(r/'precision/trace.npz')
assert np.array_equal(a['ee_command'],b['ee_command'])
assert np.array_equal(a['velocity_command'],b['velocity_command'])
print('BOTH_VIDEOS_COMPLETE; ALL_EE_AND_BASE_COMMANDS_IDENTICAL',flush=True)
PY
