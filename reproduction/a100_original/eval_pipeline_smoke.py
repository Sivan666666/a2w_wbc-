"""Short inference check only, never a benchmark or final-checkpoint result."""
import argparse
import json
import os
from pathlib import Path
from eval_after_training import LOGS, run_case, dump

p = argparse.ArgumentParser()
p.add_argument('variant', choices=['baseline', 'precision'])
p.add_argument('gpu', type=int)
p.add_argument('--attempt', default='trial')
a = p.parse_args()
exp = 'a2w_z1_wbc_wheel_v2' + ('_precision' if a.variant == 'precision' else '')
runs = list((LOGS / 'rsl_rl' / exp).glob(f'*_{a.variant}_train_seed42_20260926'))
assert len(runs) == 1
ckpt = runs[0] / 'model_29000.pt'
assert ckpt.is_file()
output = LOGS / 'eval_pipeline_smoke_20260927' / a.variant / a.attempt
result = run_case(a.variant, ckpt, output, 20260916,
                  0.55 if a.variant == 'baseline' else 0.10, a.gpu, seconds=2, num_envs=4)
dump(output.parent / 'passed.json', {'passed': True, 'checkpoint': str(ckpt), 'output': str(output),
     'note': '2-second inference pipeline smoke only, not a benchmark', 'samples': result['samples']})
print('EVAL_PIPELINE_SMOKE_PASSED', a.variant, flush=True)
