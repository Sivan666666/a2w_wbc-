"""Run three native seeds for a single task/checkpoint; fail on any incomplete run."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--task', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)

def status(**value):
    (a.output / 'status.json').write_text(json.dumps(value, indent=2)+'\n')

try:
    summaries = []
    for seed in [20260916, 20260917, 20260918]:
        out = a.output / f'seed{seed}'
        cmd = [sys.executable, 'scripts/eval_wheel_compare_20260921.py', '--task', a.task,
               '--checkpoint', a.checkpoint, '--output', str(out), '--seed', str(seed),
               '--num_envs', '64', '--seconds', '200', '--headless', '--device', 'cuda:0']
        status(stage='EVALUATING', seed=seed, completed=len(summaries))
        with (a.output/f'seed{seed}.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert json.loads((out/'status.json').read_text())['stage'] == 'COMPLETE'
        summaries.append(json.loads((out/'summary.json').read_text()))
    (a.output/'summaries.json').write_text(json.dumps(summaries, indent=2)+'\n')
    status(stage='COMPLETE', completed=3)
except BaseException:
    status(stage='FAILED', traceback=traceback.format_exc())
    raise
