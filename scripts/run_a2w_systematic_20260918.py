"""Sequential three-seed evaluation; never restarts/changes training."""
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
checkpoint = Path('logs/rsl_rl/a2w_z1_wbc/2026-09-14_14-25-49_repro_seed42_20260914_142542/model_4999.pt').resolve()
seeds = [20260916, 20260917, 20260918]

def status(**data):
    (root / 'status.json').write_text(json.dumps(data, indent=2) + '\n')

try:
    summaries = []
    for seed in seeds:
        output = root / f'eval_seed{seed}'
        command = [sys.executable, 'scripts/eval_a2w_systematic_20260918.py',
                   '--checkpoint', str(checkpoint), '--output', str(output),
                   '--seed', str(seed), '--num_envs', '64', '--seconds', '200',
                   '--headless', '--device', 'cuda:0']
        status(stage='EVALUATING', seed=seed, completed_seeds=len(summaries), command=command)
        with (root / f'eval_seed{seed}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert json.loads((output/'status.json').read_text())['stage'] == 'COMPLETE'
        summaries.append(json.loads((output/'summary.json').read_text()))
    import numpy as np
    names = ['ee_native_m', 'ee_visual_world_m', 'orientation_rad',
             'velocity_xy_m_s', 'yaw_rate_rad_s', 'tilt_deg', 'height_m']
    pooled = {}
    for name in names:
        arrays = []
        for seed in seeds:
            with np.load(root / f'eval_seed{seed}' / 'trace.npz') as data:
                arrays.append(data[name].ravel().astype(np.float64))
        x = np.concatenate(arrays)
        pooled[name] = dict(mean=float(x.mean()), rmse=float(np.sqrt(np.mean(x*x))),
                            p95=float(np.quantile(x, .95)), max=float(x.max()))
    results = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   protocol='3 seeds x 64 environments x 200 seconds, native PLAY, pre-reset samples, all frames.',
                   samples=sum(s['samples'] for s in summaries), seeds=summaries, pooled=pooled,
                   non_timeout_resets=sum(s['non_timeout_resets'] for s in summaries),
                   total_resets=sum(s['total_resets'] for s in summaries),
                   note='Primary EE metric uses native mixed coordinates (link0 XY + world Z), not world Euclidean distance. Visual-world target metric reported separately. Not a success-rate benchmark or matched-target cross-framework ranking.')
    (root/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    status(stage='COMPLETE', completed_seeds=3, results=str(root/'results.json'))
    print(json.dumps(results, indent=2), flush=True)
except BaseException:
    status(stage='FAILED', traceback=traceback.format_exc())
    raise
