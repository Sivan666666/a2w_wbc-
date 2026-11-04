"""Read-only evaluation of an archived PPO checkpoint, after the active stage.

No PPO training; all six runs use the archived physical evaluation protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--training_source_manifest', type=Path, required=True)
p.add_argument('--wait_for', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
expected = '15f1d7c295d42ee12d75e1a7c241612c2a73471000da1beb3bc4a22a083fe87c'
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == expected
a.output.mkdir(parents=True, exist_ok=False)
status = dict(stage='WAITING', completed=[], checkpoint=str(a.checkpoint),
              checkpoint_sha256=expected, training_hardware='A100',
              training_log_iteration=10000, physical_transitions=983138304,
              training_log_elapsed_seconds=26675,
              evaluation_hardware='PS1 GPU3 shared with live BPTT training')

def save():
    temp = a.output / 'status.tmp'
    temp.write_text(json.dumps(status, indent=2))
    os.replace(temp, a.output / 'status.json')

save()
try:
    deadline = time.monotonic() + 7200
    while True:
        previous = json.loads(a.wait_for.read_text())
        if previous['stage'] == 'COMPLETE':
            break
        if previous['stage'] == 'FAILED':
            raise RuntimeError('Previous evaluation failed; inspect before continuing')
        if time.monotonic() > deadline:
            raise RuntimeError('Previous evaluation did not complete within two hours')
        time.sleep(30)
    status['overlap_started_unix'] = time.time()
    reports = {}
    mapping = {'ee_native_m': 'ee_native_m', 'ee_world_m': 'ee_visual_world_m',
               'orientation_rad': 'orientation_rad', 'vel_xy_mps': 'velocity_xy_m_s',
               'vel_yaw_rps': 'yaw_rate_rad_s'}
    for domain, x_min in [('expanded', .1), ('original', .55)]:
        results = []
        for seed in (20260916, 20260917, 20260918):
            name = f'{domain}_{seed}'
            out = a.output / name
            status.update(stage='EVALUATING', active=name)
            save()
            command = [sys.executable, str(Path(__file__).with_name('eval_wheelv2_a100_domains.py')),
                       '--checkpoint', str(a.checkpoint), '--output', str(out),
                       '--training_source_manifest', str(a.training_source_manifest),
                       '--seed', str(seed), '--x_min', str(x_min), '--num_envs', '64',
                       '--seconds', '200', '--headless', '--device', 'cuda:0']
            with (a.output / f'{name}.log').open('w') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            results.append(json.loads((out / 'summary.json').read_text()))
            status['completed'].append(name)
            save()
        reports[domain] = {
            k: dict(mean=statistics.mean(r[v]['mean'] for r in results),
                    seed_std=statistics.stdev(r[v]['mean'] for r in results))
            for k, v in mapping.items()}
        reports[domain]['failure_resets'] = sum(r['non_timeout_resets'] for r in results)
    status.update(stage='COMPLETE', overlap_finished_unix=time.time())
    (a.output / 'summary.json').write_text(json.dumps(dict(domains=reports, metadata=status), indent=2))
    save()
except Exception as exc:
    status.update(stage='FAILED', error=repr(exc), finished_unix=time.time())
    save()
    raise
