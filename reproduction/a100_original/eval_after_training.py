"""Persistent, isolated 50K completion watcher + paired-domain evaluation.

Never stops/restarts training. Never substitutes an intermediate checkpoint.
One instance per variant; waits for that training process to exit AND final
checkpoint to exist, then uses its released GPU. No automatic failed-run retry.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = Path('/data01/niuwa/sivan/logs/leggedmanip_wheelv2_20260926')
OUTPUT = LOGS / 'eval_50k_20260927'
SEEDS = [20260916, 20260917, 20260918]
DOMAINS = [('original', 0.55), ('expanded', 0.10)]


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def same_process(pid, variant):
    try:
        cmd = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    except FileNotFoundError:
        return False
    return b'scripts/rsl_rl/train.py' in cmd and f'{variant}_train_seed42_20260926'.encode() in cmd


def run_case(variant, checkpoint, output, seed, x_min, gpu, seconds=200, num_envs=64):
    """Also used by short pipeline tests; output must never already exist."""
    output = Path(output)
    assert not output.exists(), f'Refusing to overwrite {output}'
    output.parent.mkdir(parents=True, exist_ok=True)
    tag = f'eval_{variant}_{output.parent.name}_{output.name}'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WHEEL_RUN=tag)
    cmd = ['bash', str(ROOT / 'env.sh'), 'python',
           'scripts/eval_wheelv2_a100_domains.py', '--checkpoint', str(checkpoint),
           '--output', str(output), '--seed', str(seed), '--x_min', str(x_min),
           '--num_envs', str(num_envs), '--seconds', str(seconds),
           '--headless', '--device', 'cuda:0',
           f'--kit_args=--/log/file={output.parent / (output.name + "_kit.log")}']
    with output.with_suffix('.log').open('x') as log:
        child = subprocess.run(cmd, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT)
    # Kit shutdown can hide Python exceptions in the process exit code.
    assert child.returncode == 0, f'Evaluation process failed: {child.returncode}'
    assert json.loads((output / 'status.json').read_text())['stage'] == 'COMPLETE'
    result = json.loads((output / 'summary.json').read_text())
    assert result['samples'] == round(seconds / 0.02) * num_envs
    assert result['ee_x_range_requested'] == [x_min, 0.85]
    return result


def aggregate(results):
    keys = ['ee_native_m', 'ee_visual_world_m', 'orientation_rad',
            'velocity_xy_m_s', 'yaw_rate_rad_s', 'tilt_deg', 'height_m']
    counts = [r['samples'] for r in results]
    total = sum(counts)
    out = {'seeds': [r['seed'] for r in results], 'samples': total,
           'env_seconds': sum(r['num_envs'] * r['seconds_per_env'] for r in results),
           'non_timeout_resets': sum(r['non_timeout_resets'] for r in results),
           'termination_counts': {k: sum(r['termination_counts'][k] for r in results)
                                  for k in results[0]['termination_counts']}}
    for key in keys:
        means = [r[key]['mean'] for r in results]
        out[key] = {
            'mean': sum(m * n for m, n in zip(means, counts)) / total,
            'seed_mean_sample_std': statistics.stdev(means) if len(means) > 1 else 0.0,
            'rmse': math.sqrt(sum(r[key]['rmse']**2 * n for r, n in zip(results, counts)) / total),
            'max': max(r[key]['max'] for r in results),
        }
    out['per_seed'] = results
    return out


def comparison_if_ready():
    # Synchronize simultaneous variant completion; don't report half a table.
    with (OUTPUT / 'comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = [OUTPUT / name / 'summary.json' for name in ('baseline', 'precision')]
        if not all(p.exists() for p in files):
            return
        result = {name: json.loads(path.read_text()) for name, path in zip(('baseline', 'precision'), files)}
        dump(OUTPUT / 'comparison.json', result)
        lines = ['# WheelV2 50K paired-domain evaluation', '',
                 '3 seeds × 64 env × 200 seconds per method/domain. All frames; no warmup removal.',
                 'PLAY observation noise off; reset randomization, actuator delays and native terminations retained.',
                 'Position columns show mean ± sample SD of the three seed means (cm).', '',
                 '| Domain | Method | Native EE (cm) | Visual target world EE (cm) | Orientation (deg) | XY velocity error (m/s) | Yaw-rate error (rad/s) | Non-timeout resets |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for domain, _ in DOMAINS:
            for variant in ('baseline', 'precision'):
                r = result[variant]['domains'][domain]
                pos = [f"{r[k]['mean']*100:.2f} ± {r[k]['seed_mean_sample_std']*100:.2f}"
                       for k in ('ee_native_m', 'ee_visual_world_m')]
                lines.append(f"| {domain} | {variant} | {pos[0]} | {pos[1]} | "
                             f"{math.degrees(r['orientation_rad']['mean']):.2f} | "
                             f"{r['velocity_xy_m_s']['mean']:.4f} | {r['yaw_rate_rad_s']['mean']:.4f} | "
                             f"{r['non_timeout_resets']} |")
        lines += ['', 'Native EE = norm(link0-local x/y errors, world z error).',
                  'Visual world EE = distance to upstream visualization target; these are different metrics.',
                  'The expanded range is out-of-training-distribution for baseline. Range and reward both changed in precision.',
                  'This is random-target tracking, not a settled-point reach test or continuous handwriting benchmark.', '']
        (OUTPUT / 'comparison.md').write_text('\n'.join(lines))


def watch(variant):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / variant
    out.mkdir(exist_ok=True)
    lock = (out / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = out / 'watcher_status.json'
    if state.exists() and json.loads(state.read_text())['stage'] in ('COMPLETE', 'FAILED', 'EVALUATING'):
        raise RuntimeError('Existing terminal/interrupted evaluation; inspect before manual restart')
    campaign = json.loads((LOGS / 'campaign.json').read_text())
    job = campaign['jobs'][variant]
    gpu = job['gpu']

    def update(stage, **kwargs):
        dump(state, dict(stage=stage, variant=variant, watcher_pid=os.getpid(),
                         train_pid=job['pid'], gpu=gpu, updated_unix=time.time(), **kwargs))

    try:
        while True:
            if same_process(job['pid'], variant):
                update('WAITING_FOR_TRAINING')
                time.sleep(60)
                continue
            exp = 'a2w_z1_wbc_wheel_v2' + ('_precision' if variant == 'precision' else '')
            runs = list((LOGS / 'rsl_rl' / exp).glob(f'*_{variant}_train_seed42_20260926'))
            assert len(runs) == 1, f'Expected one formal run, found {runs}'
            checkpoint = runs[0] / 'model_49999.pt'
            assert checkpoint.is_file(), 'Training exited without final 50K checkpoint; not evaluating an intermediate model'
            usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                              '--format=csv,noheader,nounits'], text=True)
            memory = {int(row.split(',')[0]): int(row.split(',')[1]) for row in usage.strip().splitlines()}
            if memory[gpu] > 256:
                update('WAITING_FOR_GPU', memory_mib=memory[gpu])
                time.sleep(60)
                continue
            break
        import torch
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['iter'] == 49999, f"Wrong final iteration {saved.get('iter')}"
        del saved
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        manifest = dict(variant=variant, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                        seeds=SEEDS, num_envs=64, seconds_per_env=200,
                        domains=dict(DOMAINS), evaluation_task='A2W-Z1-WBC-WheelV2-Play',
                        eval_script_sha256=hashlib.sha256((ROOT / 'LeggedManip_Lab/scripts/eval_wheelv2_a100_domains.py').read_bytes()).hexdigest())
        dump(out / 'manifest.json', manifest)
        domains = {}
        completed = 0
        for domain, x_min in DOMAINS:
            results = []
            for seed in SEEDS:
                update('EVALUATING', domain=domain, seed=seed, completed_cases=completed, total_cases=6)
                result = run_case(variant, checkpoint, out / domain / f'seed{seed}', seed, x_min, gpu)
                assert result['checkpoint_sha256'] == digest
                results.append(result)
                completed += 1
            domains[domain] = aggregate(results)
            dump(out / domain / 'aggregate.json', domains[domain])
        dump(out / 'summary.json', {**manifest, 'requested_domains': dict(DOMAINS), 'domains': domains})
        update('COMPLETE', completed_cases=completed, total_cases=6)
        comparison_if_ready()
    except BaseException:
        update('FAILED', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['baseline', 'precision'])
    args = parser.parse_args()
    watch(args.variant)
