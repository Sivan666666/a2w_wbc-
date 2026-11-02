"""PS1 host watchdog: wait for the original container, then launch one resume job."""
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/mnt/sda/ps/leggedmanip-lab/LeggedManip_Lab')
OLD = 'leggedmanip-a2w-wheel-v2-train-20260920'
NEW = 'leggedmanip-a2w-wheel-v2-resume20k-20260920'
RUN = ROOT / 'logs/rsl_rl/a2w_z1_wbc_wheel_v2/2026-09-20_11-58-34_wheel_v2_seed42_20260920'
OUT = ROOT / 'logs/rsl_rl/a2w_z1_wbc_wheel_v2/continue_5k_to20k_seed42_20260920'
AUDIT = ROOT / 'logs/wheel_v2_chain_20k_20260920'
AUDIT.mkdir(parents=True, exist_ok=True)
lock = (AUDIT / 'watcher.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def call(*args):
    return subprocess.check_output(args, text=True).strip()


def status(stage, **kwargs):
    value = dict(stage=stage, time_utc=datetime.now(timezone.utc).isoformat(),
                 source_container=OLD, resume_container=NEW, resume_output=str(OUT), **kwargs)
    temp = AUDIT / 'status.tmp'
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(AUDIT / 'status.json')
    print(json.dumps(value), flush=True)


try:
    validation = json.loads((ROOT / 'logs/wheel_v2_resume_validation_20260920/status.json').read_text())
    assert validation['stage'] == 'VALIDATED' and validation['loaded_actor_critic_optimizer_exact']
    # Explicit single-launch behavior, even if someone restarts the watcher.
    if subprocess.run(['docker', 'inspect', NEW], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('Resume container already exists; refusing a duplicate launch')
    paths = [ROOT / 'scripts/resume_wheel_v2_20k.py']
    paths += list((ROOT / 'source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/a2w_z1_wheel_v2').glob('*.py'))
    source = ROOT / 'source/LeggedManip_Lab/LeggedManip_Lab'
    paths += [source / rel for rel in (
        'assets/a2w_z1/a2w_z1_articulation_cfg.py',
        'tasks/manager_based/leggedmanip_lab/config/a2w_z1/wbc_env_cfg.py',
        'tasks/manager_based/leggedmanip_lab/config/b2_z1/wbc_env_cfg.py',
        'tasks/manager_based/leggedmanip_lab/config/b2_z1/agents/rsl_rl_ppo_cfg.py',
        'tasks/manager_based/leggedmanip_lab/leggedmanip_lab_env_cfg.py',
        'tasks/manager_based/leggedmanip_lab/mdp/rewards.py',
        'tasks/manager_based/leggedmanip_lab/mdp/curriculums.py',
    )]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    status('WAITING_FOR_5K', source_hashes=hashes)
    code = int(call('docker', 'wait', OLD).splitlines()[-1])
    assert code == 0, f'Original training exit code {code}; no automatic resume'
    ckpt = RUN / 'model_4999.pt'
    assert ckpt.is_file() and ckpt.stat().st_size > 0, 'Missing final 5K checkpoint'
    for p, digest in hashes.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest, f'Code changed while waiting: {p}'
    assert not OUT.exists(), 'Resume output already exists'
    status('LAUNCHING', checkpoint=str(ckpt))
    command = f'source {ROOT}/scripts/repro_env.sh && python scripts/resume_wheel_v2_20k.py --checkpoint {ckpt} --output {OUT} --total_iterations 20000 --expected_previous_iterations 5000 --num_envs 4096 --headless --device cuda:0'
    cid = call('bash', str(ROOT / 'scripts/repro_container.sh'), '2', NEW, 'bash', '-lc', command)
    status('RESUME_PROCESS_STARTED', container_id=cid, checkpoint=str(ckpt))
    # Save full stage-two console output on HDD, beyond Docker's rotation window.
    with (AUDIT / 'resume_train.log').open('a') as log:
        tail = subprocess.Popen(['docker', 'logs', '--follow', NEW], stdout=log, stderr=subprocess.STDOUT)
        code = int(call('docker', 'wait', NEW).splitlines()[-1])
        tail.wait(timeout=60)
    assert code == 0, f'Resume training exit code {code}'
    result = json.loads((OUT / 'status.json').read_text())
    assert result['stage'] == 'COMPLETE'
    status('COMPLETE_20K', final_checkpoint=result['final_checkpoint'])
except Exception as exc:
    status('FAILED', error=repr(exc))
    raise
