"""Evaluate one immutable Go2-aligned A2W checkpoint, without pausing or modifying training.

Invoke in a separate isolated container with the same runtime. Records shared-GPU
overlap explicitly; observed wall time is not a controlled hardware speedup.
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
import torch

p=argparse.ArgumentParser()
p.add_argument('--run',type=Path,required=True)
p.add_argument('--step',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
checkpoint=a.run/f'checkpoint_{a.step:07d}.pt'
actor=a.run/f'actor_{a.step:07d}.jit'
data=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert data['format']=='a2w-go2-aligned-v1' and data['step']==a.step
a.output.mkdir(parents=True,exist_ok=False)
status=dict(stage='STARTING',checkpoint=str(checkpoint),actor=str(actor),step=a.step, iteration=a.step, physical_steps_per_env=data['physical_steps_per_env'],
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            actor_sha256=hashlib.sha256(actor.read_bytes()).hexdigest(),
            checkpoint_wall_seconds=data['wall_seconds'],env_transitions=data['env_transitions'], imagined_transitions=data.get('imagined_transitions'),
            training_seed=data['config']['seed'],
            training_hardware='PS1 physical GPU 3; not A100',
            evaluation_gpu='PS1 physical GPU 3 shared with live training',
            overlap_started_unix=time.time(),completed=[])
del data
def save():
    temp=a.output/'status.tmp'
    temp.write_text(json.dumps(status,indent=2))
    os.replace(temp,a.output/'status.json')
save()
try:
    reports={}
    for domain,x_min in [('expanded',.1),('original',.55)]:
        results=[]
        for seed in (20260916,20260917,20260918):
            name=f'{domain}_{seed}';out=a.output/name
            status.update(stage='EVALUATING',active=name);save()
            command=[sys.executable,str(Path(__file__).with_name('eval_supermimic.py')),
                     '--actor',str(actor),'--output',str(out),'--seed',str(seed),
                     '--task','A2W-Z1-WBC-WheelV2-Play','--x_min',str(x_min),
                     '--num_envs','64','--seconds','200','--headless','--device','cuda:0']
            with (a.output/f'{name}.log').open('w') as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
            results.append(json.loads((out/'result.json').read_text()))
            status['completed'].append(name);save()
        assert len({r['samples'] for r in results})==1
        keys=('ee_native_m','ee_world_m','orientation_rad','vel_xy_mps','vel_yaw_rps')
        reports[domain]={k:dict(mean=statistics.mean(r[k] for r in results),
                               seed_std=statistics.stdev(r[k] for r in results)) for k in keys}
        reports[domain]['failure_resets']=sum(r['failure_resets'] for r in results)
    (a.output/'summary.json').write_text(json.dumps(dict(domains=reports,metadata=status),indent=2))
    status.update(stage='COMPLETE',overlap_finished_unix=time.time());save()
except Exception as exc:
    status.update(stage='FAILED',error=repr(exc),overlap_finished_unix=time.time());save()
    raise
