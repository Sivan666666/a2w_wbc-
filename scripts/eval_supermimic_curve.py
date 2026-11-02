"""Small held-out physical evaluations of six saved checkpoints per method.

Single evaluation seed / 20 s is a learning-curve diagnostic, not the final
3-seed 200 s benchmark. X values are actual saved steps, never rounded budgets.
"""
import json
import subprocess
import sys
from pathlib import Path
import torch

root=Path(sys.argv[1]).resolve()
rows=[]
for method in ('bptt','ppo'):
    paths=sorted((root/method).glob('actor_[0-9]*.jit'))
    candidates={int(p.stem.split('_')[-1]):p for p in paths}
    if method=='ppo':
        candidates[0]=root/'bptt/actor_0000000.jit'  # identical paired random Actor
        index={x['step']:x for x in map(json.loads,(root/'ppo/checkpoint_index.jsonl').read_text().splitlines())}
    total=max(candidates)
    selected=sorted({min(candidates,key=lambda n:abs(n-total*f/5)) for f in range(6)})
    for step in selected:
        out=root/f'curve_{method}_{step:07d}'
        subprocess.run([sys.executable,str(Path(__file__).with_name('eval_supermimic.py')),
                        '--actor',str(candidates[step]),'--output',str(out),'--num_envs','64',
                        '--seconds','20','--seed','20261111','--headless','--device','cuda:0'],check=True)
        score=json.loads((out/'result.json').read_text())
        if method=='bptt' or step==0:
            checkpoint=torch.load(root/f'bptt/checkpoint_{step:07d}.pt',map_location='cpu',weights_only=False)
            timing={k:checkpoint[k] for k in ('env_transitions','wall_seconds')}
            del checkpoint
        else:
            timing={k:index[step][k] for k in ('env_transitions','wall_seconds')}
        rows.append(dict(method=method,step=step,**timing,**score))
(root/'learning_curves.json').write_text(json.dumps(rows,indent=2))
