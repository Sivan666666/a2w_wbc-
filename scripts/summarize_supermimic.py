"""Aggregate physical evaluations only; never substitute imagined metrics."""
import json
import statistics
import sys
from pathlib import Path

root=Path(sys.argv[1])
keys=('ee_native_m','ee_world_m','orientation_rad','vel_xy_mps','vel_yaw_rps')
rows={}
for method in ('bptt','ppo'):
    files=sorted(root.glob(f'eval_{method}_*/result.json'))
    if len(files)!=3:
        raise RuntimeError(f'{method}: expected 3 held-out evaluations, found {len(files)}')
    data=[json.loads(p.read_text()) for p in files]
    assert len({d['samples'] for d in data})==1
    rows[method]={k:dict(mean=statistics.mean(d[k] for d in data),
                         seed_std=statistics.stdev(d[k] for d in data)) for k in keys}
    rows[method]['failure_resets']=sum(d['failure_resets'] for d in data)
    rows[method]['eval_seeds']=[d['seed'] for d in data]
(root/'comparison.json').write_text(json.dumps(rows,indent=2))
lines=['# From-scratch A2W: BPTT vs PPO','',
       'One training seed; three held-out evaluation seeds. Not a multi-training-seed significance claim.','',
       '| Method | Native EE (cm) | World target (cm) | Orientation (deg) | XY (m/s) | Yaw (rad/s) | Failure resets |',
       '|---|---:|---:|---:|---:|---:|---:|']
for method,row in rows.items():
    values=[row[k]['mean']*scale for k,scale in zip(keys,(100,100,180/3.141592653589793,1,1))]
    lines.append('| '+method+' | '+' | '.join(f'{v:.4f}' for v in values)+f" | {row['failure_resets']} |")
(root/'comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
