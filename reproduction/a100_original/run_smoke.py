"""Run one PPO smoke, require a newly created final checkpoint and zero exit."""
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
logs=Path('/data01/niuwa/sivan/logs/leggedmanip_wheelv2_20260926')
variant=sys.argv[1];gpu=int(sys.argv[2])
assert variant in ('baseline','precision')
experiment='a2w_z1_wbc_wheel_v2'+('_precision' if variant=='precision' else '')
pattern=f'*_{variant}_smoke_seed42_20260926/model_2.pt'
parent=logs/'rsl_rl'/experiment
existing=set(parent.glob(pattern))
with (logs/f'{variant}_smoke.log').open('w') as log:
    p=subprocess.run(['bash',str(root/'run_training.sh'),variant,str(gpu),'smoke'],
                     stdout=log,stderr=subprocess.STDOUT)
found=set(parent.glob(pattern))-existing
result=dict(returncode=p.returncode,checkpoint=str(next(iter(found))) if len(found)==1 else None)
(logs/f'{variant}_smoke_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
raise SystemExit(0 if p.returncode==0 and len(found)==1 else 1)
