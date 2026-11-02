"""Launch exactly two jobs after simulator audit and both PPO smoke passes."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
root=Path(__file__).resolve().parent
logs=Path('/data01/niuwa/sivan/logs/leggedmanip_wheelv2_20260926')
logs.mkdir(parents=True,exist_ok=True)
lock=(logs/'launch.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=logs/'campaign.json'
if status.exists(): raise SystemExit('campaign.json already exists; inspect before any restart')
assert json.loads((logs/'audit.json').read_text())['passed'] is True
for variant in ('baseline','precision'):
    result=json.loads((logs/f'{variant}_smoke_result.json').read_text())
    assert result['returncode']==0
    assert Path(result['checkpoint']).is_file()
memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
for row in memory.strip().splitlines():
    idx,used=map(int,row.split(','))
    if idx in (0,1) and used>100: raise SystemExit(f'GPU {idx} is occupied ({used}MiB); not starting')
jobs={}
status.write_text(json.dumps(dict(state='launching',jobs=jobs),indent=2)+'\n')
for variant,gpu in (('baseline',0),('precision',1)):
    path=logs/f'{variant}_train.log'
    with path.open('x') as log:
        process=subprocess.Popen(['bash',str(root/'run_training.sh'),variant,str(gpu),'train'],
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True,cwd=root)
    jobs[variant]=dict(pid=process.pid,gpu=gpu,log=str(path),started_unix=time.time(),
                       iterations=50000,seed=42,initialization='fresh')
    status.write_text(json.dumps(dict(state='launched_not_yet_verified',jobs=jobs),indent=2)+'\n')
print(status.read_text())
