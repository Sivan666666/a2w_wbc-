"""Isolated mode pipeline: three seeds, no checkpoint or training writes."""
import json
import subprocess
import sys
import traceback
from pathlib import Path

mode, root = sys.argv[1], Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=False)
checkpoint = Path('logs/rsl_rl/a2w_z1_wbc/2026-09-14_14-25-49_repro_seed42_20260914_142542/model_4999.pt').resolve()
def status(**data):
    (root/'status.json').write_text(json.dumps(data,indent=2)+'\n')
try:
    for seed in (20260916,20260917,20260918):
        command = [sys.executable,'scripts/eval_a2w_precision.py','--mode',mode,'--seed',str(seed),
                   '--checkpoint',str(checkpoint),'--output',str(root/f'seed{seed}'),
                   '--num_envs','64','--trials','10','--duration','20','--headless','--device','cuda:0']
        status(stage='EVALUATING',seed=seed,mode=mode,command=command)
        with (root/f'seed{seed}.log').open('w') as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert json.loads((root/f'seed{seed}/status.json').read_text())['stage']=='COMPLETE'
    import numpy as np
    from a2w_precision_metrics import summarize
    arrays={}
    summaries=[]
    for seed in (20260916,20260917,20260918):
        summaries.append(json.loads((root/f'seed{seed}/summary.json').read_text()))
        with np.load(root/f'seed{seed}/trace.npz') as d:
            for k in d.files: arrays.setdefault(k,[]).append(d[k])
    data={k:np.concatenate(v,axis=0) for k,v in arrays.items()}
    result=summarize(data,mode)
    result['seeds']=summaries
    (root/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    status(stage='COMPLETE',mode=mode)
except BaseException:
    status(stage='FAILED',traceback=traceback.format_exc())
    raise
