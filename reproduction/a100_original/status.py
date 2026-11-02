"""Read-only status of this two-run campaign (no automatic restarts)."""
import json
import re
from pathlib import Path
logs=Path('/data01/niuwa/sivan/logs/leggedmanip_wheelv2_20260926')
campaign=logs/'campaign.json'
if not campaign.exists():
    print(json.dumps({'state':'not_launched'}));raise SystemExit(1)
result=json.loads(campaign.read_text())
for variant,job in result['jobs'].items():
    cmd=Path('/proc')/str(job['pid'])/'cmdline'
    try: args=cmd.read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError: args=''
    job['process_running']=('scripts/rsl_rl/train.py' in args and f'{variant}_train_seed42_20260926' in args)
    with Path(job['log']).open('rb') as f:
        f.seek(0,2);end=f.tell();f.seek(max(0,end-96000));text=f.read().decode(errors='replace')
    text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
    iterations=re.findall(r'Learning iteration\s+(\d+)\s*/\s*(\d+)',text)
    job['latest_iteration']=int(iterations[-1][0]) if iterations else None
    job['target_iterations']=int(iterations[-1][1]) if iterations else None
    etas=re.findall(r'ETA:\s*([^\r\n]+)',text)
    job['log_eta']=etas[-1].strip() if etas else None
    job['recent_traceback']='Traceback (most recent call last)' in text
    exp='a2w_z1_wbc_wheel_v2'+('_precision' if variant=='precision' else '')
    runs=sorted((logs/'rsl_rl'/exp).glob(f'*_{variant}_train_seed42_20260926'))
    job['run_dir']=str(runs[-1]) if runs else None
    job['checkpoint_files']=sorted(p.name for p in runs[-1].glob('model_*.pt')) if runs else []
result['state']='running' if all(j['process_running'] and j['latest_iteration'] is not None and not j['recent_traceback'] for j in result['jobs'].values()) else 'inspect_jobs'
print(json.dumps(result,indent=2))
