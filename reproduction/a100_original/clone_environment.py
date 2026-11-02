"""Resume the private environment copy in parallel on the shared filesystem.

rsync size+mtime checks repair an interrupted destination file. Never writes
to the source or uses hardlinks back to it. Existing unrelated paths untouched.
"""
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
root=Path(__file__).resolve().parent
source=Path('/data01/niuwa/sivan/workspace/locowheeledlegged-repro/runtime/env')
dest=root/'runtime/env'
split={'','lib','lib/python3.11','lib/python3.11/site-packages',
       'lib/python3.11/site-packages/isaacsim',
       'lib/python3.11/site-packages/isaacsim/extscache',
       'lib/python3.11/site-packages/isaacsim/exts',
       'lib/python3.11/site-packages/isaacsim/extsDeprecated',
       'lib/python3.11/site-packages/isaacsim/kit',
       'lib/python3.11/site-packages/isaacsim/kit/exts',
       'lib/python3.11/site-packages/isaacsim/kit/extscore'}
tasks=[]
def visit(relative):
    (dest/relative).mkdir(parents=True,exist_ok=True)
    for p in (source/relative).iterdir():
        r=p.relative_to(source)
        # Previous application's writable user data is not part of the runtime.
        # In particular its screenshot directory is root-owned on this server.
        if str(r) in ('lib/python3.11/site-packages/isaacsim/kit/data',
                      'lib/python3.11/site-packages/isaacsim/kit/logs',
                      'lib/python3.11/site-packages/isaacsim/kit/cache'):
            continue
        if str(r) in split and p.is_dir() and not p.is_symlink(): visit(r)
        else: tasks.append((p,dest/r.parent))
visit(Path(''))
def transfer(pair):
    p,parent=pair
    if p.name=='torch' and os.environ.get('WHEEL_WAIT_TORCH_PID'):
        proc=Path('/proc')/os.environ['WHEEL_WAIT_TORCH_PID']/'cmdline'
        while proc.exists():
            try: command=proc.read_bytes()
            except FileNotFoundError: break
            if b'rsync' not in command or str(p).encode() not in command: break
            time.sleep(1)
    subprocess.run(['rsync','-a','--',str(p),str(parent)+'/'],check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for i,_ in enumerate(pool.map(transfer,tasks),1):
        if i%100==0: print(f'Copied/verified {i}/{len(tasks)} entries',flush=True)
subprocess.run(['rsync','-a',str(source.parent/'compat'),str(root/'runtime')+'/'],check=True)
(root/'runtime/ENV_COPY_COMPLETE.json').write_text(json.dumps(dict(source=str(source),destination=str(dest),entries=len(tasks),hardlinks=False),indent=2)+'\n')
print('ENV_COPY_COMPLETE',flush=True)
