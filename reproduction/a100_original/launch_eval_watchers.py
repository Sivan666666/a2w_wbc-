"""Detach the two CPU-only waiters after inference-pipeline smoke passes."""
import fcntl
import json
import os
import subprocess
import sys
import time
from eval_after_training import ROOT, LOGS, OUTPUT, dump

OUTPUT.mkdir(parents=True, exist_ok=True)
lock = (OUTPUT / 'launch.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
launch = OUTPUT / 'launch.json'
assert not launch.exists(), 'Watchers previously launched; inspect, do not duplicate'
for variant in ('baseline', 'precision'):
    assert json.loads((LOGS / 'eval_pipeline_smoke_20260927' / variant / 'passed.json').read_text())['passed']
jobs = {}
dump(launch, {'stage': 'LAUNCHING', 'jobs': jobs})
for variant in ('baseline', 'precision'):
    logpath = OUTPUT / f'{variant}_watcher.log'
    with logpath.open('x') as log:
        p = subprocess.Popen([sys.executable, str(ROOT / 'eval_after_training.py'), variant],
                             cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
    jobs[variant] = dict(pid=p.pid, log=str(logpath), launched_unix=time.time())
    dump(launch, {'stage': 'LAUNCHED', 'jobs': jobs})
print(launch.read_text())
