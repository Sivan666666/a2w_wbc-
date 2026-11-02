"""Produce provenance for the isolated snapshot and explicit two-run experiment."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
files={}
for name in ('LeggedManip_Lab/source','IsaacLab/source'):
    for p in sorted((root/name).rglob('*')):
        if p.suffix in ('.py','.urdf','.toml') and '__pycache__' not in p.parts:
            files[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
out=dict(
    original_leggedmanip_commit='4e12109d1eccb1d9da2e4a35f189fabea3f13677',
    isaaclab_commit='47c9f95eeab90dc3611981d894d59f163191f5b1',
    source='PS1 existing WheelV2 snapshot; adapters included, not pristine upstream',
    original_wheelv2_cfg_sha256='b49dbf2839cbe9d8acc2d90d3622732a7697c069f9f579129d1981a3168cf229',
    initialization='fresh, agent.resume=false; never smoke checkpoints',
    iterations=50000,num_envs=4096,seed=42,policy_hz=50,physics_hz=200,
    baseline=dict(task='A2W-Z1-WBC-WheelV2',gpu=0,x=[.55,.85],reward='unchanged'),
    precision=dict(task='A2W-Z1-WBC-WheelV2-Precision',gpu=1,x=[.1,.85],
        reward='retain 4.5*exp(-e^2/0.1), add exp(-e^2/0.01)+1.5*exp(-e^2/0.0025)'),
    comparison_note='Two changes in precision run; not a one-factor reward ablation. Evaluate on shared and expanded ranges separately.',
    final_checkpoint='model_49999.pt (zero-based iterations)',
    source_files_sha256=files,
)
out['source_manifest_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
(root/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='source_files_sha256'},indent=2))
