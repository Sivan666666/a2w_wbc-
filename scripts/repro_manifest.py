import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
lab=root.parent/"IsaacLab_5_1"
def git(path,*args):
    return subprocess.check_output(["git","-C",str(path),*args],text=True).strip()
files=list((root/"scripts").glob("repro*"))+list((root/"scripts").glob("prepare_a2w*"))
files+=list((root/"source").rglob("a2w_z1/*.py"))
sha={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
versions={}
for name in ("isaacsim","isaaclab","isaaclab-rl","torch","numpy","gymnasium","rsl-rl-lib","mujoco"):
    try: versions[name]=md.version(name)
    except md.PackageNotFoundError: versions[name]="missing"
result={"upstream_commit":git(root,"rev-parse","HEAD"),"isaaclab_commit":git(lab,"rev-parse","HEAD"),
    "python":sys.version,"prefix":sys.prefix,"versions":versions,"patch_hashes":sha,
    "container_base":"robodojo-ps1-runtime:cuda12.8 (8b79618b29c9), immutable shared image; separate new containers",
    "python_base":"/home/ps/miniconda3/envs/isaacsim5.1 mounted read-only for venv interpreter/stdlib",
    "upstream_diff_sha256":hashlib.sha256(subprocess.check_output(["git","-C",str(root),"diff","--binary"])).hexdigest(),
    "train":{"seed":42,"iterations":5000,"num_envs":4096,"algorithm":"PPO","start":"from scratch, no resume"},
    "isolation":"private venv + private copied Isaac Sim binaries; no modifications to donor environments",
    "render_only_overrides":"headless training without remote floor material/sky HDR; physics unchanged",
    "status":"manifest does not attest training completion; inspect per-run logs/checkpoints"}
out=root/"logs/reproduction_manifest.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
