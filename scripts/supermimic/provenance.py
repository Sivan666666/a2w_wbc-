"""Hash actual training source and selected installed dependencies, no secrets."""
import hashlib
import importlib.metadata
import json
from pathlib import Path


def capture(output):
    import LeggedManip_Lab
    root=Path(__file__).resolve().parents[2]
    files={}
    for folder in ('scripts','source','reproduction'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.py','.sh','.urdf') and '__pycache__' not in str(p):
                files[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    versions={}
    for package in ('torch','isaaclab','isaacsim','rsl-rl-lib','tensordict'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='not installed as distribution'
    module=Path(LeggedManip_Lab.__file__).resolve()
    assert module.is_relative_to(root), f'Wrong source package imported: {module}'
    payload=dict(root=str(root),imported_package=str(module),source_sha256=files,versions=versions)
    (Path(output)/'source_manifest.json').write_text(json.dumps(payload,indent=2))
