"""Offline provenance verification; no simulator, credentials or network required."""
import argparse
import ast
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--allow-relocated-urdf', action='store_true')
    a = p.parse_args()
    expected = json.loads((ROOT / 'docs/provenance/published_files.json').read_text())
    bad = []
    for name, digest in expected.items():
        path = ROOT / name
        if a.allow_relocated_urdf and name.endswith('/a2wz1_wbc.urdf'):
            # Verify exact original geometry and physics, ignoring only mesh directory.
            def normalize(data):
                r = ET.fromstring(data)
                for m in r.iter('mesh'):
                    m.set('filename', Path(m.get('filename')).name)
                return ET.tostring(r)
            original = ROOT / 'docs/provenance/a2wz1_wbc.training.urdf'
            if not path.is_file() or normalize(path.read_bytes()) != normalize(original.read_bytes()):
                bad.append(name)
        elif not path.is_file() or sha(path) != digest:
            bad.append(name)
    assert not bad, f'Hash mismatch or missing files: {bad}'
    count = 0
    for top in ('source', 'scripts', 'reproduction'):
        for path in (ROOT / top).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
            count += 1
    xml = ET.parse(ROOT / 'source/LeggedManip_Lab/LeggedManip_Lab/assets/a2w_z1/a2wz1_wbc.urdf').getroot()
    joints = [j for j in xml.findall('joint') if j.get('type') != 'fixed']
    assert len(joints) == 22
    assert len([j for j in joints if j.get('type') == 'continuous']) == 4
    assert next(j for j in xml.findall('joint') if j.get('name') == 'jointGripper').get('type') == 'fixed'
    for mesh in xml.iter('mesh'):
        assert (ROOT / 'source/LeggedManip_Lab/LeggedManip_Lab/assets/a2w_z1/meshes' / Path(mesh.get('filename')).name).is_file()
    print(json.dumps({'passed': True, 'verified_files': len(expected), 'python_syntax_checked': count,
                      'active_joints': len(joints), 'wheels': 4, 'gripper_fixed': True}, indent=2))


if __name__ == '__main__':
    main()
