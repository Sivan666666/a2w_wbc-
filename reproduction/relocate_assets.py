"""Explicit publication-only path migration; never changes robot geometry/physics.

The committed URDF is the exact A100 training snapshot (absolute mesh paths).
Run this once in a fresh clone before using Isaac's URDF importer. The diff is
intentional and all non-path XML content is asserted unchanged. Generated USD
must not be reused from an earlier location; a fresh clone includes no USD.
"""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'source/LeggedManip_Lab/LeggedManip_Lab/assets/a2w_z1'


def normalized_xml(text):
    root = ET.fromstring(text)
    for mesh in root.iter('mesh'):
        mesh.set('filename', Path(mesh.get('filename')).name)
    return ET.tostring(root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true', help='Check current mesh references without writing')
    a = p.parse_args()
    path = ASSET / 'a2wz1_wbc.urdf'
    old = path.read_text()
    for mesh in ET.fromstring(old).iter('mesh'):
        expected = ASSET / 'meshes' / Path(mesh.get('filename')).name
        if not expected.is_file():
            raise FileNotFoundError(expected)
        if a.check and Path(mesh.get('filename')) != expected:
            raise RuntimeError('Mesh path is not relocated: run reproduction/relocate_assets.py')
    if a.check:
        print('ASSET_PATHS_OK')
        return
    def replace(m):
        return m.group(1) + str(ASSET / 'meshes' / Path(m.group(2)).name) + m.group(3)
    new = re.sub(r'(<mesh\s+filename=")([^"]+)(")', replace, old)
    assert normalized_xml(old) == normalized_xml(new), 'Unexpected non-path XML edit'
    if old != new:
        if (ASSET / 'usd' / 'a2wz1_wbc.usd').exists():
            raise RuntimeError('Generated USD already exists; use a fresh clone to avoid stale USD')
        path.write_text(new)
    report = {'before_sha256': hashlib.sha256(old.encode()).hexdigest(),
              'after_sha256': hashlib.sha256(new.encode()).hexdigest(),
              'only_mesh_paths_changed': True, 'asset_dir': str(ASSET)}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
