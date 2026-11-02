"""Rebuild the publication's audit from a frozen training snapshot + official checkout.

This maintainer utility writes audit artifacts, not training code. No downloads.
"""
import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '4e12109d1eccb1d9da2e4a35f189fabea3f13677'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot-root', type=Path, required=True, help='Parent containing LeggedManip_Lab and IsaacLab')
    ap.add_argument('--upstream', type=Path, required=True, help='Official checkout at the recorded commit')
    a = ap.parse_args()
    assert subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    official = set(subprocess.check_output(['git', '-C', str(a.upstream), 'ls-tree', '-r', '--name-only', COMMIT], text=True).splitlines())
    dest = ROOT / 'docs/provenance'
    original = json.loads((dest / 'training_manifest.json').read_text())
    audit = []
    for name, sha in original['source_files_sha256'].items():
        p = a.snapshot_root / name
        assert p.is_file() and digest(p) == sha, f'Training source mismatch: {name}'
        audit.append(name)
    write_json(dest / 'training_hash_audit.json', {'passed': True, 'matched': len(audit),
               'scope': 'Every file recorded by the original source manifest (LeggedManip_Lab + IsaacLab); auxiliary scripts/assets not covered by that original manifest are inventoried separately.'})
    rows = []
    patch = []
    published = {}
    for top in ('source', 'scripts'):
        for p in sorted((ROOT / top).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:
                continue
            rel = str(p.relative_to(ROOT))
            s = a.snapshot_root / 'LeggedManip_Lab' / rel
            assert s.is_file() and p.read_bytes() == s.read_bytes(), f'Unexpected publication edit: {rel}'
            published[rel] = digest(p)
            exists = rel in official
            before = (a.upstream / rel).read_bytes() if exists else b''
            status = 'unchanged' if exists and before == p.read_bytes() else ('modified' if exists else 'added')
            rows.append({'path': rel, 'status': status,
                         'official_sha256': hashlib.sha256(before).hexdigest() if exists else None,
                         'snapshot_sha256': digest(s), 'published_sha256': digest(p), 'bytes': p.stat().st_size})
            if status != 'unchanged' and p.suffix in ('.py', '.sh', '.toml', '.txt', '.json', '.urdf'):
                patch.extend(difflib.unified_diff(before.decode().splitlines(keepends=True), p.read_text().splitlines(keepends=True),
                              fromfile='a/' + rel if exists else '/dev/null', tofile='b/' + rel))
    snapshot_files = [p for p in (a.snapshot_root / 'LeggedManip_Lab').rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc')]
    omitted = []
    for p in snapshot_files:
        rel = str(p.relative_to(a.snapshot_root / 'LeggedManip_Lab'))
        if not (ROOT / rel).is_file():
            omitted.append({'path': rel, 'sha256': digest(p), 'bytes': p.stat().st_size,
                            'reason': 'Generated USD/cache or non-A2W USD asset: not required for A2W URDF training; not a training modification.'})
    write_json(dest / 'file_diff_inventory.json', {'upstream_commit': COMMIT, 'files': rows, 'omitted_from_publication': omitted})
    (dest / 'training_changes.patch').write_text(''.join(patch))
    urdf = ROOT / 'source/LeggedManip_Lab/LeggedManip_Lab/assets/a2w_z1/a2wz1_wbc.urdf'
    (dest / 'a2wz1_wbc.training.urdf').write_bytes(urdf.read_bytes())
    for top in ('reproduction',):
        for p in sorted((ROOT / top).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                published[str(p.relative_to(ROOT))] = digest(p)
    published['docs/provenance/a2wz1_wbc.training.urdf'] = digest(dest / 'a2wz1_wbc.training.urdf')
    write_json(dest / 'published_files.json', published)
    lines = ['# 相对官方基础 commit 的逐文件清单', '', f'官方 commit：`{COMMIT}`。', '',
             '本清单覆盖发布的全部 source/scripts 文件；未列在下表的同路径文件与官方逐字节相同。',
             '完整哈希、未改文件与有意不发布的USD列表见 `file_diff_inventory.json`。', '',
             '| 状态 | 文件 | 字节数 |', '|---|---|---:|']
    lines.extend(f"| {r['status']} | `{r['path']}` | {r['bytes']} |" for r in rows if r['status'] != 'unchanged')
    lines += ['', '## 发布包装与历史脚本', '',
              '`reproduction/a100_original/` 为A100启动/审计/评测工具原样归档。',
              '`reproduction/train.sh`、`relocate_assets.py`、本审计工具和文档是2026-11-02发布时新增，不是50K训练期间的算法变动。',
              '历史scripts中B2检查、MuJoCo与20K续训工具均标明用途；它们不属于Precision 50K主训练路径。', '']
    (dest / 'FILE_CHANGES.md').write_text('\n'.join(lines))
    counts = {k: sum(r['status'] == k for r in rows) for k in ('unchanged', 'modified', 'added')}
    print(json.dumps({'snapshot_hashes_verified': len(audit), 'published_source_script_counts': counts,
                      'omitted_binary_cache_files': len(omitted), 'published_manifest_entries': len(published)}, indent=2))


if __name__ == '__main__':
    main()
