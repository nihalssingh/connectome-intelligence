"""Check rebuilt MLX packs against the stored hashes in data/pack_manifests/ (standard library only).

Each stored manifest is the one the port's compiler wrote for our packs; it records the SHA-256 of every array file.
A pack that has not been built is reported as missing.

    python3 expression/check_packs.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / 'external' / 'drosophila-brain-mlx' / 'data' / 'pack'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    status = {}
    for m in sorted((ROOT / 'data' / 'pack_manifests').glob('*/manifest.json')):
        name = m.parent.name
        arrays = json.loads(m.read_text())['arrays']
        if not (PACKS / name).is_dir():
            status[name] = 'missing'
            continue
        bad = [a for a, v in arrays.items()
               if not (PACKS / name / v['file']).is_file() or sha256(PACKS / name / v['file']) != v['file_sha256']]
        status[name] = 'OK' if not bad else 'differs: ' + ', '.join(bad)
    for name, s in status.items():
        print(f'{name:45s} {s}')
    n_ok = sum(s == 'OK' for s in status.values())
    print(f'\n{len(status)} packs: {n_ok} OK, {sum(s == "missing" for s in status.values())} missing, '
          f'{len(status) - n_ok - sum(s == "missing" for s in status.values())} differ')
    return bool(status) and n_ok == len(status)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
