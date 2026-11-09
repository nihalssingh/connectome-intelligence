"""Download the third-party input data that this repository does not store (about 80 MB; standard library only).

1. FlyWire cell-type annotations (github.com/flyconnectome/flywire_annotations), pinned to the commits used here and
   checked by SHA-256: v3.2.0 (root IDs of FlyWire v783) and v1.1.0 (root IDs of v630).
2. The authors' archived simulation outputs of Shiu et al. 2024 (Edmond, doi:10.17617/3.CZODIW): the Fig. 1D sweep
   and three replicate sweeps, about 26 MB pulled out of a 4.5 GB zip by HTTP range requests (stage1/fetch_archive.py).

    python3 fetch_data.py
"""
import hashlib
import runpy
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = 'https://raw.githubusercontent.com/flyconnectome/flywire_annotations'
ANNOTATIONS = [
    # (commit, file in the repo, local path, sha256)
    ('a83b2776d60d5764cef36b927f5f9679c16c47a2',  # tag v3.2.0
     'supplemental_files/Supplemental_file1_neuron_annotations.tsv',
     'data/flywire_annotations/Supplemental_file1_neuron_annotations.tsv',
     'b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd'),
    ('df6bb136f5b3d91c3992df4e8de2642329e2a384',  # tag v1.1.0
     'supplemental_files/Supplemental_file1_annotations.tsv',
     'data/flywire_annotations_v1.1.0/Supplemental_file1_annotations.tsv',
     '55c99c61eecf8db6cc36f1a684b35e4c4208afbab02197d753ccc8dc2a6e2e76'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    for commit, src, dest, digest in ANNOTATIONS:
        dest = ROOT / dest
        if dest.is_file() and sha256(dest) == digest:
            print(f'ok (already present) {dest.relative_to(ROOT)}')
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.part')
        urllib.request.urlretrieve(f'{RAW}/{commit}/{src}', tmp)
        if sha256(tmp) != digest:
            tmp.unlink()
            raise SystemExit(f'SHA-256 mismatch for {src} at {commit[:12]}; not saved')
        tmp.replace(dest)
        print(f'ok {dest.relative_to(ROOT)}')
    runpy.run_path(str(ROOT / 'stage1' / 'fetch_archive.py'), run_name='__main__')


if __name__ == '__main__':
    main()
