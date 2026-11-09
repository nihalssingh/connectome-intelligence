"""Trial-level responses of every one-per-side neuron type, for a noise-corrected canalization test.

Re-runs the census's 100 Hz condition for every cell type in the library (both sides) with the census's own seeds, so
the DN counts are identical to the census, and keeps per-trial counts of every neuron that belongs to a type with
exactly one neuron per side (all super classes). Output: results/expression/mirror_trials/<variant>.npz

    .venv-mlx/bin/python expression/mirror_trials.py v783_alfix_all
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlxrun  # noqa: E402
from census import library  # noqa: E402
from mirror_brain import PACKS, pairs_by_class  # noqa: E402
from wiring import annotations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'expression' / 'mirror_trials'


def main(variant='v783_alfix_all', dose=100.0, doses=(25.0, 50.0, 100.0, 200.0), trials=10):
    ann = annotations()
    lib = library(ann)
    ids = np.load(PACKS / variant / 'neuron_ids.npy')
    pairs = pairs_by_class(ann, ids)
    cls = np.concatenate([[k] * len(v) for k, v in pairs.items()])
    P = np.concatenate([pairs[k] for k in pairs])                      # [n_pairs, 2] (left, right) pack indices
    keep = np.unique(P.ravel())
    di = list(doses).index(dose)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f'{variant}.npz'
    done = dict(np.load(f, allow_pickle=True)) if f.is_file() else {}
    counts = {k: v for k, v in done.items() if k.startswith('s__')}
    t0 = time.time()
    for si, s in enumerate(lib):
        key = 's__' + s['sid']
        if key in counts:
            continue
        idx = mlxrun.to_idx(s['ids'], variant)
        es = 8 if len(idx) > 300 else 1
        c = mlxrun.run_counts(variant, idx, dose, n_trials=trials, t_s=1.0,
                              seed=10_000_000 + 10_000 * si + 100 * di, edge_split=es)   # census seed for this dose
        c[:, idx] = -1                                                  # mark driven cells
        counts[key] = c[:, keep].astype(np.int16)
        if (si + 1) % 20 == 0 or si + 1 == len(lib):
            np.savez_compressed(f, keep=keep, pairs=P, pair_class=cls, **counts)
            print(f'[{si + 1}/{len(lib)}] {(time.time() - t0) / 60:.1f} min', flush=True)
    np.savez_compressed(f, keep=keep, pairs=P, pair_class=cls, **counts)
    (OUT / f'{variant}.json').write_text(json.dumps({'variant': variant, 'dose_Hz': dose, 'trials': trials,
                                                     'n_stimuli': len(counts), 'n_neurons_kept': int(len(keep))}))


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['v783_alfix_all']))
