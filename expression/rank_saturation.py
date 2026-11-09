"""Is the neck the bottleneck? Rank of the DN output vs number of stimuli, and DN vs whole-brain dimensionality.

1. Saturation: cross-validated rank (analyze.cvpca) of the DN output on random subsets of the stimulus library. If the
   rank keeps growing with the number of stimuli, the stimulus set, not the channel, limits it (the same logic as
   recovering an LLM's hidden size from its outputs: the number of queries must exceed the hidden size).
2. Bottleneck index: participation ratio of trial-mean responses (sqrt of mean counts) of the 1,299 DNs vs the other
   responding neurons, over the same stimulus x dose conditions. Means only (the census keeps trial-level data for DNs
   and motor neurons, not for the whole brain), so both sides carry the same trial-noise inflation.

    .venv-mlx/bin/python expression/rank_saturation.py v783_alfix_all
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import CEN, OUT, cvpca, ignition_flags, load  # noqa: E402

PACKS = Path(__file__).resolve().parents[1] / 'external' / 'drosophila-brain-mlx' / 'data' / 'pack'
SIZES = (15, 30, 60, 120, 240)


def saturation(Y, sizes, n_rep=5, seed=0):
    """Y: [S, C, T, F] (stimuli, conditions per stimulus, trials, DNs) -> cvPCA rank on random stimulus subsets."""
    rng = np.random.default_rng(seed)
    S, C, T, F = Y.shape
    rows = []
    for n in sizes:
        reps = 1 if n == S else n_rep
        rk, pr = [], []
        for r in range(reps):
            sub = np.sort(rng.choice(S, n, replace=False)) if n < S else np.arange(S)
            c = cvpca(Y[sub].reshape(n * C, T, F), seed=r)
            rk.append(c['rank_above_null'])
            pr.append(c['participation_ratio'])
        rows.append({'n_stimuli': int(n), 'n_conditions': int(n * C), 'rank_mean': float(np.mean(rk)),
                     'rank_sd': float(np.std(rk)), 'pr_mean': float(np.mean(pr)), 'n_rep': reps})
    return rows


def participation_ratio(X):
    """X: [C, N] condition x neuron; eigenvalues of the centred C x C Gram matrix, so N can be large."""
    X = X - X.mean(0, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh((X @ X.T).astype(np.float64)), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else 0.0


def whole_brain_means(variant, lib):
    """sqrt(mean count) of every neuron that ever responds, for every stimulus x dose -> ([S*D, M] float32, columns)."""
    d = CEN / variant
    files = [d / (s['sid'].replace(':', '__').replace('/', '_') + '.npz') for s in lib]
    cols = set()
    for f in files:
        for idx in np.load(f, allow_pickle=True)['mean_idx']:
            cols.update(np.asarray(idx, dtype=np.int64).tolist())
    cols = np.array(sorted(cols), dtype=np.int64)
    pos = {c: i for i, c in enumerate(cols)}
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        for idx, val in zip(z['mean_idx'], z['mean_val']):
            v = np.zeros(len(cols), np.float32)
            v[[pos[i] for i in np.asarray(idx, dtype=np.int64).tolist()]] = np.sqrt(np.asarray(val, dtype=np.float32))
            rows.append(v)
    return np.stack(rows), cols


def main(variant='v783_alfix_all'):
    lib, doses, dn, nact, dn_ids = load(variant)
    Y = np.sqrt(dn)                                                  # [S, D, T, F]
    S, D, T, F = Y.shape
    sizes = [s for s in SIZES if s < S] + [S]
    res = {'variant': variant, 'n_stimuli': S, 'doses': doses.tolist(), 'n_DN': F,
           'saturation_by_dose': {f'{int(d)}Hz': saturation(Y[:, i:i + 1], sizes) for i, d in enumerate(doses)},
           'saturation_all_doses': saturation(Y, sizes, seed=1)}
    # bottleneck index on trial means, same transform for DNs and for the rest of the brain; with and without the
    # stimuli that ignite the stereotyped LAL/VES/CRE state (they dominate whole-brain variance)
    ids = np.load(PACKS / variant / 'neuron_ids.npy')
    W, cols = whole_brain_means(variant, lib)                        # [S*D, M]
    is_dn = np.isin(ids[cols], dn_ids)
    flag, flagged = ignition_flags(lib, nact, doses, dn)
    rng = np.random.default_rng(2)
    res['bottleneck'] = {}
    for tag, keep in (('all_stimuli', np.ones(S, bool)), ('without_ignition', ~flag)):
        rows_c = np.repeat(keep, D)                                  # W rows are stimulus-major, D per stimulus
        Wk = W[rows_c]
        on = Wk.max(0) > 0
        dn_on, rest_on = on & is_dn, on & ~is_dn
        rest = Wk[:, rest_on]
        n_dn = int(dn_on.sum())
        pr_rand = [participation_ratio(rest[:, rng.choice(rest.shape[1], n_dn, replace=False)]) for _ in range(20)]
        res['bottleneck'][tag] = {'n_stimuli': int(keep.sum()), 'n_conditions': int(rows_c.sum()), 'n_DN_responding': n_dn,
                                  'n_rest_responding': int(rest_on.sum()),
                                  'pr_DN': participation_ratio(np.sqrt(dn[keep].mean(2)).reshape(int(keep.sum()) * D, F)),
                                  'pr_rest_of_brain': participation_ratio(rest),
                                  'pr_random_rest_matched_mean': float(np.mean(pr_rand)),
                                  'pr_random_rest_matched_sd': float(np.std(pr_rand))}
    res['bottleneck']['flagged'] = flagged
    res['bottleneck']['note'] = ('trial means, sqrt(mean count); participation ratio over stimulus x dose conditions; '
                                 'matched = random sets of responding non-DN neurons, as many as responding DNs')
    (OUT / f'rank_saturation_{variant}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'saturation_by_dose'}, indent=1))
    for k, v in res['saturation_by_dose'].items():
        print(k, [(r['n_stimuli'], round(r['rank_mean'], 1)) for r in v])


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['v783_alfix_all']))
