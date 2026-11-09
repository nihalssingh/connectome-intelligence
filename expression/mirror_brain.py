"""Canalization or amplification? Left/right similarity of the DN output vs the rest of the brain, same stimuli.

For every cell type driven once on the left and once on the right (census, 100 Hz), compare the two responses in an
ipsi/contra frame over every neuron type with exactly one neuron per side, separately for descending neurons and for
each other super class. Trial means only (the census keeps whole-brain responses as means), sqrt(mean count) for all
classes. If the DN output is more symmetric than the neurons upstream of it, the output layer canalizes wiring
differences; if less, it amplifies them.

    .venv-mlx/bin/python expression/mirror_brain.py v783_alfix_all
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import CEN, OUT  # noqa: E402
from wiring import annotations  # noqa: E402

PACKS = Path(__file__).resolve().parents[1] / 'external' / 'drosophila-brain-mlx' / 'data' / 'pack'


def pairs_by_class(ann, ids):
    """One-per-side neuron types -> {super_class: int array [P, 2] of (left, right) pack indices}."""
    pos = {int(r): i for i, r in enumerate(ids)}
    a = ann[ann.cell_type.notna() & ann.side.isin(['left', 'right']) & ann.root_id.isin(pos.keys())]
    n = a.groupby(['cell_type', 'side']).size().unstack(fill_value=0)
    one = n[(n.get('left', 0) == 1) & (n.get('right', 0) == 1)].index
    a = a[a.cell_type.isin(one)]
    out = {}
    for (sc, ct), g in a.groupby(['super_class', 'cell_type']):
        if len(g) != 2 or set(g.side) != {'left', 'right'}:
            continue
        l, r = g[g.side == 'left'].root_id.iloc[0], g[g.side == 'right'].root_id.iloc[0]
        out.setdefault(sc, []).append((pos[int(l)], pos[int(r)]))
    return {k: np.array(v) for k, v in out.items()}


def main(variant='v783_alfix_all', dose=100.0):
    ann = annotations()
    ids = np.load(PACKS / variant / 'neuron_ids.npy')
    lib = json.loads((CEN / variant / 'library.json').read_text())
    pairs = pairs_by_class(ann, ids)
    by = {}
    for s in lib:
        by.setdefault((s['kind'], s['cell_type']), {})[s['side']] = s
    rows = []
    for (kind, ct), d in sorted(by.items()):
        if not {'left', 'right'} <= d.keys():
            continue
        vec = {}
        for side in ('left', 'right'):
            f = CEN / variant / (d[side]['sid'].replace(':', '__').replace('/', '_') + '.npz')
            if not f.is_file():
                break
            z = np.load(f, allow_pickle=True)
            di = list(z['doses']).index(dose)
            v = np.zeros(len(ids), np.float32)
            v[np.asarray(z['mean_idx'][di], dtype=np.int64)] = np.sqrt(np.asarray(z['mean_val'][di], dtype=np.float32))
            v[z['driven']] = np.nan                                  # driven cells are not part of the response
            vec[side] = v
        if len(vec) < 2:
            continue
        row = {'kind': kind, 'cell_type': ct, 'n_left': d['left'].get('n'), 'n_right': d['right'].get('n')}
        for sc, P in pairs.items():
            L, R = vec['left'], vec['right']
            Lf = np.r_[L[P[:, 0]], L[P[:, 1]]]                       # ipsi then contra, left stimulus
            Rf = np.r_[R[P[:, 1]], R[P[:, 0]]]                       # ipsi then contra, right stimulus
            ok = np.isfinite(Lf) & np.isfinite(Rf) & ((Lf > 0) | (Rf > 0))
            if ok.sum() < 5:
                continue                                             # too few responding pairs in this class
            one_silent = Lf[ok].max() == 0 or Rf[ok].max() == 0
            r = 0.0 if one_silent or Lf[ok].std() == 0 or Rf[ok].std() == 0 else float(np.corrcoef(Lf[ok], Rf[ok])[0, 1])
            row[sc] = {'r': r, 'one_side_silent': bool(one_silent), 'n_pairs_responding': int(ok.sum())}
        rows.append(row)
    classes = sorted(pairs)
    summary = {}
    for sc in classes:
        v = [r[sc]['r'] for r in rows if sc in r]
        if v:
            summary[sc] = {'n_one_per_side_types': int(len(pairs[sc])), 'n_stimulus_types': len(v),
                           'median_r': float(np.median(v)), 'iqr': [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}
    # paired comparison: stimuli where both the DNs and the central brain respond
    both = [(r['descending']['r'], r['central']['r']) for r in rows if 'descending' in r and 'central' in r]
    if both:
        b = np.array(both)
        summary['paired_DN_minus_central'] = {'n': len(b), 'median_diff': float(np.median(b[:, 0] - b[:, 1])),
                                              'frac_DN_more_symmetric': float((b[:, 0] > b[:, 1]).mean())}
    res = {'variant': variant, 'dose_Hz': dose, 'summary': summary, 'rows': rows}
    tf = OUT / 'mirror_trials' / f'{variant}.npz'
    if tf.is_file():
        res['noise_corrected'] = noise_corrected(tf, lib)
    (OUT / f'mirror_brain_{variant}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))
    if 'noise_corrected' in res:
        print(json.dumps(res['noise_corrected']['summary'], indent=1))


def noise_corrected(path, lib):
    """Same comparison on trial-level data (mirror_trials.py): r of trial means divided by sqrt of the two split-half
    (Spearman-Brown) reliabilities, per stimulus type and super class, over pairs responding to either side."""
    z = np.load(path, allow_pickle=True)
    keep, P, pc = z['keep'], z['pairs'], z['pair_class']
    col = np.searchsorted(keep, P)                                   # [n_pairs, 2] columns of (left, right)
    by = {}
    for s in lib:
        by.setdefault((s['kind'], s['cell_type']), {})[s['side']] = s['sid']
    rows = []
    for (kind, ct), d in sorted(by.items()):
        if not {'left', 'right'} <= d.keys() or 's__' + d['left'] not in z or 's__' + d['right'] not in z:
            continue
        L, R = z['s__' + d['left']].astype(np.float32), z['s__' + d['right']].astype(np.float32)
        row = {'kind': kind, 'cell_type': ct}
        for sc in np.unique(pc):
            c = col[pc == sc]
            Lf = np.concatenate([L[:, c[:, 0]], L[:, c[:, 1]]], 1)   # ipsi, contra (left stimulus)
            Rf = np.concatenate([R[:, c[:, 1]], R[:, c[:, 0]]], 1)   # ipsi, contra (right stimulus)
            valid = (Lf >= 0).all(0) & (Rf >= 0).all(0)              # drop driven cells
            Lf, Rf = np.sqrt(np.clip(Lf[:, valid], 0, None)), np.sqrt(np.clip(Rf[:, valid], 0, None))
            ok = (Lf.mean(0) > 0) | (Rf.mean(0) > 0)
            if ok.sum() < 5:
                continue
            Lf, Rf = Lf[:, ok], Rf[:, ok]
            T = Lf.shape[0]

            def rel(X):
                a, b = X[: T // 2].mean(0), X[T // 2:].mean(0)
                if a.std() == 0 or b.std() == 0:
                    return np.nan
                r = np.corrcoef(a, b)[0, 1]
                return 2 * r / (1 + r)
            if Lf.mean(0).max() == 0 or Rf.mean(0).max() == 0:     # silent on one side: no similarity
                row[str(sc)] = {'r': 0.0, 'r_nc': 0.0, 'rel_left': np.nan, 'rel_right': np.nan, 'one_side_silent': True,
                                'n_pairs_responding': int(ok.sum())}
                continue
            if Lf.mean(0).std() == 0 or Rf.mean(0).std() == 0:
                continue
            r = float(np.corrcoef(Lf.mean(0), Rf.mean(0))[0, 1])
            rl, rr = rel(Lf), rel(Rf)
            row[str(sc)] = {'r': r, 'rel_left': float(rl), 'rel_right': float(rr), 'n_pairs_responding': int(ok.sum()),
                            'one_side_silent': False, 'r_nc': float(r / np.sqrt(rl * rr)) if rl > 0 and rr > 0 else np.nan}
        rows.append(row)
    summary = {}
    for sc in np.unique(pc):
        v = [r[str(sc)] for r in rows if str(sc) in r]
        if v:
            nc = [x['r_nc'] for x in v if np.isfinite(x['r_nc'])]
            summary[str(sc)] = {'n_stimulus_types': len(v), 'median_r': float(np.median([x['r'] for x in v])),
                                'median_r_noise_corrected': float(np.median(nc)) if nc else np.nan,
                                'median_reliability': float(np.nanmedian([(x['rel_left'] + x['rel_right']) / 2 for x in v]))}
    both = [(r['descending']['r_nc'], r['central']['r_nc']) for r in rows
            if 'descending' in r and 'central' in r and np.isfinite(r['descending']['r_nc']) and np.isfinite(r['central']['r_nc'])]
    if both:
        b = np.array(both)
        summary['paired_DN_minus_central_noise_corrected'] = {
            'n': len(b), 'median_diff': float(np.median(b[:, 0] - b[:, 1])),
            'frac_DN_more_symmetric': float((b[:, 0] > b[:, 1]).mean())}
    return {'summary': summary, 'rows': rows}


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['v783_alfix_all']))
