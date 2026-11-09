"""DN census (part 2): what the fly brain sends down its neck.

Drive every sensory cell type and every visual-projection type (FlyWire v783, flywire_annotations v3.2.0), one side
at a time, at several rates, and record every descending neuron (DN), every brain motor neuron and the size of the
whole-brain response. One .npz per stimulus, so the census can be resumed.

    .venv-mlx/bin/python expression/census.py --variant v783_alfix_all --doses 25 50 100 200 --trials 10
    .venv-mlx/bin/python expression/census.py --variant v783_alfix_all --bilateral --tag _bilateral   # both sides at once
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlxrun  # noqa: E402
from wiring import annotations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'expression' / 'census'


def library(ann, min_per_side_sensory=3, min_per_side_vpn=5):
    """Stimuli = (cell_type, side) groups present on both sides with enough neurons."""
    lib = []
    for kind, sc, k in (('sensory', 'sensory', min_per_side_sensory), ('vpn', 'visual_projection', min_per_side_vpn)):
        a = ann[(ann.super_class == sc) & ann.cell_type.notna() & ann.side.isin(['left', 'right'])]
        n = a.groupby(['cell_type', 'side']).size().unstack(fill_value=0)
        keep = n[(n.get('left', 0) >= k) & (n.get('right', 0) >= k)].index
        for ct in sorted(keep):
            for side in ('left', 'right'):
                ids = a[(a.cell_type == ct) & (a.side == side)].root_id.astype(np.int64).tolist()
                cls = a[a.cell_type == ct].cell_class.mode()
                lib.append({'sid': f'{kind}:{ct}:{side}', 'kind': kind, 'cell_type': ct, 'side': side,
                            'cell_class': str(cls.iloc[0]) if len(cls) else '', 'ids': ids})
    return lib


def bilateral(lib):
    """The same cell types driven on both sides at once (one stimulus per cell type)."""
    by = {}
    for s in lib:
        by.setdefault((s['kind'], s['cell_type']), []).append(s)
    return [{'sid': f'{k}:{ct}:both', 'kind': k, 'cell_type': ct, 'side': 'both', 'cell_class': v[0]['cell_class'],
             'ids': [i for x in v for i in x['ids']]} for (k, ct), v in by.items()]


def readout_sets(ann, variant):
    ix = mlxrun.index(variant)
    def pick(mask):
        a = ann[mask & ann.root_id.isin(ix.keys())]
        return a.root_id.astype(np.int64).to_numpy(), np.array([ix[int(i)] for i in a.root_id], dtype=np.int64)
    dn_ids, dn_idx = pick(ann.super_class == 'descending')
    mn_ids, mn_idx = pick(ann.super_class == 'motor')
    return dn_ids, dn_idx, mn_ids, mn_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', default='v783_alfix_all')
    ap.add_argument('--doses', type=float, nargs='+', default=[25, 50, 100, 200])
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--kinds', nargs='+', default=['sensory', 'vpn'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--bilateral', action='store_true', help='drive each cell type on both sides at once')
    args = ap.parse_args()

    ann = annotations()
    lib = [s for s in library(ann) if s['kind'] in args.kinds]
    if args.bilateral:
        lib = bilateral(lib)
    seed0 = 20_000_000 if args.bilateral else 10_000_000                # bilateral trials never reuse one-sided seeds
    dn_ids, dn_idx, mn_ids, mn_idx = readout_sets(ann, args.variant)
    out = OUT / (args.variant + args.tag)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'library.json').write_text(json.dumps([{k: v for k, v in s.items() if k != 'ids'} |
                                                  {'n': len(s['ids']), 'n_driven': len(mlxrun.to_idx(s['ids'], args.variant))}
                                                  for s in lib], indent=0))
    np.savez(out / 'readouts.npz', dn_ids=dn_ids, mn_ids=mn_ids)
    print(f'{len(lib)} stimuli x {len(args.doses)} doses x {args.trials} trials; {len(dn_ids)} DNs, {len(mn_ids)} MNs',
          flush=True)
    t_start = time.time()
    for si, s in enumerate(lib):
        f = out / (s['sid'].replace(':', '__').replace('/', '_') + '.npz')
        if f.is_file():
            continue
        t0 = time.time()
        idx = mlxrun.to_idx(s['ids'], args.variant)
        es = 8 if len(idx) > 300 else 1
        dn, mn, nact, tot, mean_idx, mean_val = [], [], [], [], [], []
        for di, dose in enumerate(args.doses):
            c = mlxrun.run_counts(args.variant, idx, dose, n_trials=args.trials, t_s=1.0,
                                  seed=seed0 + 10_000 * si + 100 * di, edge_split=es)
            c[:, idx] = 0  # the driven neurons themselves are not part of the readout
            dn.append(c[:, dn_idx].astype(np.int16))
            mn.append(c[:, mn_idx].astype(np.int16))
            nact.append((c > 0).sum(1))
            tot.append(c.sum(1))
            m = c.mean(0)
            nz = np.nonzero(m)[0]
            mean_idx.append(nz.astype(np.int32))
            mean_val.append(m[nz].astype(np.float32))
        mi, mv = np.empty(len(mean_idx), dtype=object), np.empty(len(mean_val), dtype=object)
        mi[:], mv[:] = mean_idx, mean_val                             # always a 1-D ragged object array
        tmp = f.with_name(f.stem + '.tmp.npz')
        np.savez_compressed(tmp, doses=np.array(args.doses), dn=np.stack(dn), mn=np.stack(mn), n_active=np.stack(nact),
                            total=np.stack(tot), driven=idx, mean_idx=mi, mean_val=mv)
        os.replace(tmp, f)                                             # atomic: a killed run never leaves half a file
        el = time.time() - t_start
        print(f'[{si + 1}/{len(lib)}] {s["sid"]} n={len(idx)} active={[int(np.median(a)) for a in nact]} '
              f'DN>0={[int((d.mean(0) > 0).sum()) for d in dn]} {time.time() - t0:.1f}s (elapsed {el / 60:.1f} min)',
              flush=True)


if __name__ == '__main__':
    main()
