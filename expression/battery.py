"""Behaviour battery + DN probe set for any wiring pack (real, corrected or regrown brains).

Validated circuit tests from Shiu et al. 2024 (stimulus sets from their figures.ipynb, mapped to v783 in stimsets.py):
sugar -> MN9 at 50/100/200 Hz, bitter and Ir94e suppression of sugar, water -> MN9, and grooming specificity
(JON-CE but not JON-F drives aBN1). Probe set: 24 cell types (the 23 present in the census library, x both sides = 46 probes) at 100 Hz, so a
regrown brain's DN 'words' can be compared with the real brain's. Seeds depend only on the condition, so every brain
sees the same input spike trains (common random numbers); a seed offset >= N_TRIALS (e.g. 500) gives an
independent replicate.

    .venv-mlx/bin/python expression/battery.py v783_alfix_all
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage1'))
import mlxrun  # noqa: E402
from census import library, readout_sets  # noqa: E402
from stimsets import ABN1, CONDITIONS, MN9C, MN9I  # noqa: E402
from wiring import annotations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'expression' / 'battery'
PROBE_TYPES = ['LB3c', 'LB3d', 'LB1a', 'LB2a', 'JO-CM', 'JO-EV1', 'JO-FV', 'JO-B2', 'ORN_DM1', 'ORN_DA1', 'ORN_VA1v',
               'ORN_DL5', 'TRN_VP2', 'HRN_VP4', 'BM_Taste', 'BM_Ant', 'LC4', 'LPLC2', 'LC10a', 'LC16', 'LPLC1', 'LC6',
               'LC11', 'LC9']
N_TRIALS = 10


def probes(lib):
    by = {(s['cell_type'], s['side']): s for s in lib}
    out = []
    for t in PROBE_TYPES:
        if (t, 'left') in by and (t, 'right') in by:
            out += [by[(t, 'left')], by[(t, 'right')]]
    return out


def main(variant, seed_offset=0):
    # trial k of a condition uses seed + k, so a replicate needs an offset of at least N_TRIALS to be independent
    if 0 < seed_offset < N_TRIALS or seed_offset >= 1000 - N_TRIALS:
        raise ValueError(f'seed_offset must be in [{N_TRIALS}, {1000 - N_TRIALS}) for an independent replicate')
    t0 = time.time()
    ix = mlxrun.index(variant)
    ann = annotations()
    res = {'variant': variant, 'seed_offset': seed_offset, 'conditions': {}}
    read = {'MN9_contra': ix[MN9C], 'MN9_ipsi': ix[MN9I], 'aBN1': ix[ABN1]}
    for k, (name, (nids, rate)) in enumerate(CONDITIONS.items()):
        missing = [i for i in nids if int(i) not in ix]
        if missing:
            raise ValueError(f'{name}: {len(missing)} IDs not in pack {variant}: {missing[:3]}')
        idx = mlxrun.to_idx(nids, variant)
        c = mlxrun.run_counts(variant, idx, rate, n_trials=N_TRIALS, t_s=1.0, seed=5_000_000 + 1000 * k + seed_offset)
        row = {r: float(c[:, i].mean()) for r, i in read.items()}
        row.update({r + '_sd': float(c[:, i].std()) for r, i in read.items()})
        row['n_active'] = float(np.median((c > 0).sum(1)))
        res['conditions'][name] = row
    lib = library(ann)
    pr = probes(lib)
    dn_ids, dn_idx, _, _ = readout_sets(ann, variant)
    dn = np.zeros((len(pr), N_TRIALS, len(dn_idx)), dtype=np.int16)
    nact = np.zeros((len(pr), N_TRIALS), dtype=np.int32)
    for j, s in enumerate(pr):
        idx = mlxrun.to_idx(s['ids'], variant)
        si = [x['sid'] for x in lib].index(s['sid'])
        c = mlxrun.run_counts(variant, idx, 100, n_trials=N_TRIALS, t_s=1.0,
                              seed=6_000_000 + 1000 * si + seed_offset, edge_split=8 if len(idx) > 300 else 1)
        c[:, idx] = 0
        dn[j] = c[:, dn_idx]
        nact[j] = (c > 0).sum(1)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f'{variant}' + (f'_rep{seed_offset}' if seed_offset else '')
    np.savez_compressed(OUT / f'{tag}.npz', dn=dn, n_active=nact, probes=np.array([s['sid'] for s in pr]),
                        dn_ids=dn_ids)
    res['probe_median_active'] = float(np.median(nact))
    res['seconds'] = round(time.time() - t0, 1)
    (OUT / f'{tag}.json').write_text(json.dumps(res, indent=1))
    c = res['conditions']
    print(f"{tag}: MN9 contra 50/100/200 = {c['sugar_50']['MN9_contra']:.1f}/{c['sugar_100']['MN9_contra']:.1f}/"
          f"{c['sugar_200']['MN9_contra']:.1f}; +bitter {c['sugar_bitter_100']['MN9_contra']:.1f}; "
          f"+Ir94e {c['sugar_ir94e_100']['MN9_contra']:.1f}; water {c['water_100']['MN9_contra']:.1f}; "
          f"aBN1 CE/F {c['JON_CE_100']['aBN1']:.1f}/{c['JON_F_100']['aBN1']:.1f}; probes median active "
          f"{res['probe_median_active']:.0f}; {res['seconds']} s", flush=True)


if __name__ == '__main__':
    for v in sys.argv[1].split(','):
        main(v, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
