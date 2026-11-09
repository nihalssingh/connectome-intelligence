"""Command-neuron recruitment check against Braun et al. 2024 (Nature 630:686, 'Descending networks transform command
signals into population motor control').

Braun et al. activated DN sets optogenetically and imaged GNG DN axons: recruitment DNp09 > aDN2 > MDN (Fig. 2i). They
call DNs with many DN partners 'broadcasters' (behaviour degrades in headless flies) and those with few 'standalone'
(behaviour persists headless). Here every set is driven bilaterally in the whole-brain LIF and we count the other DNs
that fire. Root IDs: aDN1, aDN2 and MDN as listed in Braun et al.'s Methods; the rest by FlyWire cell type. GNG DNs
(types named DNg..., 841 of 1,299) are Braun et al.'s connectome stand-in for the imaged cells, a superset of them.

    .venv-mlx/bin/python expression/braun.py v783_alfix_all
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlxrun  # noqa: E402
from wiring import annotations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BROADCAST, STANDALONE = 'broadcaster', 'standalone'
# (name, selector, Braun class, downstream DNs reported by Braun et al.)
SETS = [
    ('DNp09', ('type', 'DNp09'), BROADCAST, 32),
    # Braun's left aDN2 (v630 720575940624220925) has a new root ID in v783; both v783 DNge078 cells are used
    ('aDN2', ('ids', [720575940636308075, 720575940629806974]), BROADCAST, 23),
    ('DNp42', ('type', 'DNp42'), BROADCAST, 29),
    ('aDN1', ('ids', [720575940616185531, 720575940624319124]), BROADCAST, 26),
    ('DNa01', ('type', 'DNa01'), BROADCAST, 25),
    ('DNb02', ('type', 'DNb02'), BROADCAST, 20),
    ('DNa02', ('type', 'DNa02'), BROADCAST, 18),
    ('MDN', ('type', 'MDN'), STANDALONE, 14),
    ('giant fibre (DNp01)', ('type', 'DNp01'), STANDALONE, 4),
    ('oviDN', ('prefix', 'oviDN'), STANDALONE, 4),
    ('DNg11', ('type', 'DNg11'), STANDALONE, 4),
    ('DNg14', ('type', 'DNg14'), STANDALONE, 0),
]
RATES = [25, 50, 100, 200]


def select(ann, sel):
    kind, v = sel
    if kind == 'ids':
        return [int(i) for i in v]
    if kind == 'type':
        return ann[ann.cell_type == v].root_id.astype(np.int64).tolist()
    return ann[ann.cell_type.astype(str).str.startswith(v)].root_id.astype(np.int64).tolist()


def monosynaptic_dn_partners(con, dn_set, driven, thr=5):
    e = con[con.Presynaptic_ID.isin(driven) & (con.Connectivity >= thr) & con.Postsynaptic_ID.isin(dn_set)]
    return int(e.Postsynaptic_ID[~e.Postsynaptic_ID.isin(driven)].nunique())


def main(variant='v783_alfix_all', n_trials=10):
    ann = annotations()
    con = pd.read_parquet(ROOT / 'external/Drosophila_brain_model/Connectivity_783.parquet',
                          columns=['Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity'])
    ix = mlxrun.index(variant)
    dn = ann[(ann.super_class == 'descending') & ann.root_id.isin(ix.keys())]
    dn_ids = dn.root_id.astype(np.int64).to_numpy()
    dn_idx = np.array([ix[int(i)] for i in dn_ids])
    gng = dn.cell_type.astype(str).str.startswith('DNg').to_numpy()
    rows, per_trial = [], {}
    for k, (name, sel, cls, braun_n) in enumerate(SETS):
        ids = [i for i in select(ann, sel) if i in ix]
        idx = mlxrun.to_idx(ids, variant)
        driven = np.isin(dn_ids, ids)
        # this code's union count over the set's cells at >= 5 synapses (it does not reproduce Braun's own counts)
        row = {'set': name, 'n_cells': len(idx), 'braun_class': cls, 'braun_downstream_DNs': braun_n,
               'connectome_downstream_DNs_ge5': monosynaptic_dn_partners(con, set(dn_ids.tolist()), set(ids))}
        for r in RATES:
            c = mlxrun.run_counts(variant, idx, r, n_trials=n_trials, t_s=1.0, seed=777_000 + 1000 * k + r)
            rates = c[:, dn_idx].astype(float)  # spikes in 1 s = Hz
            m = rates.mean(0)
            rec = (m >= 1.0) & ~driven
            row[f'recruited_{r}Hz'] = int(rec.sum())
            row[f'recruited_gng_{r}Hz'] = int((rec & gng).sum())
            row[f'summed_rate_{r}Hz'] = float(m[~driven].sum())
            row[f'n_active_{r}Hz'] = int(np.median((c > 0).sum(1)))
            # threshold sensitivity and a bootstrap over trials (95% CI) for the GNG count (stand-in for the imaged cells)
            for thr in (0.5, 2.0):
                row[f'recruited_gng_{r}Hz_thr{thr}'] = int(((m >= thr) & ~driven & gng).sum())
            rng = np.random.default_rng(k * 1000 + r)
            boot = [int(((rates[rng.integers(0, n_trials, n_trials)].mean(0) >= 1.0) & ~driven & gng).sum())
                    for _ in range(1000)]
            row[f'recruited_gng_{r}Hz_ci95'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            boot = [int(((rates[rng.integers(0, n_trials, n_trials)].mean(0) >= 1.0) & ~driven).sum()) for _ in range(1000)]
            row[f'recruited_{r}Hz_ci95'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            per_trial[f'{name}|{r}'] = rates.astype(np.int16)
        rows.append(row)
        print({k2: row[k2] for k2 in ('set', 'n_cells', 'braun_class', 'braun_downstream_DNs',
                                      'connectome_downstream_DNs_ge5', 'recruited_50Hz', 'recruited_200Hz',
                                      'recruited_gng_200Hz', 'n_active_200Hz')}, flush=True)
    out = ROOT / 'results' / 'expression' / f'braun_recruitment_{variant}.json'
    out.write_text(json.dumps(rows, indent=1))
    np.savez_compressed(out.with_suffix('.npz'), dn_ids=dn_ids, gng=gng, **per_trial)


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['v783_alfix_all']))
