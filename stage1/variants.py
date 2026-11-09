"""Dataset version and taste-neuron typing: does sugar -> MN9 survive the v630 -> v783 dataset change and the 2026
GRN retyping?

Inputs: results/stage1/runs/v630 (Fig. 1D protocol) and results/stage1/runs/v783 (groups 'archived_freqs'
and 'grn_classes' of run_reference.py). Effects are judged against the model's run-to-run variability.

    .venv/bin/python stage1/variants.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures  # noqa: E402
from analysis import compare_runs, load_spikes, rate_matrix  # noqa: E402
from neurons import MN9_L, MN9_R, REMAP_630_783, grn_sets_783  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R630, R783 = ROOT / 'results' / 'stage1' / 'runs' / 'v630', ROOT / 'results' / 'stage1' / 'runs' / 'v783'
OUT = ROOT / 'results' / 'stage1'
MN9_CONTRA, MN9_IPSI_783 = MN9_L, REMAP_630_783[MN9_R]


def mn9_stats(rates, ipsi_id):
    out = {}
    for name, i in (('contra', MN9_CONTRA), ('ipsi', ipsi_id)):
        r = rates.loc[i] if i in rates.index else 0 * rates.iloc[0]
        out[name] = {'mean_Hz': float(r.mean()), 'sd_Hz': float(r.std(ddof=1)), 'sem_Hz': float(r.std(ddof=1) / np.sqrt(len(r)))}
    return out


def main():
    summary = {'version_comparison': [], 'grn_sets': []}
    # --- v630 vs v783, same 21 GRNs (one remapped ID), same parameters
    for f in range(20, 201, 20):
        p630, p783 = R630 / f'sugarR_{f}Hz.parquet', R783 / f'sugarR_{f}Hz.parquet'
        if not (p630.is_file() and p783.is_file()):
            continue
        a, b = rate_matrix(load_spikes(p630)), rate_matrix(load_spikes(p783))
        shared = a.index.intersection(b.index)  # unchanged root IDs; edited neurons get new IDs
        changed_active = int(len(a.index.difference(b.index)) + len(b.index.difference(a.index)))
        _, s, _ = compare_runs(a.loc[shared], b.loc[shared], n_perm=2000, seed=f)
        m630, m783 = mn9_stats(a, MN9_R), mn9_stats(b, MN9_IPSI_783)
        d = m783['contra']['mean_Hz'] - m630['contra']['mean_Hz']
        summary['version_comparison'].append({
            'freq': f, 'MN9_contra_v630': m630['contra'], 'MN9_contra_v783': m783['contra'],
            'MN9_ipsi_v630': m630['ipsi'], 'MN9_ipsi_v783': m783['ipsi'],
            'MN9_contra_diff_Hz': d, 'MN9_contra_diff_z': d / np.hypot(m630['contra']['sem_Hz'], m783['contra']['sem_Hz']) if d else 0.0,
            'n_active_v630': int((a.sum(1) > 0).sum()), 'n_active_v783': int((b.sum(1) > 0).sum()),
            'n_active_with_changed_ids': changed_active, 'shared_id_perm_p': s['perm_p'],
            'shared_id_pearson_r': s['pearson_r_mean_rates'], 'shared_id_fdr_hits': s['n_mw_q_lt_0.05'],
        })

    # --- annotation-defined GRN sets on v783
    sets = grn_sets_783()
    for name, ids in sets.items():
        for f in (50, 100, 200):
            p = R783 / (f'sugarR_{f}Hz.parquet' if name == 'shiu21' and f in (100, 200) else f'{name}_{f}Hz.parquet')
            if not p.is_file():
                continue
            r = rate_matrix(load_spikes(p))
            summary['grn_sets'].append({'set': name, 'n_grns': len(ids), 'freq': f, 'n_active': int((r.sum(1) > 0).sum()),
                                        **{f'MN9_{k}': v for k, v in mn9_stats(r, MN9_IPSI_783).items()}})

    (OUT / 'variants.json').write_text(json.dumps(summary, indent=1))
    if summary['version_comparison']:
        figures.version_comparison(summary['version_comparison'], OUT / 'figures' / 'v630_vs_v783.png')
    if summary['grn_sets']:
        figures.grn_set_dotplot(summary['grn_sets'], OUT / 'figures' / 'grn_classes_v783.png')
    for row in summary['version_comparison']:
        print(f"{row['freq']:3d} Hz  MN9 contra v630 {row['MN9_contra_v630']['mean_Hz']:6.2f}  v783 {row['MN9_contra_v783']['mean_Hz']:6.2f}"
              f"  diff {row['MN9_contra_diff_Hz']:+6.2f} (z {row['MN9_contra_diff_z']:+.1f})  shared-ID p {row['shared_id_perm_p']:.3f}"
              f"  active {row['n_active_v630']}/{row['n_active_v783']} ({row['n_active_with_changed_ids']} with changed IDs)")
    for row in summary['grn_sets']:
        print(f"{row['set']:24s} n={row['n_grns']:2d} {row['freq']:3d} Hz: MN9 contra {row['MN9_contra']['mean_Hz']:6.2f} ± {row['MN9_contra']['sd_Hz']:5.2f}"
              f"  ipsi {row['MN9_ipsi']['mean_Hz']:6.2f}  active {row['n_active']}")


if __name__ == '__main__':
    main()
