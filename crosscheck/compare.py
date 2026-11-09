"""Score the MLX port against the Brian2 reference model on the cross-check conditions (crosscheck/run.py).

Per condition: the whole-trial permutation test of stage1/analysis.compare_runs over every active neuron, the number of
active neurons per trial outside the stimulated set (the runaway state shows up here), and both MN9 cells.

    .venv/bin/python crosscheck/compare.py      # -> results/crosscheck/summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage1'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import compare_runs, load_spikes, rate_matrix  # noqa: E402
from neurons import ANNOTATIONS, MN9_L, MN9_R, to_v783  # noqa: E402
from run import CONDITIONS  # noqa: E402

RES = ROOT / 'results' / 'crosscheck'
MN9 = [MN9_L, to_v783([MN9_R])[0]]


def active_outside(df, stimulated, n_run):
    """Neurons with at least one spike in a trial, not counting the stimulated ones; one value per trial."""
    act = df[~df.flywire_id.isin(stimulated)].groupby('trial').flywire_id.nunique()
    return act.reindex(range(n_run), fill_value=0).to_numpy()


def main():
    ann = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False, usecols=['root_id', 'side']).set_index('root_id').side
    rows = []
    for k, (name, wiring, _, rate, n_run, _) in enumerate(CONDITIONS):
        runs = {e: RES / e / wiring / f'{name}.parquet' for e in ('brian2', 'mlx')}
        if not all(p.is_file() for p in runs.values()):
            print(f'missing {wiring}/{name}; run crosscheck/run.py for both engines first')
            continue
        stim = json.loads(runs['brian2'].with_suffix('.json').read_text())['stimulated']
        spikes = {e: load_spikes(p) for e, p in runs.items()}
        rates = {e: rate_matrix(df, n_run=n_run) for e, df in spikes.items()}
        _, s, _ = compare_runs(rates['brian2'], rates['mlx'], n_perm=2000, seed=k)
        row = {'condition': name, 'wiring': wiring, 'rate_Hz': rate, 'trials': n_run, 'n_stimulated': len(stim),
               'perm_p': s['perm_p'], 'fdr_hits': s['n_mw_q_lt_0.05'], 'r_mean_rates': s['pearson_r_mean_rates'],
               'neurons_compared': s['n_neurons_compared']}
        for e, df in spikes.items():
            a = active_outside(df, stim, n_run)
            row[f'active_{e}'] = {'median': float(np.median(a)), 'min': int(a.min()), 'max': int(a.max())}
            row[f'MN9_{e}_Hz'] = {f"{i} ({ann.get(i, '?')})": float(rates[e].loc[i].mean()) if i in rates[e].index
                                  else 0.0 for i in MN9}
        rows.append(row)
        print(json.dumps(row), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / 'summary.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
