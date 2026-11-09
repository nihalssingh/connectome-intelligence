"""Is our re-run 'within the original model's variability'? Calibrate against the authors' own replicates.

The Edmond archive holds four independent sugar-only sweeps of the same 21 GRNs (Fig. 1D, and the
second-stimulus-at-0-Hz conditions of Figs. 3a, 3b and 4e). At each shared frequency we run every
pairwise trial-permutation test: authors vs authors gives the empirical null for the model's
run-to-run variability; our re-run vs each author run is judged against it.

    .venv/bin/python stage1/replicates.py      (after stage1/fetch_archive.py and the v630 sweep)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures  # noqa: E402
from analysis import compare_runs, load_spikes, rate_matrix  # noqa: E402
from neurons import BITTER_R, IR94E_R, MN9_L, MN9_R, WATER_R  # noqa: E402
from report import input_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / 'data' / 'shiu_archive' / 'results'
RUNS = ROOT / 'results' / 'stage1' / 'runs' / 'v630'
OUT = ROOT / 'results' / 'stage1'
FREQS = list(range(20, 201, 20))


def author_runs(f):
    return {
        'Fig1D': ARCH / 'figure_1' / 'data' / f'sugarR_{f}Hz.parquet',
        'Fig3a': ARCH / 'figure_3' / 'data' / f'Sugar_Bitter_{f}_Hz_0_Hz.parquet',
        'Fig3b': ARCH / 'figure_3' / 'data' / f'Sugar_Ir94e_{f}_Hz_0_Hz.parquet',
        'Fig4e': ARCH / 'figure_4' / 'data' / f'Sugar_Water_{f}_Hz_0_Hz.parquet',
    }


def our_runs(f):
    runs = {'ours': RUNS / f'sugarR_{f}Hz.parquet'}
    if f == 200:
        runs.update({f'ours_rep{k}': RUNS / f'sugarR_200Hz_rep{k}.parquet' for k in range(2, 5)})
    return {k: p for k, p in runs.items() if p.is_file()}


def main():
    rows, mn9_rows, inputs = [], [], []
    stray = {'Fig3a': 0, 'Fig3b': 0, 'Fig4e': 0}  # spikes of the 0-Hz second-stimulus neurons
    for f in FREQS:
        runs = {**author_runs(f), **our_runs(f)}
        rates = {k: rate_matrix(load_spikes(p)) for k, p in runs.items()}
        for k, ids in (('Fig3a', BITTER_R), ('Fig3b', IR94E_R), ('Fig4e', WATER_R)):
            stray[k] += int(rates[k].reindex(ids).fillna(0).to_numpy().sum())  # spikes/s summed over trials
        for k, r in rates.items():
            mn9 = r.loc[MN9_L] if MN9_L in r.index else 0 * r.iloc[0]
            mn9_rows.append({'freq': f, 'run': k, 'MN9_L_mean': float(mn9.mean()), 'MN9_L_sem': float(mn9.std(ddof=1) / np.sqrt(len(mn9))),
                             'MN9_R_mean': float(r.loc[MN9_R].mean()) if MN9_R in r.index else 0.0})
            inputs.append({'freq': f, 'run': k, **input_check(r, f)})
        for a, b in itertools.combinations(rates, 2):
            kind = 'authors-authors' if not (a.startswith('ours') or b.startswith('ours')) else (
                'ours-ours' if a.startswith('ours') and b.startswith('ours') else 'ours-authors')
            _, s, _ = compare_runs(rates[a], rates[b], n_perm=2000, seed=f)
            rows.append({'freq': f, 'a': a, 'b': b, 'kind': kind, 'perm_p': s['perm_p'],
                         'frac_abs_z': s['frac_abs_z_gt_1.96'], 'r': s['pearson_r_mean_rates'], 'fdr_hits': s['n_mw_q_lt_0.05']})
        print(f'{f} Hz done ({len(runs)} runs)', flush=True)

    summary = {'pairs': rows, 'MN9': mn9_rows, 'inputs': inputs, 'by_kind': {},
               'second_stimulus_neuron_spikes_all_freqs': stray}
    for kind in ('authors-authors', 'ours-authors', 'ours-ours'):
        p = np.array([r['perm_p'] for r in rows if r['kind'] == kind])
        if len(p):
            summary['by_kind'][kind] = {
                'n_pairs': int(len(p)), 'frac_p_lt_0.05': float(np.mean(p < 0.05)), 'median_p': float(np.median(p)),
                'ks_uniform_p': float(stats.kstest(p, 'uniform').pvalue),
                'max_fdr_hits': int(max(r['fdr_hits'] for r in rows if r['kind'] == kind)),
            }
    pa = [r['perm_p'] for r in rows if r['kind'] == 'authors-authors']
    po = [r['perm_p'] for r in rows if r['kind'] == 'ours-authors']
    summary['ours_vs_authors_pvalue_distribution_same_as_authors_vs_authors'] = float(stats.mannwhitneyu(pa, po).pvalue)
    # each authors' sweep enters 3 of the 6 authors-authors pairs at a rate, so those 60 p-values are not independent:
    # test one averaged p-value per rate against 0.5 (the mean of a uniform p-value), and split low and high rates
    aa_rate = np.array([np.mean([r['perm_p'] for r in rows if r['kind'] == 'authors-authors' and r['freq'] == f])
                        for f in FREQS])
    def band(r):
        return '100_200Hz' if r['freq'] >= 100 else '20_80Hz'
    summary['authors_authors_rate_level'] = {
        'per_rate_mean_p': aa_rate.tolist(),
        'wilcoxon_vs_0.5_p': float(stats.wilcoxon(aa_rate - 0.5).pvalue),
        'median_p_by_band': {f'{b}_{k}': float(np.median([r['perm_p'] for r in rows if r['kind'] == k and band(r) == b]))
                             for b in ('20_80Hz', '100_200Hz') for k in ('authors-authors', 'ours-authors')},
    }

    # input rates: are the authors' sweeps unbiased, and ours?
    for who in ('authors', 'ours'):
        t = np.array([i['t'] for i in inputs if i['run'].startswith('ours') == (who == 'ours')])
        summary['by_kind'][f'input_t_{who}'] = {'n': int(len(t)), 'mean_t': float(t.mean()), 'se_of_mean_under_null': float(1 / np.sqrt(len(t)))}

    # MN9 envelope: our mean vs the authors' 4-replicate spread at each frequency
    env = []
    for f in FREQS:
        a = np.array([m['MN9_L_mean'] for m in mn9_rows if m['freq'] == f and not m['run'].startswith('ours')])
        o = [m['MN9_L_mean'] for m in mn9_rows if m['freq'] == f and m['run'] == 'ours']
        if o:
            env.append({'freq': f, 'authors_min': float(a.min()), 'authors_max': float(a.max()), 'authors_mean': float(a.mean()),
                        'authors_between_run_sd': float(a.std(ddof=1)), 'ours': float(o[0]),
                        'z_vs_authors': float((o[0] - a.mean()) / (a.std(ddof=1) * np.sqrt(1 + 1 / len(a))))})
    summary['MN9_envelope'] = env
    (OUT / 'replicates.json').write_text(json.dumps(summary, indent=1))
    figures.replicate_calibration(rows, mn9_rows, OUT / 'figures' / 'replicate_calibration.png')
    print(json.dumps(summary['by_kind'], indent=1))
    print('MN9 envelope:', json.dumps(env, indent=0))


if __name__ == '__main__':
    main()
