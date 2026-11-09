"""Build the Stage 1 results: reproduction tests, frequency response, figures, summary.json.

    .venv/bin/python stage1/report.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures  # noqa: E402
from analysis import compare_runs, first_spike_latency, hop_distance, input_drive, load_spikes, psth, rate_matrix  # noqa: E402
from neurons import EXAMPLE_SILENCED, MN9_L, MN9_R, REF, SUGAR_R, id2label, id2name  # noqa: E402
from run_reference import FIG1D_FREQS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results' / 'stage1' / 'runs' / 'v630'
SAVED = REF / 'results' / 'example'
ARCHIVE = ROOT / 'data' / 'shiu_archive' / 'results' / 'figure_1' / 'data'  # stage1/fetch_archive.py
OUT = ROOT / 'results' / 'stage1'
FIG = OUT / 'figures'
DT = 1e-4  # s, Brian2 default time step used by the reference model
ROUNDUP_2, G2N_1_2 = 720575940623211725, 720575940620874757  # sez_neurons.pickle names; CB0553 / CB0616


def expected_input_rate(r):
    """Rate of a Poisson-driven neuron in the reference model: a Poisson event arriving on the
    step the neuron spikes is erased by the reset, so the rate is r / (1 + r*dt)."""
    return r / (1 + r * DT)


def input_check(rates, r):
    """Measured mean rate of the stimulated GRNs vs. the exact expectation (t-test over trials)."""
    per_trial = rates.reindex(SUGAR_R).fillna(0).mean(0).to_numpy()
    t = stats.ttest_1samp(per_trial, expected_input_rate(r))
    return {'expected_Hz': expected_input_rate(r), 'measured_Hz': float(per_trial.mean()),
            'se_Hz': float(per_trial.std(ddof=1) / np.sqrt(len(per_trial))), 't': float(t.statistic), 'p': float(t.pvalue)}


def mn9(rates):
    out = {}
    for name, i in (('MN9_L', MN9_L), ('MN9_R', MN9_R)):
        r = rates.loc[i] if i in rates.index else pd.Series(np.zeros(rates.shape[1]))
        out[name] = {'mean_Hz': float(r.mean()), 'sd_Hz': float(r.std(ddof=0))}  # ddof=0 as in utils.get_rate
    return out


def compare(ref_path, new_path, label):
    ref, new = rate_matrix(load_spikes(ref_path)), rate_matrix(load_spikes(new_path))
    table, summary, t_null = compare_runs(ref, new)
    summary['MN9_L_z'] = float(table.loc[MN9_L, 'z']) if MN9_L in table.index else None
    summary['MN9_R_z'] = float(table.loc[MN9_R, 'z']) if MN9_R in table.index else None
    summary['label'] = label
    return table, summary, t_null


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {'dataset': 'FlyWire v630 (paper version)', 'n_trials': 30, 't_run_s': 1.0}
    names = id2name()
    labels = id2label()

    # --- frequency response (Fig. 1D protocol) and input check per frequency
    freqs = [f for f in FIG1D_FREQS if (RUNS / f'sugarR_{f}Hz.parquet').is_file()]
    sweep = {}
    for f in freqs:
        rates = rate_matrix(load_spikes(RUNS / f'sugarR_{f}Hz.parquet'))
        sweep[f] = {'input': input_check(rates, f), 'n_active': int((rates.sum(1) > 0).sum()), **mn9(rates)}
    summary['frequency_sweep'] = sweep
    if freqs:
        z = np.array([sweep[f]['input']['t'] for f in freqs])
        summary['input_check_across_frequencies'] = {
            'n': len(freqs), 'mean_t': float(z.mean()), 'sd_t': float(z.std(ddof=1)),
            'n_abs_t_gt_2': int(np.sum(np.abs(z) > 2)),
        }

    saved_pts = {}
    for f, p in ((100, SAVED / 'sugarR_100Hz.parquet'), (200, SAVED / 'sugarR.parquet')):
        saved_pts[f] = mn9(rate_matrix(load_spikes(p)))
    summary['saved_example_MN9'] = saved_pts

    # --- the published Fig. 1D runs (authors' archive): per-frequency exchangeability tests
    pub = {f: ARCHIVE / f'sugarR_{f}Hz.parquet' for f in freqs if (ARCHIVE / f'sugarR_{f}Hz.parquet').is_file()}
    vs_pub = {}
    for f, p in pub.items():
        table, s, t_null = compare(p, RUNS / f'sugarR_{f}Hz.parquet', f'sugarR {f} Hz: published vs re-run')
        s['MN9_published'] = mn9(rate_matrix(load_spikes(p)))
        vs_pub[f] = s
        if f in (100, 200):
            table.to_csv(OUT / f'compare_published_{f}Hz.csv')
            figures.reproduction_check(table, s, t_null, FIG / f'reproduction_{f}Hz.png',
                                       f'sugar GRNs at {f} Hz: published run (authors’ archive) vs. this re-run, 30 trials each')
    if vs_pub:
        p = np.array([s['perm_p'] for s in vs_pub.values()])
        zm = np.array([s['MN9_L_z'] for s in vs_pub.values() if s['MN9_L_z'] is not None])
        summary['vs_published_fig1d'] = {
            'n_frequencies': len(p), 'n_perm_p_lt_0.05': int(np.sum(p < 0.05)), 'ks_uniform_p': float(stats.kstest(p, 'uniform').pvalue),
            'max_fdr_hits': int(max(s['n_mw_q_lt_0.05'] for s in vs_pub.values())),
            'min_pearson_r': float(min(s['pearson_r_mean_rates'] for s in vs_pub.values())),
            'MN9_L_abs_z_max': float(np.abs(zm).max()), 'per_frequency': vs_pub,
        }

    if freqs:
        fa = np.array(freqs)
        curves = {
            'MN9 contra': (np.array([sweep[f]['MN9_L']['mean_Hz'] for f in freqs]), np.array([sweep[f]['MN9_L']['sd_Hz'] for f in freqs]), figures.ORANGE),
            'MN9 ipsi': (np.array([sweep[f]['MN9_R']['mean_Hz'] for f in freqs]), np.array([sweep[f]['MN9_R']['sd_Hz'] for f in freqs]), figures.BLUE),
        }
        pf = sorted(vs_pub)
        published = {
            'MN9 contra, published (mean ± SD)': (pf, [vs_pub[f]['MN9_published']['MN9_L']['mean_Hz'] for f in pf], [vs_pub[f]['MN9_published']['MN9_L']['sd_Hz'] for f in pf], figures.ORANGE),
            'MN9 ipsi, published (mean ± SD)': (pf, [vs_pub[f]['MN9_published']['MN9_R']['mean_Hz'] for f in pf], [vs_pub[f]['MN9_published']['MN9_R']['sd_Hz'] for f in pf], figures.BLUE),
        }
        figures.frequency_response(fa, curves, FIG / 'frequency_response.png', published)

    # --- the repo's shipped example outputs (note: the 'default' one was run at 200 Hz, not model.py's 150 Hz)
    tests = {}
    for f, saved in ((100, 'sugarR_100Hz.parquet'), (200, 'sugarR.parquet')):
        new = RUNS / f'sugarR_{f}Hz.parquet'
        if new.is_file():
            table, s, t_null = compare(SAVED / saved, new, f'sugarR {f} Hz: repo example vs re-run')
            s['input_saved'] = input_check(rate_matrix(load_spikes(SAVED / saved)), f)
            s['input_rerun'] = input_check(rate_matrix(load_spikes(new)), f)
            tests[f'sugarR_{f}Hz'] = s
    for i in EXAMPLE_SILENCED:
        new = RUNS / f'sugarR_100Hz-{i}.parquet'
        if new.is_file():
            _, s, _ = compare(SAVED / f'sugarR-{i}.parquet', new, f'sugarR 100 Hz, silence {i}')
            s['MN9_saved'] = mn9(rate_matrix(load_spikes(SAVED / f'sugarR-{i}.parquet')))
            s['MN9_rerun'] = mn9(rate_matrix(load_spikes(new)))
            tests[f'silence_{i}'] = s
    summary['reproduction_tests'] = tests

    # --- run-to-run variability: all pairs among saved + independent 200 Hz re-runs
    reps = [('repo_example', SAVED / 'sugarR.parquet'), ('published_fig1d', ARCHIVE / 'sugarR_200Hz.parquet'),
            ('rerun1', RUNS / 'sugarR_200Hz.parquet')]
    reps += [(f'rerun{k}', RUNS / f'sugarR_200Hz_rep{k}.parquet') for k in range(2, 5)]
    reps = [(n, p) for n, p in reps if p.is_file()]
    if len(reps) > 2:
        pairs = []
        for (na, pa), (nb, pb) in itertools.combinations(reps, 2):
            _, s, _ = compare(pa, pb, f'{na} vs {nb}')
            pairs.append({k: s[k] for k in ('label', 'perm_p', 'frac_abs_z_gt_1.96', 'pearson_r_mean_rates', 'MN9_L_z', 'MN9_R_z')})
        summary['replicates_200Hz'] = {
            'inputs': {n: input_check(rate_matrix(load_spikes(p)), 200) for n, p in reps},
            'MN9': {n: mn9(rate_matrix(load_spikes(p))) for n, p in reps},
            'pairs': pairs,
        }

    # --- single-trial raster + 30-trial rates at 200 Hz, with hop distance from the GRNs
    path = RUNS / 'sugarR_200Hz.parquet'
    if path.is_file():
        df = load_spikes(path)
        rates = rate_matrix(df).mean(1)
        lat = first_spike_latency(df)
        comp, con = REF / '2023_03_23_completeness_630_final.csv', REF / '2023_03_23_connectivity_630_final.parquet'
        hops = hop_distance(comp, con, SUGAR_R)
        hops_exc = hop_distance(comp, con, SUGAR_R, excitatory_only=True)
        downstream = rates.drop(index=[i for i in SUGAR_R if i in rates.index]).sort_values(ascending=False)
        top = downstream.index[:40]
        for must in (MN9_L, MN9_R):
            if must not in top:
                top = top.append(pd.Index([must]))
        responders = list(lat.loc[top].sort_values('latency_ms').index)
        # pathway story: a first-wave 2nd-order neuron, MN9's dominant excitatory input, and MN9 itself
        keys = [MN9_L, MN9_R, ROUNDUP_2, G2N_1_2]
        pst = psth(df, keys, bin_ms=20)
        figures.sugar_response(df[df.trial == 0], SUGAR_R, responders, labels, {MN9_L, MN9_R}, pst,
                               {k: labels.get(k, str(k)) for k in keys}, FIG / 'sugar_response_200Hz.png', 200)
        summary['MN9_input_drive_200Hz'] = {
            name: input_drive(con, i, rates).head(6).assign(label=lambda d: d.index.map(lambda j: labels.get(j, ''))).reset_index().to_dict('records')
            for name, i in (('MN9_L', MN9_L), ('MN9_R', MN9_R))
        }
        resp = pd.DataFrame({
            'name': [names.get(i, '') for i in downstream.index], 'label': [labels.get(i, '') for i in downstream.index], 'rate_Hz': downstream.values,
            'latency_ms': lat.reindex(downstream.index)['latency_ms'].values,
            'hops': hops.reindex(downstream.index).values, 'hops_excitatory': hops_exc.reindex(downstream.index).values,
        }, index=downstream.index)
        resp.to_csv(OUT / 'responders_200Hz.csv')
        summary['propagation_200Hz'] = {
            'n_active_downstream': int(len(downstream)),
            'active_by_hops': {str(int(k)) if np.isfinite(k) else 'unreachable': int(v) for k, v in resp['hops'].value_counts().sort_index().items()},
            'MN9_L': {'hops': float(hops[MN9_L]), 'hops_excitatory': float(hops_exc[MN9_L]), 'latency_ms': float(lat.loc[MN9_L, 'latency_ms'])},
            'MN9_R': {'hops': float(hops[MN9_R]), 'hops_excitatory': float(hops_exc[MN9_R]), 'latency_ms': float(lat.loc[MN9_R, 'latency_ms'])},
        }

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'frequency_sweep'}, indent=1))


if __name__ == '__main__':
    main()
