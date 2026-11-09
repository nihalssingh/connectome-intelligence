"""Acceptance test for any implementation of the Shiu et al. model: is it the same model as published?

Give it a directory of spike files in the reference format (one row per spike: t [s], trial,
flywire_id) named like the authors' Fig. 1D runs, sugarR_{rate}Hz.parquet (21 sugar GRNs, FlyWire
v630, 30 trials x 1 s). Each rate is compared with the authors' archived run by a trial-permutation
exchangeability test over all active neurons (stage1/analysis.compare_runs).

PASS requires, across the rates present:
  - the permutation p-values are consistent with uniform (KS p > 0.01),
  - no neuron differs after Benjamini-Hochberg FDR at any rate,
  - contralateral MN9 within |z| < 3.5 at every rate (~1% family-wise false alarms over 20 rates),
  - per-neuron mean-rate correlation r > 0.99 at every rate.
Calibration and power: run stage1/test_analysis.py directly; pytest runs only its pass/fail guards.

    .venv/bin/python stage1/acceptance.py results/stage1/runs/v630
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import compare_runs, load_spikes, rate_matrix  # noqa: E402
from neurons import MN9_L  # noqa: E402

ARCHIVE = Path(__file__).resolve().parents[1] / 'data' / 'shiu_archive' / 'results' / 'figure_1' / 'data'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('spike_dir', type=Path)
    ap.add_argument('--pattern', default='sugarR_{f}Hz.parquet', help='candidate file name pattern')
    ap.add_argument('--json', type=Path, help='write the per-rate table here')
    args = ap.parse_args()

    rows = []
    for f in range(10, 201, 10):
        cand, ref = args.spike_dir / args.pattern.format(f=f), ARCHIVE / f'sugarR_{f}Hz.parquet'
        if not (cand.is_file() and ref.is_file()):
            continue
        table, s, _ = compare_runs(rate_matrix(load_spikes(ref)), rate_matrix(load_spikes(cand)), n_perm=2000, seed=f)
        z = float(table.loc[MN9_L, 'z']) if MN9_L in table.index else 0.0
        rows.append({'rate_Hz': f, 'perm_p': s['perm_p'], 'fdr_hits': s['n_mw_q_lt_0.05'],
                     'pearson_r': s['pearson_r_mean_rates'], 'MN9_contra_z': z})
        print(f"{f:4d} Hz  perm p {s['perm_p']:.3f}  FDR hits {s['n_mw_q_lt_0.05']}  r {s['pearson_r_mean_rates']:.4f}  MN9 z {z:+.2f}", flush=True)
    if not rows:
        sys.exit(f'no candidate files matching {args.pattern} in {args.spike_dir} (and stage1/fetch_archive.py run?)')

    p = np.array([r['perm_p'] for r in rows])
    ks_p = float(stats.kstest(p, 'uniform').pvalue) if len(p) > 1 else float(p[0])
    print(f"KS uniformity p = {ks_p:.2f}; {int((p < 0.05).sum())} of {len(p)} rates with perm p < 0.05")
    checks = {
        'p-values uniform (KS p > 0.01)': ks_p > 0.01,
        'no FDR hits at any rate': all(r['fdr_hits'] == 0 for r in rows),
        'MN9 contra |z| < 3.5 at every rate': all(abs(r['MN9_contra_z']) < 3.5 for r in rows),
        'per-neuron rate r > 0.99 at every rate': all(r['pearson_r'] > 0.99 for r in rows),
    }
    for k, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {k}")
    verdict = all(checks.values())
    print(f"\n{'PASS' if verdict else 'FAIL'}: {len(rows)} rates compared with the published runs")
    if args.json:
        args.json.write_text(json.dumps({'rows': rows, 'ks_p': ks_p, 'checks': checks, 'pass': verdict}, indent=1))
    sys.exit(0 if verdict else 1)


if __name__ == '__main__':
    main()
