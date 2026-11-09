"""MLX port vs the reference Brian2 model on FlyWire v783: same 21 GRNs, same rates, trial-permutation test."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage1'))
from scipy import stats  # noqa: E402

from analysis import compare_runs, load_spikes, rate_matrix  # noqa: E402

rows = []
for f in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
    ref = rate_matrix(load_spikes(ROOT / f'results/stage1/runs/v783/sugarR_{f}Hz.parquet'))
    new = rate_matrix(load_spikes(ROOT / f'results/stage1/runs/mlx_v783/sugarR_{f}Hz.parquet'))
    _, s, _ = compare_runs(ref, new, n_perm=2000, seed=f)
    rows.append({'rate_Hz': f, 'perm_p': s['perm_p'], 'fdr_hits': s['n_mw_q_lt_0.05'], 'r': s['pearson_r_mean_rates']})
    print(rows[-1], flush=True)
ks = stats.kstest([r['perm_p'] for r in rows], 'uniform').pvalue
print('KS uniformity p =', round(ks, 3), '| p<0.05:', sum(r['perm_p'] < 0.05 for r in rows), 'of', len(rows))
(ROOT / 'results/expression/validate_mlx_v783.json').write_text(json.dumps({'rows': rows, 'ks_p': ks}, indent=1))
