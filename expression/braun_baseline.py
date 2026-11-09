"""How much of the model's DN recruitment is already in the wiring? Spearman correlation of the spiking model's
recruited-DN counts (expression/braun.py) with two structural baselines: the number of DN partners at >= 5 synapses,
and the same count set to 0 for inhibitory DN sets (the model's sign rule). Also against Braun et al.'s own counts.

    .venv-mlx/bin/python expression/braun_baseline.py v783_alfix_all
"""
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import braun  # noqa: E402
from wiring import annotations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(variant='v783_alfix_all'):
    rows = json.loads((ROOT / 'results' / 'expression' / f'braun_recruitment_{variant}.json').read_text())
    ann = annotations()
    con = pd.read_parquet(ROOT / 'data' / 'wiring' / variant / 'Connectivity_783.parquet',
                          columns=['Presynaptic_ID', 'Excitatory'])
    sign = con.groupby('Presynaptic_ID').Excitatory.first()
    for (name, sel, cls, bn), r in zip(braun.SETS, rows):
        s = sign.reindex(braun.select(ann, sel)).dropna()
        r['set_sign'] = 'excitatory' if (s > 0).mean() > 0.5 else 'inhibitory'
        r['signed_count_baseline'] = r['connectome_downstream_DNs_ge5'] if r['set_sign'] == 'excitatory' else 0
    out = {}
    for lab, key in [('all >=5-synapse DN partners', 'connectome_downstream_DNs_ge5'),
                     ('signed count', 'signed_count_baseline'), ('Braun et al. counts', 'braun_downstream_DNs')]:
        x = [r[key] for r in rows]
        out[lab] = {f'{d}Hz': float(spearmanr(x, [r[f'recruited_{d}Hz'] for r in rows]).statistic) for d in braun.RATES}
        out[lab].update({f'gng_{d}Hz': float(spearmanr(x, [r[f'recruited_gng_{d}Hz'] for r in rows]).statistic)
                         for d in braun.RATES})
    res = {'rows': rows, 'spearman_LIF_vs_baselines': out,
           'braun_vs_wiring_counts_rho': float(spearmanr([r['braun_downstream_DNs'] for r in rows],
                                                         [r['connectome_downstream_DNs_ge5'] for r in rows]).statistic)}
    (ROOT / 'results' / 'expression' / f'braun_baseline_{variant}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['v783_alfix_all']))
