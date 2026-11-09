"""The fly dictionary: which descending neurons each stimulus drives, translated into behaviour where experiments allow.

Behaviour labels only for DN types whose activation phenotype is published (sources in KNOWN). Everything else is
reported by FlyWire type name. Output: results/expression/dictionary_<variant>.json and a markdown table.

    .venv-mlx/bin/python expression/dictionary.py v783_alfix_all
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load  # noqa: E402
from wiring import annotations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# FlyWire type (or root IDs) -> (behaviour, source)
KNOWN = {
    'DNp01': ('escape take-off (giant fibre)', 'von Reyn et al. 2014'),
    'DNp09': ('forward walking', 'Bidaye et al. 2020; Sapkal et al. 2024; Braun et al. 2024'),
    'DNg100': ('forward walking (BDN2)', 'Sapkal et al. 2024'),
    'DNg97': ('forward walking (oDN1)', 'Sapkal et al. 2024'),
    'MDN': ('backward walking', 'Bidaye et al. 2014'),
    'DNp42': ('backing up', 'Braun et al. 2024'),
    'DNg62': ('antennal grooming (aDN1)', 'Hampel et al. 2015; Braun et al. 2024 (their aDN1 IDs are DNg62 in v3.2.0)'),
    'DNge078': ('antennal grooming (aDN2)', 'Hampel et al. 2015; Braun et al. 2024'),
    'DNa01': ('in-place turning', 'Braun et al. 2024'),
    'DNa02': ('turning (graded with rotation)', 'Braun et al. 2024; Rayshubskiy et al. 2025'),
    'DNb02': ('turning', 'Braun et al. 2024'),
    'DNg11': ('foreleg rubbing', 'Braun et al. 2024'),
    'DNg14': ('abdominal dipping', 'Braun et al. 2024'),
    'oviDNa_a': ('oviposition / abdomen curling', 'Braun et al. 2024; Wang et al. 2020'),
    'oviDNa_b': ('oviposition', 'Wang et al. 2020'),
    'oviDNb': ('oviposition', 'Wang et al. 2020'),
}


def main(variant='v783_alfix_all', dose=100.0, top=5, min_total_hz=20.0):
    ann = annotations().set_index('root_id')
    lib, doses, dn, nact, dn_ids = load(variant)
    di = list(doses).index(dose)
    types = ann.reindex(dn_ids).cell_type.astype(str).values
    sides = ann.reindex(dn_ids).side.astype(str).values
    mean = dn[:, di].mean(1)
    rows = []
    for s, m in zip(lib, mean):
        tot = m.sum()
        order = np.argsort(-m)[:top]
        labelled = [i for i in np.flatnonzero(m >= 1) if types[i] in KNOWN]
        lab_share = float(m[labelled].sum() / tot) if tot > 0 else 0.0
        beh = {}
        for i in labelled:
            beh[KNOWN[types[i]][0]] = beh.get(KNOWN[types[i]][0], 0) + float(m[i])
        rows.append({'stimulus': s['sid'], 'cell_class': s['cell_class'], 'n_DN_responding': int((m >= 1).sum()),
                     'total_DN_rate_Hz': float(tot),
                     'top_DNs': [f'{types[i]}({sides[i][0]}) {m[i]:.0f} Hz' for i in order if m[i] >= 1],
                     'known_behaviour_share': lab_share,
                     'behaviours': dict(sorted(beh.items(), key=lambda kv: -kv[1]))})
    out = ROOT / 'results' / 'expression' / f'dictionary_{variant}.json'
    out.write_text(json.dumps({'dose_Hz': dose, 'known': {k: list(v) for k, v in KNOWN.items()}, 'rows': rows}, indent=1))
    # a short human-readable table of the stimuli whose output reaches a DN with a known behaviour
    # rank only stimuli with a substantial DN response, so a single weakly driven DN cannot top the table
    hits = sorted([r for r in rows if r['behaviours'] and r['total_DN_rate_Hz'] >= min_total_hz],
                  key=lambda r: -r['known_behaviour_share'])
    md = ['| stimulus | DNs responding | top DNs | behaviours read out (share of DN output) |', '|---|---|---|---|']
    for r in hits[:40]:
        b = ', '.join(f'{k} {v / max(r["total_DN_rate_Hz"], 1e-9):.0%}' for k, v in list(r['behaviours'].items())[:3])
        md.append(f"| {r['stimulus']} | {r['n_DN_responding']} | {', '.join(r['top_DNs'][:3])} | {b} |")
    (ROOT / 'results' / 'expression' / f'dictionary_{variant}.md').write_text('\n'.join(md) + '\n')
    print(f'{len(rows)} stimuli; {len(hits)} reach a DN with a known behaviour at {dose:.0f} Hz; '
          f'median share of output on known DNs {np.median([r["known_behaviour_share"] for r in rows]):.2f}')
    print('\n'.join(md[:15]))


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['v783_alfix_all']))
