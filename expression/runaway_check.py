"""Does each wiring variant still run away? Inputs known to ignite the v783 runaway state (reported by xiangdoz,
in issue #11 of the reference repo, and others): one glomerulus, all ORNs, right-side LB3 sugar at 150 Hz, thermo- and
hygrosensory neurons, plus Shiu et al.'s 21 sugar neurons as the calibration check. 3 trials each. MN9 contra and ipsi
are relative to the stimulated side (FlyWire's side annotation). Rows for the variants given replace stored rows.

    .venv-mlx/bin/python expression/runaway_check.py v783 v783_alfix_all v783_alfix_ach
    # -> results/expression/runaway_check.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage1'))
import mlxrun  # noqa: E402
from neurons import MN9_L, MN9_R, SUGAR_R, to_v783  # noqa: E402
from wiring import annotations  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'results' / 'expression' / 'runaway_check.json'
# the two MN9 cells by FlyWire side (the reference repo's "left" MN9 is on FlyWire's right)
MN9_BY_SIDE = {'right': MN9_L, 'left': to_v783([MN9_R])[0]}


def tests():
    """name -> (stimulated root IDs, rate in Hz, FlyWire side of the stimulus)."""
    ann = annotations()
    sens = ann[ann.super_class == 'sensory']

    def ids(mask):
        return sens[mask].root_id.astype(np.int64).tolist()

    return {
        'ORN_VL2p left 20 Hz': (ids((sens.cell_type == 'ORN_VL2p') & (sens.side == 'left')), 20, 'left'),
        'all ORNs left 30 Hz': (ids((sens.cell_class == 'olfactory') & (sens.side == 'left')), 30, 'left'),
        'LB3b+c right 150 Hz': (ids(sens.cell_type.isin(['LB3b', 'LB3c']) & (sens.side == 'right')), 150, 'right'),
        'thermosensory left 150 Hz': (ids((sens.cell_class == 'thermosensory') & (sens.side == 'left')), 150, 'left'),
        'hygrosensory left 150 Hz': (ids((sens.cell_class == 'hygrosensory') & (sens.side == 'left')), 150, 'left'),
        # the reference repo's "right" sugar GRNs are on FlyWire's left
        'Shiu 21 sugar 100 Hz': (to_v783(SUGAR_R), 100, 'left'),
    }


def main(variants):
    rows = []
    cases = tests()
    for variant in variants:
        ix = mlxrun.index(variant)
        for name, (nids, rate, side) in cases.items():
            t0 = time.time()
            idx = mlxrun.to_idx(nids, variant)
            c = mlxrun.run_counts(variant, idx, rate, n_trials=3, t_s=1.0, seed=7)
            contra, ipsi = MN9_BY_SIDE['right' if side == 'left' else 'left'], MN9_BY_SIDE[side]
            row = {'variant': variant, 'test': name, 'n_driven': len(idx), 'active_per_trial': (c > 0).sum(1).tolist(),
                   'MN9_contra_Hz': float(c[:, ix[contra]].mean()), 'MN9_ipsi_Hz': float(c[:, ix[ipsi]].mean()),
                   'wall_s': round(time.time() - t0, 1)}
            rows.append(row)
            print(row, flush=True)
    old = json.loads(OUT.read_text()) if OUT.is_file() else []
    OUT.write_text(json.dumps([r for r in old if r['variant'] not in variants] + rows, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
