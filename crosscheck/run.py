"""Run the same conditions in both simulators: the authors' Brian2 model and the MLX port.

The MLX port passed the acceptance test on the sugar protocol only. These conditions check it, against the unchanged
reference model, where parts 2 and 3 lean on it most (FlyWire v783):

  shiu21_100Hz          Shiu et al.'s 21 sugar GRNs at 100 Hz on the corrected wiring, 30 trials
  LB3bc_right_150Hz     all LB3b + LB3c taste neurons on FlyWire's right side at 150 Hz, 10 trials, on the published
                        wiring (where this input throws the brain into the runaway state) and on the corrected wiring

Writes spike files in the reference parquet format plus a provenance JSON per run to
results/crosscheck/<engine>/<wiring>/. Then crosscheck/compare.py scores the two engines against each other.

    .venv/bin/python crosscheck/run.py --engine brian2 --n-proc 6
    .venv-mlx/bin/python crosscheck/run.py --engine mlx
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'external' / 'Drosophila_brain_model'
MLX = ROOT / 'external' / 'drosophila-brain-mlx'
sys.path.insert(0, str(ROOT / 'stage1'))
from neurons import ANNOTATIONS, SUGAR_R, to_v783  # noqa: E402

WIRING = {  # Brian2 tables and MLX pack of each wiring
    'v783': (REF / 'Completeness_783.csv', REF / 'Connectivity_783.parquet', MLX / 'data' / 'pack' / 'v783'),
    'v783_alfix_all': (ROOT / 'data' / 'wiring' / 'v783_alfix_all' / 'Completeness_783.csv',
                       ROOT / 'data' / 'wiring' / 'v783_alfix_all' / 'Connectivity_783.parquet',
                       MLX / 'data' / 'pack' / 'v783_alfix_all'),
}


def lb3bc_right():
    ann = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False, usecols=['root_id', 'super_class', 'cell_type', 'side'])
    m = (ann.super_class == 'sensory') & ann.cell_type.isin(['LB3b', 'LB3c']) & (ann.side == 'right')
    return sorted(ann[m].root_id.astype('int64').tolist())


# (name, wiring, stimulated neurons, rate in Hz, trials, MLX seed); MLX trial n uses seed + n
CONDITIONS = [
    ('shiu21_100Hz', 'v783_alfix_all', lambda: to_v783(SUGAR_R), 100, 30, 910_000),
    ('LB3bc_right_150Hz', 'v783', lb3bc_right, 150, 10, 920_000),
    ('LB3bc_right_150Hz', 'v783_alfix_all', lb3bc_right, 150, 10, 930_000),
]


def commit(path):
    return subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine', choices=['brian2', 'mlx'], required=True)
    ap.add_argument('--n-proc', type=int, default=6, help='Brian2 only: parallel trials, ~2.2 GB RAM each')
    args = ap.parse_args()

    if args.engine == 'brian2':
        import run_reference as R  # adds the reference model to sys.path
        from brian2 import Hz
    else:
        from lif import core
        from lif.experiment import default_params as mlx_params, run_exp as mlx_run

    for name, wiring, ids_fn, rate, n_run, seed in CONDITIONS:
        out = ROOT / 'results' / 'crosscheck' / args.engine / wiring
        out.mkdir(parents=True, exist_ok=True)
        if (out / f'{name}.parquet').is_file():
            print(f'>>> skip {args.engine}/{wiring}/{name} (exists)', flush=True)
            continue
        comp, con, pack = WIRING[wiring]
        ids = ids_fn()
        t0 = time.time()
        if args.engine == 'brian2':
            p = deepcopy(R.default_params)
            p['r_poi'], p['n_run'] = rate * Hz, n_run
            R.run_exp(exp_name=name, neu_exc=ids, neu_slnc=[], path_res=out, path_comp=comp, path_con=con, params=p,
                      n_proc=args.n_proc)
            meta = R.provenance(wiring, name, rate, len(ids), [], args.n_proc, time.time() - t0, p)
        else:
            p = dict(mlx_params)
            p['r_poi'], p['n_run'] = float(rate), n_run
            mlx_run(name, ids, out, params=p, pack=core.load_pack(pack), seed=seed, engine='fused', edge_split=1)
            meta = {'exp_name': name, 'wiring': wiring, 'r_poi_Hz': rate, 'n_run': n_run, 'n_excited': len(ids),
                    'seed': seed, 'engine': 'fused', 'pack': str(pack.relative_to(ROOT)), 'mlx_port_commit': commit(MLX),
                    'walltime_s': round(time.time() - t0, 1), 'machine': platform.platform(),
                    'python': platform.python_version(), 'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
        meta['stimulated'] = ids
        (out / f'{name}.json').write_text(json.dumps(meta, indent=1))
        print(f'    done {args.engine}/{wiring}/{name}: {len(ids)} neurons, {n_run} trials, {meta["walltime_s"]} s',
              flush=True)


if __name__ == '__main__':
    main()
