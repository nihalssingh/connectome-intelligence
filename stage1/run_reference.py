"""Run the Shiu et al. reference model, unchanged, for the Stage 1 experiments.

The model is imported from the pristine clone in external/Drosophila_brain_model;
this script only picks stimuli/parameters and writes a provenance sidecar per run.

    .venv/bin/python stage1/run_reference.py --dataset 630 --group fig1d --n-proc 6
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'external' / 'Drosophila_brain_model'
sys.path.insert(0, str(REF))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import brian2  # noqa: E402
import numpy  # noqa: E402
from brian2 import Hz  # noqa: E402
from model import default_params, run_exp  # noqa: E402

from neurons import EXAMPLE_SILENCED, SUGAR_R, grn_sets_783, to_v783  # noqa: E402

DATASETS = {
    '630': (REF / '2023_03_23_completeness_630_final.csv', REF / '2023_03_23_connectivity_630_final.parquet'),
    '783': (REF / 'Completeness_783.csv', REF / 'Connectivity_783.parquet'),
}

FIG1D_FREQS = list(range(10, 201, 10))  # figures.ipynb, Fig. 1D


def groups(dataset='630'):
    """Named experiment groups -> ordered list of (exp_name, r_poi_Hz, neu_exc, neu_slnc)."""
    # 200 and 100 Hz first: they are directly comparable to the repo's saved example outputs
    sweep_order = [200, 100] + [f for f in FIG1D_FREQS if f not in (100, 200)]
    sugar = to_v783(SUGAR_R) if dataset == '783' else SUGAR_R
    g = {
        'fig1d': [(f'sugarR_{f}Hz', f, sugar, []) for f in sweep_order],
        # the frequencies of the authors' four archived replicate sweeps
        'archived_freqs': [(f'sugarR_{f}Hz', f, sugar, []) for f in [200, 100] + [f for f in range(20, 181, 20) if f != 100]],
        'silencing': [(f'sugarR_100Hz-{i}', 100, sugar, [i]) for i in EXAMPLE_SILENCED],
        # independent repeats of the 200 Hz run: the empirical run-to-run variability of the model
        'replicates': [(f'sugarR_200Hz_rep{k}', 200, sugar, []) for k in range(2, 5)],
        # a few rates for acceptance-test controls (run with --w-syn-scale)
        'control_rates': [(f'sugarR_{f}Hz', f, sugar, []) for f in (50, 100, 200)],
    }
    if dataset == '783':  # annotation-defined GRN sets (flywire_annotations v3.2.0 IDs are v783)
        sets = grn_sets_783()
        g['grn_classes'] = [(f'{name}_{f}Hz', f, ids, []) for f in (100, 50, 200) for name, ids in sets.items()
                            if not (name == 'shiu21' and f in (100, 200))]  # those two are in archived_freqs
    return g


def provenance(dataset, exp_name, r_poi, n_exc, slnc, n_proc, walltime, used_params):
    commit = subprocess.run(['git', '-C', str(REF), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    params = {k: str(v) for k, v in used_params.items() if k != 'eqs'}
    return {
        'exp_name': exp_name, 'dataset': dataset, 'r_poi_Hz': r_poi, 'n_excited': n_exc, 'silenced': slnc,
        'params': params, 'eqs': default_params['eqs'], 'n_proc': n_proc, 'walltime_s': round(walltime, 1),
        'reference_commit': commit, 'brian2': brian2.__version__, 'numpy': numpy.__version__,
        'python': platform.python_version(), 'machine': platform.platform(),
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=DATASETS, default='630')
    ap.add_argument("--group", nargs="+", default=["fig1d"], help="see groups()")
    ap.add_argument('--n-proc', type=int, default=6, help='parallel trials; each needs ~2.2 GB RAM')
    ap.add_argument('--n-run', type=int, default=default_params['n_run'])
    ap.add_argument('--w-syn-scale', type=float, default=1.0,
                    help='scale the per-synapse weight (a deliberately wrong model, e.g. for acceptance-test controls)')
    args = ap.parse_args()

    path_comp, path_con = DATASETS[args.dataset]
    suffix = '' if args.w_syn_scale == 1.0 else f'_wsyn{args.w_syn_scale:g}'
    out = ROOT / 'results' / 'stage1' / 'runs' / f'v{args.dataset}{suffix}'
    out.mkdir(parents=True, exist_ok=True)

    for g in args.group:
        for exp_name, r_poi, exc, slnc in groups(args.dataset)[g]:
            if (out / f'{exp_name}.parquet').is_file():
                print(f'>>> skip {exp_name} (exists)', flush=True)
                continue
            params = deepcopy(default_params)
            params['r_poi'] = r_poi * Hz
            params['n_run'] = args.n_run
            params['w_syn'] = default_params['w_syn'] * args.w_syn_scale
            t0 = time.time()
            run_exp(exp_name=exp_name, neu_exc=exc, neu_slnc=slnc, path_res=out, path_comp=path_comp,
                    path_con=path_con, params=params, n_proc=args.n_proc)
            meta = provenance(args.dataset, exp_name, r_poi, len(exc), slnc, args.n_proc, time.time() - t0, params)
            (out / f'{exp_name}.json').write_text(json.dumps(meta, indent=1))
            print(f'    done {exp_name} in {meta["walltime_s"]} s', flush=True)


if __name__ == '__main__':
    main()
