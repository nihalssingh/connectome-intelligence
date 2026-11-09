"""Run the Fig. 1D sugar sweep on the Apple-Silicon MLX port (Kisame76/drosophila-brain-mlx), for acceptance testing.

The port's run_exp writes the same parquet columns as the published model, so stage1/acceptance.py
scores its output exactly like a Brian2 run. Each rate gets its own seed (the port draws trial n with
seed + n, so a shared seed would correlate the input across rates).

    .venv-mlx/bin/python stage1/run_mlx.py --pack external/drosophila-brain-mlx/data/pack/v630
    .venv/bin/python stage1/acceptance.py results/stage1/runs/mlx_v630
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MLX = ROOT / 'external' / 'drosophila-brain-mlx'
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lif import core  # noqa: E402
from lif.experiment import default_params, run_exp  # noqa: E402

from neurons import SUGAR_R, to_v783  # noqa: E402

FIG1D_FREQS = list(range(10, 201, 10))


def _rel(p):
    """Path relative to the repo root when possible, so provenance files hold no machine-specific paths."""
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pack', type=Path, default=MLX / 'data' / 'pack' / 'v630')
    ap.add_argument('--out', type=Path, default=ROOT / 'results' / 'stage1' / 'runs' / 'mlx_v630')
    ap.add_argument('--freqs', type=int, nargs='+', default=FIG1D_FREQS)
    ap.add_argument('--n-run', type=int, default=30)
    ap.add_argument('--engine', default='fused')
    ap.add_argument('--v783', action='store_true', help='map the sugar GRN IDs to v783 root IDs')
    args = ap.parse_args()

    pack = core.load_pack(args.pack)
    args.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(['git', '-C', str(MLX), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    for f in args.freqs:
        name = f'sugarR_{f}Hz'
        if (args.out / f'{name}.parquet').is_file():
            print(f'>>> skip {name} (exists)', flush=True)
            continue
        params = dict(default_params)
        params['r_poi'] = float(f)
        params['n_run'] = args.n_run
        t0 = time.time()
        run_exp(name, to_v783(SUGAR_R) if args.v783 else SUGAR_R, args.out, params=params, pack=pack, seed=1000 * f, engine=args.engine, edge_split=1)
        meta = {'exp_name': name, 'r_poi_Hz': f, 'n_run': args.n_run, 'seed': 1000 * f, 'engine': args.engine,
                'pack': _rel(args.pack), 'ids_mapped_to_v783': bool(args.v783),
                'note': "the pack manifest's 'dataset' field says flywire-v630 for every pack; do not rely on it",
                'mlx_port_commit': commit, 'walltime_s': round(time.time() - t0, 1),
                'machine': platform.platform(), 'python': platform.python_version(),
                'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
        (args.out / f'{name}.json').write_text(json.dumps(meta, indent=1))
        print(f'    done {name} in {meta["walltime_s"]} s', flush=True)


if __name__ == '__main__':
    main()
