"""Wiring variants of FlyWire v783 for parts 2 and 3, written as Shiu-format tables.

published : Shiu et al.'s Connectivity_783.parquet unchanged.
alfix_all : every antennal-lobe local neuron (cell_class ALLN, flywire_annotations v3.2.0; 429 cells) made
            inhibitory. Removing or re-signing these cells abolishes the whole-brain, odour-evoked runaway state
            (xiangdoz/fly-connectome-attractor; suifei/flywire-fly-lab section 48.1); a few glomeruli (e.g. ORN_DC4)
            still ignite a smaller, dose-insensitive LAL/VES/CRE state of about 550-1,150 neurons (expression/robustness.py).
alfix_ach : ALLN predicted (top_nt) or known (known_nt) cholinergic keep their published sign (112 cells, 8 of which
            are already inhibitory in the published table); all others are made inhibitory. Most AL local neurons are
            GABAergic or glutamatergic, both inhibitory in the AL (Wilson & Laurent 2005; Liu & Wilson 2013), with a
            minority of cholinergic excitatory LNs (Shang 2007).

    .venv/bin/python expression/wiring.py alfix_ach   # -> data/wiring/v783_alfix_ach/{Connectivity,Completeness}
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'external' / 'Drosophila_brain_model'
ANN = ROOT / 'data' / 'flywire_annotations' / 'Supplemental_file1_neuron_annotations.tsv'
OUT = ROOT / 'data' / 'wiring'


def annotations():
    return pd.read_csv(ANN, sep='\t', low_memory=False,
                       usecols=['root_id', 'super_class', 'cell_class', 'cell_sub_class', 'cell_type', 'hemibrain_type',
                                'ito_lee_hemilineage', 'top_nt', 'top_nt_conf', 'known_nt', 'side', 'supertype'])


def alln_to_inhibit(variant, ann=None):
    ann = annotations() if ann is None else ann
    al = ann[ann.cell_class == 'ALLN']
    if variant == 'alfix_all':
        return set(al.root_id.astype(np.int64))
    if variant == 'alfix_ach':
        ach = (al.top_nt == 'acetylcholine') | al.known_nt.astype(str).str.startswith('acetylcholine')
        ach &= ~al.known_nt.astype(str).str.startswith('gaba')
        return set(al[~ach].root_id.astype(np.int64))
    raise ValueError(variant)


def write_variant(variant):
    con = pd.read_parquet(REF / 'Connectivity_783.parquet')
    ids = alln_to_inhibit(variant)
    pre = con['Presynaptic_ID'].isin(ids).to_numpy()
    flip = pre & (con['Excitatory'].to_numpy() > 0)
    con.loc[pre, 'Excitatory'] = -1
    con.loc[pre, 'Excitatory x Connectivity'] = -con.loc[pre, 'Connectivity']
    d = OUT / f'v783_{variant}'
    d.mkdir(parents=True, exist_ok=True)
    con.to_parquet(d / 'Connectivity_783.parquet')
    shutil.copy(REF / 'Completeness_783.csv', d / 'Completeness_783.csv')
    n_exc = con.loc[flip, 'Presynaptic_ID'].nunique()
    print(f'{variant}: {len(ids)} ALLN set inhibitory ({n_exc} of them excitatory in the published table); '
          f'{int(flip.sum())} of {int(pre.sum())} of their edges flipped (+ to -); written to {d}')


if __name__ == '__main__':
    for v in sys.argv[1:]:
        write_variant(v)
