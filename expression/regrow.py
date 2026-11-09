"""Compress-and-regrow: rebuild the fly connectome from block rules and write it as a Shiu-format table + MLX pack.

Generator: a degree-corrected (not strength-corrected) microcanonical
block model. Every edge keeps its synapse count; its postsynaptic end is shuffled among edges whose postsynaptic neuron
is in the same group, then its presynaptic end likewise. Every group->group edge count and weight multiset, and every
neuron's in- and out-degree, are preserved; which neuron of a group connects to which is randomised, so a neuron's
total input and output synapses are re-drawn within its group. Self-loops created by the shuffle are dropped (under
0.08% of synapses) and parallel edges merged. Groups always include the neuron's transmitter sign, so excitatory and
inhibitory cells are never mixed.

Levels. The ladder is side-aware at every level (rules may differ between ipsi- and contralateral targets):
  type_x_side, supertype (the modal supertype of each cell type, x side), hemilineage (Ito-Lee hemilineage x side;
  neurons without one fall back to cell class x side), cell_class x side, super_class x side.
Plus one side-blind series, type_sides_pooled (central cell types with both hemispheres pooled), and two controls,
supertype_keep_readouts and hemilineage_keep_readouts: the same rules, except that the battery's readouts (every DN,
and aBN1) keep their own type x side groups. They separate a loss of central routing from the pooling of the readout
neurons themselves. A third control, hemilineage_else_supertype, groups neurons without an Ito-Lee hemilineage by
supertype instead of the cell-class/super-class fallback of the hemilineage level; the fourth pools sides but keeps the
readouts' own sides.
At every level the periphery (sensory, optic, visual projection/centrifugal, motor, endocrine) is grouped at
type x side, because receptor and photoreceptor identity are genetically explicit; it is still regrown within those
groups, so e.g. optic-lobe retinotopy is scrambled at every level. The 208 neurons the behaviour battery stimulates
(expression/stimsets.py) are singleton groups at every level, so their drive is preserved exactly.
Each (level, seed) has its own random stream.

    .venv/bin/python expression/regrow.py type_x_side 0      # -> data/wiring/regrow/type_x_side_s0/ and an MLX pack
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'wiring' / 'v783_alfix_all'
OUT = ROOT / 'data' / 'wiring' / 'regrow'
MLX = ROOT / 'external' / 'drosophila-brain-mlx'
ANN = ROOT / 'data' / 'flywire_annotations' / 'Supplemental_file1_neuron_annotations.tsv'
PERIPHERY = {'sensory', 'optic', 'visual_projection', 'visual_centrifugal', 'motor', 'endocrine', 'sensory_ascending'}
LADDER = ['type_x_side', 'supertype', 'hemilineage', 'cell_class', 'super_class']
CONTROLS = ['supertype_keep_readouts', 'hemilineage_keep_readouts',             # appended in this order: the
            'hemilineage_else_supertype', 'type_sides_pooled_keep_readouts']     # index fixes each random stream
LEVELS = LADDER + ['type_sides_pooled'] + CONTROLS     # order matters: a level's index here keys its random stream


def group_labels(level, ids, sign):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stimsets import ABN1, STIMULATED
    keep_readouts = level.endswith('_keep_readouts')
    level = level.replace('_keep_readouts', '')
    a = pd.read_csv(ANN, sep='\t', low_memory=False).drop_duplicates('root_id').set_index('root_id').reindex(ids)
    side = np.asarray(a['side'].fillna('na').astype(str).values, dtype=str)
    ctype = a['cell_type'].astype(object).values.copy()
    unt = pd.isna(ctype)
    ctype[unt] = np.array([f'untyped_{i}' for i in np.flatnonzero(unt)], dtype=object)
    ctype = ctype.astype(str)
    sc = np.asarray(a['super_class'].fillna('na').astype(str).values, dtype=str)
    cc = a['cell_class'].astype(object).values
    cc = np.where(pd.isna(cc), sc, cc).astype(str)
    periph = np.isin(sc, list(PERIPHERY))
    tside = np.char.add(np.char.add(ctype, '|'), side)
    if level == 'type_x_side':
        lab = tside
    elif level == 'type_sides_pooled':
        lab = ctype
    elif level == 'supertype':
        st = pd.Series(a['supertype'].values, index=ctype).dropna().astype(str)
        modal = st.groupby(level=0).agg(lambda x: x.value_counts().index[0])   # each cell type -> one supertype
        lab = np.char.add(np.char.add(pd.Series(ctype).map(modal).fillna(pd.Series(ctype)).values.astype(str), '|'), side)
    elif level in ('hemilineage', 'hemilineage_else_supertype'):
        hl = a['ito_lee_hemilineage'].astype(object).values
        ok = ~pd.isna(hl) & (hl.astype(str) != 'putative_primary')
        if level == 'hemilineage':
            fallback = np.char.add('cc:', cc)                              # cell class, or super class if none
        else:
            st = pd.Series(a['supertype'].values, index=ctype).dropna().astype(str)
            modal = st.groupby(level=0).agg(lambda x: x.value_counts().index[0])
            fallback = np.char.add('st:', pd.Series(ctype).map(modal).fillna(pd.Series(ctype)).values.astype(str))
        lab = np.char.add(np.char.add(np.where(ok, hl.astype(str), fallback), '|'), side)
    elif level == 'cell_class':
        lab = np.char.add(np.char.add(cc, '|'), side)
    elif level == 'super_class':
        lab = np.char.add(np.char.add(sc, '|'), side)
    else:
        raise ValueError(level)
    if keep_readouts:                                    # control: every DN and aBN1 keep their own type x side groups
        lab = np.where((sc == 'descending') | (ids == ABN1), tside, lab)
    g = np.where(periph, tside, lab).astype(object)
    stim = np.isin(ids, STIMULATED)
    g[stim] = np.array([f'stim:{i}' for i in ids[stim]], dtype=object)       # battery inputs keep their own wiring
    g = np.char.add(np.char.add(g.astype(str), '|'), np.where(sign > 0, 'E', 'I'))
    code, uniq = pd.factorize(pd.Series(g))
    return code, len(uniq)


def regrow(level, seed):
    t0 = time.time()
    comp = pd.read_csv(BASE / 'Completeness_783.csv', index_col=0)
    ids = comp.index.values
    con = pd.read_parquet(BASE / 'Connectivity_783.parquet')
    n = len(ids)
    sign = np.ones(n, dtype=np.int8)  # per-neuron sign (the table has no neuron with mixed-sign outputs)
    sign[con.Presynaptic_Index.values] = np.sign(con.Excitatory.values).astype(np.int8)
    code, n_groups = group_labels(level, ids, sign)
    rng = np.random.default_rng([seed, LEVELS.index(level)])            # independent stream per (level, seed)
    pre = con.Presynaptic_Index.values.astype(np.int64).copy()
    post = con.Postsynaptic_Index.values.astype(np.int64).copy()
    w = con.Connectivity.values.astype(np.int64)
    for arr in (post, pre):
        grp = code[arr]
        order = np.lexsort((rng.random(len(arr)), grp))
        arr[np.argsort(grp, kind='stable')] = arr[order]
    keep = pre != post
    key = pre[keep] * n + post[keep]
    uk, inv = np.unique(key, return_inverse=True)
    wsum = np.bincount(inv, weights=w[keep]).astype(np.int64)
    p_idx, q_idx = uk // n, uk % n
    s = sign[p_idx].astype(np.int64)
    out = pd.DataFrame({'Presynaptic_ID': ids[p_idx], 'Postsynaptic_ID': ids[q_idx],
                        'Presynaptic_Index': p_idx, 'Postsynaptic_Index': q_idx,
                        'Connectivity': wsum, 'Excitatory': s, 'Excitatory x Connectivity': s * wsum})
    orig = con.Presynaptic_Index.values.astype(np.int64) * n + con.Postsynaptic_Index.values
    stats = {'level': level, 'seed': seed, 'n_groups': int(n_groups), 'edges_in': int(len(con)),
             'edges_out': int(len(out)), 'self_loops_dropped': int((~keep).sum()),
             'synapses_dropped_self_loops': int(w[~keep].sum()), 'duplicates_merged': int(keep.sum() - len(uk)),
             'frac_edges_also_in_real': float(np.isin(uk, orig).mean()), 'seconds': round(time.time() - t0, 1)}
    d = OUT / f'{level}_s{seed}'
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / 'Connectivity_783.parquet')
    shutil.copy(BASE / 'Completeness_783.csv', d / 'Completeness_783.csv')
    pack = MLX / 'data' / 'pack' / f'regrow_{level}_s{seed}'
    r = subprocess.run([str(ROOT / '.venv-mlx' / 'bin' / 'python'), '-m', 'lif.compile_pack', '--completeness',
                        str(d / 'Completeness_783.csv'), '--connectivity', str(d / 'Connectivity_783.parquet'),
                        '--out', str(pack)], cwd=MLX, capture_output=True, text=True)
    stats['pack_ok'] = 'ALL CHECKS PASSED' in r.stdout
    if r.returncode != 0 or not stats['pack_ok']:
        raise RuntimeError(f'compiling {pack.name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    stats['seconds_total'] = round(time.time() - t0, 1)
    (d / 'regrow.json').write_text(json.dumps(stats, indent=1))
    (d / 'Connectivity_783.parquet').unlink()  # the pack holds the wiring; keep disk use bounded
    print(stats, flush=True)
    return stats


def _write_and_compile(out, name, stats, t0):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / 'Connectivity_783.parquet')
    shutil.copy(BASE / 'Completeness_783.csv', d / 'Completeness_783.csv')
    pack = MLX / 'data' / 'pack' / f'regrow_{name}'
    r = subprocess.run([str(ROOT / '.venv-mlx' / 'bin' / 'python'), '-m', 'lif.compile_pack', '--completeness',
                        str(d / 'Completeness_783.csv'), '--connectivity', str(d / 'Connectivity_783.parquet'),
                        '--out', str(pack)], cwd=MLX, capture_output=True, text=True)
    stats['pack_ok'] = 'ALL CHECKS PASSED' in r.stdout
    if r.returncode != 0 or not stats['pack_ok']:
        raise RuntimeError(f'compiling {pack.name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    stats['seconds_total'] = round(time.time() - t0, 1)
    (d / 'regrow.json').write_text(json.dumps(stats, indent=1))
    (d / 'Connectivity_783.parquet').unlink()
    print(stats, flush=True)
    return stats


def reweigh(level, seed):
    """The complementary ablation, after Shibusawa 2026 (MaleCNS), whose within-type shuffle pooled the two hemispheres
    (at type_x_side ours keeps them apart): keep every connection, shuffle synapse counts among the connections of the
    same group pair. Topology and signs exact; which connection is strong is randomised."""
    t0 = time.time()
    comp = pd.read_csv(BASE / 'Completeness_783.csv', index_col=0)
    ids = comp.index.values
    con = pd.read_parquet(BASE / 'Connectivity_783.parquet')
    n = len(ids)
    sign = np.ones(n, dtype=np.int8)
    sign[con.Presynaptic_Index.values] = np.sign(con.Excitatory.values).astype(np.int8)
    code, n_groups = group_labels(level, ids, sign)
    rng = np.random.default_rng([seed, 100])
    pre = con.Presynaptic_Index.values.astype(np.int64)
    post = con.Postsynaptic_Index.values.astype(np.int64)
    w = con.Connectivity.values.astype(np.int64).copy()
    blk = code[pre].astype(np.int64) * n_groups + code[post]
    order = np.lexsort((rng.random(len(w)), blk))
    w[np.argsort(blk, kind='stable')] = w[order]
    s = sign[pre].astype(np.int64)
    out = pd.DataFrame({'Presynaptic_ID': ids[pre], 'Postsynaptic_ID': ids[post], 'Presynaptic_Index': pre,
                        'Postsynaptic_Index': post, 'Connectivity': w, 'Excitatory': s, 'Excitatory x Connectivity': s * w})
    w0 = con.Connectivity.values
    stats = {'level': level, 'kind': 'weights_shuffled', 'seed': seed, 'n_groups': int(n_groups), 'edges': int(len(out)),
             'frac_edges_same_weight': float((w == w0).mean()),
             'weight_r_vs_real': float(np.corrcoef(np.log(w), np.log(w0))[0, 1]),
             'frac_edges_also_in_real': 1.0, 'seconds': round(time.time() - t0, 1)}
    return _write_and_compile(out, f'wshuffle_{level}_s{seed}', stats, t0)


def _built(name):
    """A brain counts as built once its MLX pack exists (regrow.json, the statistics, is kept in git)."""
    return (MLX / 'data' / 'pack' / f'regrow_{name}' / 'manifest.json').is_file()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'wshuffle':
        for sd in ([int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 1]):
            lv = sys.argv[2] if len(sys.argv) > 2 else 'type_x_side'
            if not _built(f'wshuffle_{lv}_s{sd}'):
                reweigh(lv, sd)
        sys.exit(0)
    levels = sys.argv[1].split(',') if len(sys.argv) > 1 else LEVELS
    seeds = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1, 2]
    for lv in levels:
        for sd in seeds:
            if _built(f'{lv}_s{sd}'):
                continue
            regrow(lv, sd)
