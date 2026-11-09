"""What the regrowth levels actually change along the tested pathways.

From the compiled MLX packs (source-major CSR) of the real corrected brain and of every regrown brain:
1. Share of connections (and of synapses) that also exist in the real brain, split into central-brain connections
   (both ends outside the periphery), connections with an optic-lobe end, and the rest.
2. Share of the readouts' input synapses on real connections: the 766 one-per-side DNs, MN9 (both cells), aBN1.
3. The 208 battery-driven neurons: share of their output synapses on real connections.
From the group labels of each level (expression/regrow.py):
4. How many central neurons and DNs are regrouped relative to cell type x side, the group context of a DN, and the
   hemilineage level's fallback (neurons without an Ito-Lee hemilineage are grouped by cell class, or super class).

    .venv-mlx/bin/python expression/compress_checks.py      # -> results/expression/compression_checks.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_analysis import BAT, PACKS, REAL, REP, WSHUFFLE, _base, _r, dn_pairs, fidelity, load  # noqa: E402
from battery import PROBE_TYPES  # noqa: E402
from census import library  # noqa: E402
from wiring import annotations  # noqa: E402
from regrow import ANN, LEVELS, PERIPHERY, group_labels  # noqa: E402
from stimsets import ABN1, MN9C, MN9I, STIMULATED, SUGAR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'expression' / 'compression_checks.json'


def edges(pack):
    d = PACKS / pack
    rp = np.load(d / 'row_ptr.npy').astype(np.int64)
    dst = np.load(d / 'destinations.npy').astype(np.int64)
    w = np.abs(np.load(d / 'signed_counts.npy').astype(np.int64))
    src = np.repeat(np.arange(len(rp) - 1, dtype=np.int64), np.diff(rp))
    return np.load(d / 'neuron_ids.npy'), src, dst, w


def floor_on(ref, probes, pairs, n_boot=2000, seed=0):
    """compress_analysis.mirror_floor restricted to some (left, right) DN column pairs."""
    sid = list(probes)
    rs = []
    for i, s in enumerate(sid):
        if not s.endswith(':left') or s.replace(':left', ':right') not in sid:
            continue
        j = sid.index(s.replace(':left', ':right'))
        L, R = ref[i].mean(0), ref[j].mean(0)
        r = _r(np.r_[L[pairs[:, 0]], L[pairs[:, 1]]], np.r_[R[pairs[:, 1]], R[pairs[:, 0]]])
        if r is not None:
            rs.append(r)
    rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(rs, len(rs))) for _ in range(n_boot)]
    return {'median_r': float(np.median(rs)), 'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], 'n': len(rs)}


def dn_word_splits(ids, sign, pos, ann_ids):
    """DN words on subsets of the one-per-side DN pairs: pairs a level regroups vs pairs it leaves alone (supertype),
    and pairs with vs without an Ito-Lee hemilineage (hemilineage). Plus a relabelling baseline for multi-cell DNs."""
    jr, ref, probes, dn_ids = load(REAL)
    _, rep, _, _ = load(REP)
    pairs = dn_pairs(dn_ids)
    px = np.array([[pos[int(dn_ids[a])], pos[int(dn_ids[b])]] for a, b in pairs])   # pack indices of each pair
    t_code, _ = group_labels('type_x_side', ids, sign)
    out = {}
    hl = ann_ids.ito_lee_hemilineage.astype(object).values
    has_hl = ~pd.isna(hl) & (hl.astype(str) != 'putative_primary')
    for lv, split in (('supertype', 'regrouped'), ('hemilineage', 'hemilineage')):
        code, T = group_labels(lv, ids, sign)
        n_tside = pd.Series(t_code).groupby(code).nunique().reindex(range(T)).fillna(0).values
        if split == 'regrouped':
            m = (n_tside[code[px]] > 1).any(1)                         # either member merged with other type x side groups
            names = ('pairs_regrouped', 'pairs_left_alone')
        else:
            m = ~has_hl[px].all(1)                                      # either member without a hemilineage (fallback)
            names = ('pairs_fallback', 'pairs_with_hemilineage')
        row = {}
        for nm, sel in zip(names, (m, ~m)):
            cols = np.sort(pairs[sel].ravel())
            row[nm] = {'n_pairs': int(sel.sum()), 'yardstick': floor_on(ref, probes, pairs[sel]),
                       'brains': [fidelity(ref, load(f'regrow_{lv}_s{sd}')[1], cols)['median_r'] for sd in (0, 1, 2)
                                  if (BAT / f'regrow_{lv}_s{sd}.json').is_file()]}
        out[lv] = row
    # relabelling baseline: real type-mates of multi-cell DN types, permuted within type x side, against the re-run
    an = ann_ids.reindex(dn_ids)
    multi = np.setdiff1d(np.arange(len(dn_ids)), pairs.ravel())
    key = np.char.add(np.char.add(an.cell_type.astype(str).values[multi], '|'), an.side.astype(str).values[multi])
    rng = np.random.default_rng(7)
    rs = []
    for _ in range(20):
        perm = multi.copy()
        for k in np.unique(key):
            ix = np.flatnonzero(key == k)
            perm[ix] = multi[ix][rng.permutation(len(ix))]
        rs.append(fidelity(rep[..., multi], ref[..., perm])['median_r'])
    out['multi_DN_relabelled_within_type_vs_rerun'] = {'median': float(np.median(rs)), 'min': float(np.min(rs)), 'max': float(np.max(rs))}
    return out


def main():
    ann = pd.read_csv(ANN, sep='\t', low_memory=False).drop_duplicates('root_id').set_index('root_id')
    ids, rs, rd, rw = edges(REAL)
    n = len(ids)
    a = ann.reindex(ids)
    sc = a.super_class.astype(str).values
    central = ~np.isin(sc, list(PERIPHERY))
    optic = sc == 'optic'
    order = np.argsort(rs * n + rd)
    real_keys, real_w = (rs * n + rd)[order], rw[order]
    dn_ids = np.load(ROOT / 'results' / 'expression' / 'census' / REAL / 'readouts.npz')['dn_ids']
    pos = {int(i): k for k, i in enumerate(ids)}
    paired = np.array([pos[int(i)] for i in dn_ids[np.sort(dn_pairs(dn_ids).ravel())]])
    mn9 = np.array([pos[int(MN9C)], pos[int(MN9I)]])
    abn1 = np.array([pos[int(ABN1)]])
    stim = np.array([pos[int(i)] for i in set(STIMULATED) if int(i) in pos])
    probe_cells = np.array(sorted({pos[int(i)] for x in library(annotations()) if x['cell_type'] in PROBE_TYPES
                                   for i in x['ids'] if int(i) in pos}))
    sugar = np.array([pos[int(i)] for i in set(SUGAR) if int(i) in pos])

    def on_real(keys):
        i = np.minimum(np.searchsorted(real_keys, keys), len(real_keys) - 1)
        return real_keys[i] == keys, real_w[i]

    def brain(pack):
        bid, s, d, w = edges(pack)
        assert np.array_equal(bid, ids), pack
        real, w0 = on_real(s * n + d)
        cc, op = central[s] & central[d], optic[s] | optic[d]
        frac = lambda m: {'edges': float(real[m].mean()), 'synapses': float(w[m & real].sum() / w[m].sum())}
        inp = lambda tgt: float(w[np.isin(d, tgt) & real].sum() / w[np.isin(d, tgt)].sum())
        same = real & (w == w0)                                         # the real pair with its real synapse count
        inp_same = lambda tgt: float(w[np.isin(d, tgt) & same].sum() / w[np.isin(d, tgt)].sum())
        po = np.isin(s, probe_cells)
        return {'all': frac(np.ones(len(s), bool)), 'central_to_central': frac(cc), 'optic_lobe': frac(op),
                'other': frac(~cc & ~op),
                'readout_input_synapses_on_real_edges': {'paired_DNs': inp(paired), 'MN9': inp(mn9), 'aBN1': inp(abn1)},
                'battery_neuron_output_synapses_on_real_edges': float(w[np.isin(s, stim) & real].sum() / w[np.isin(s, stim)].sum()),
                'same_pair_and_count': {'central_to_central_synapses': float(w[cc & same].sum() / w[cc].sum()),
                                        'paired_DN_inputs': inp_same(paired), 'MN9_inputs': inp_same(mn9), 'aBN1_inputs': inp_same(abn1),
                                        'battery_neuron_outputs': float(w[np.isin(s, stim) & same].sum() / w[np.isin(s, stim)].sum())},
                'probe_cell_output_synapses_on_real_edges': {'all': float(w[po & real].sum() / w[po].sum()),
                                                             'onto_central': float(w[po & real & central[d]].sum() / w[po & central[d]].sum())},
                # for count-shuffled brains (every connection real): share of the real synapses whose connection kept its count
                'real_synapses_on_unchanged_counts': {k: float(w0[m & real & (w == w0)].sum() / w0[m & real].sum())
                                                      for k, m in (('all', np.ones(len(s), bool)), ('paired_DN_inputs', np.isin(d, paired)),
                                                                   ('MN9_inputs', np.isin(d, mn9)), ('aBN1_inputs', np.isin(d, abn1)))}}

    res = {'brains': {}, 'levels': {}}
    for lv in LEVELS + WSHUFFLE:
        for sd in (0, 1, 2):
            tag = f'regrow_{lv}_s{sd}'
            if (BAT / f'{tag}.json').is_file() and (PACKS / tag).is_dir():
                res['brains'][tag] = brain(tag)
                print(tag, json.dumps(res['brains'][tag])[:240], flush=True)
    # group-level facts (do not depend on the seed)
    b_ids, con, sign = _base()
    assert np.array_equal(b_ids, ids)
    t_code, _ = group_labels('type_x_side', ids, sign)
    mn9_in = np.isin(rd, mn9)
    sug_out = np.isin(rs, sugar)
    hl = a.ito_lee_hemilineage.astype(object).values
    no_hl = central & (pd.isna(hl) | (hl.astype(str) == 'putative_primary'))
    pp = central & ~pd.isna(hl) & (hl.astype(str) == 'putative_primary')     # Ito-Lee 'putative primary' neurons
    na = central & pd.isna(hl)                                               # no Ito-Lee annotation at all
    dn_all = np.array([pos[int(i)] for i in dn_ids])
    ct = a.cell_type.astype(str).values
    for lv in LEVELS:
        code, T = group_labels(lv, ids, sign)
        size = np.bincount(code, minlength=T)
        n_tside = pd.Series(t_code).groupby(code).nunique().reindex(range(T)).fillna(0).values
        regrouped = n_tside[code] > 1                      # the neuron's group merges several type x side groups
        dn_types = pd.Series(ct[dn_all]).groupby(code[dn_all]).nunique()
        is_dn = np.zeros(n, bool)
        is_dn[dn_all] = True
        row = {'groups': int(T), 'frac_central_regrouped': float(regrouped[central].mean()),
               'MN9_input_synapses_from_regrouped_DNs': float(rw[mn9_in & is_dn[rs] & regrouped[rs]].sum() / rw[mn9_in].sum()),
               'frac_DNs_regrouped': float(regrouped[dn_all].mean()),
               'frac_paired_DNs_regrouped': float(regrouped[paired].mean()),
               'MN9_input_synapses_from_unregrouped_neurons': float(rw[mn9_in & ~regrouped[rs]].sum() / rw[mn9_in].sum()),
               'sugar_output_synapses_to_unregrouped_neurons': float(rw[sug_out & ~regrouped[rd]].sum() / rw[sug_out].sum()),
               'DN_group_context_median': {'DN_types': float(np.median(dn_types.reindex(code[dn_all]).values)),
                                           'neurons': float(np.median(size[code[dn_all]]))}}
        if lv.startswith('hemilineage') and lv != 'hemilineage_else_supertype':
            row['hemilineage_fallback'] = {
                'central_neurons_without_hemilineage': int(no_hl.sum()), 'frac_of_central': float(no_hl[central].mean()),
                'largest_fallback_group': int(size[np.unique(code[no_hl])].max()),
                'MN9_input_synapses_from_fallback_neurons': float(rw[mn9_in & no_hl[rs]].sum() / rw[mn9_in].sum()),
                'sugar_output_synapses_to_fallback_neurons': float(rw[sug_out & no_hl[rd]].sum() / rw[sug_out].sum()),
                'paired_DNs_without_hemilineage': int(no_hl[paired].sum()), 'aBN1_without_hemilineage': bool(no_hl[abn1][0]),
                'putative_primary': {'neurons': int(pp.sum()), 'MN9_input_synapses': float(rw[mn9_in & pp[rs]].sum() / rw[mn9_in].sum()),
                                     'sugar_output_synapses': float(rw[sug_out & pp[rd]].sum() / rw[sug_out].sum())},
                'unannotated': {'neurons': int(na.sum()), 'MN9_input_synapses': float(rw[mn9_in & na[rs]].sum() / rw[mn9_in].sum()),
                                'sugar_output_synapses': float(rw[sug_out & na[rd]].sum() / rw[sug_out].sum())},
                'aBN1_putative_primary': bool(pp[abn1][0])}
        res['levels'][lv] = row
        print(lv, json.dumps(row)[:300], flush=True)
    res['n_paired_DNs'], res['n_battery_neurons'], res['n_sugar_GRNs'] = int(len(paired)), int(len(stim)), int(len(sugar))
    res['n_probe_cells'] = int(len(probe_cells))
    res['n_probe_cells_also_battery_singletons'] = int(len(np.intersect1d(probe_cells, stim)))
    res['dn_word_splits'] = dn_word_splits(ids, sign, pos, a)
    by_sc = pd.Series(rw[mn9_in], index=sc[rs[mn9_in]]).groupby(level=0).sum()
    res['MN9_input_synapses_by_super_class'] = {k: float(v / by_sc.sum()) for k, v in by_sc.sort_values(ascending=False).items()}
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
