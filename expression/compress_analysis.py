"""Compression analysis: do brains regrown from block rules still behave, and still 'say' the same things?

For each regrown brain (expression/regrow.py + battery.py): behaviour calls in the style of Shiu et al.'s binary
predictions, DN-word fidelity against the real (corrected) brain on 46 probe stimuli, and the description length of
the rules it was grown from. Yardsticks: the real brain re-run with independent noise (ceiling) and the real brain's
left vs right hemisphere (two developmental replicates of the same circuit, but also two reconstructions).
Similarity is computed over the DNs that respond in either brain (as in the hemisphere comparison of analyze.py); the
median over all paired DNs, shared silence included, is also reported (median_r_incl_shared_silence).

    .venv-mlx/bin/python expression/compress_analysis.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regrow import BASE, LADDER, LEVELS, group_labels  # noqa: E402
from stimsets import CONDITIONS  # noqa: E402
from wiring import annotations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BAT = ROOT / 'results' / 'expression' / 'battery'
REAL, REP = 'v783_alfix_all', 'v783_alfix_all_rep500'   # rep500: independent trial seeds (offset >= n_trials)
GENOME_MB = 35.9         # D. melanogaster release 6 assembly, 2 bits per base (context only; not a wiring list)
N_TRIALS = 10
PACKS = ROOT / 'external' / 'drosophila-brain-mlx' / 'data' / 'pack'
WSHUFFLE = ['wshuffle_type_x_side']   # complementary ablation: exact wiring, counts shuffled within type x side blocks


def load(tag):
    j = json.loads((BAT / f'{tag}.json').read_text())
    z = np.load(BAT / f'{tag}.npz')
    return j, np.sqrt(z['dn'].astype(np.float32)), z['probes'], z['dn_ids']


def behaviours(j, n=N_TRIALS):
    """Binary calls in the style of Shiu et al.'s predictions. A difference counts only if it exceeds 2 standard errors
    (from the stored trial SDs); calls that depend on a base behaviour are None ('not testable') when it is absent."""
    c = j['conditions']
    m = lambda k, r='MN9_contra': c[k][r]
    se = lambda k, r='MN9_contra': c[k][r + '_sd'] / math.sqrt(n)
    diff = lambda a, b, ra='MN9_contra', rb='MN9_contra': (m(a, ra) - m(b, rb)) > 2 * math.hypot(se(a, ra), se(b, rb))
    s100 = m('sugar_100')
    b = {'MN9_sugar_50': m('sugar_50'), 'MN9_sugar_100': s100, 'MN9_sugar_200': m('sugar_200'),
         'bitter_ratio': m('sugar_bitter_100') / s100 if s100 > 0 else math.nan,
         'ir94e_ratio': m('sugar_ir94e_100') / s100 if s100 > 0 else math.nan,
         'water_MN9': m('water_100'), 'aBN1_JON_CE': m('JON_CE_100', 'aBN1'), 'aBN1_JON_F': m('JON_F_100', 'aBN1'),
         'MN9_ipsi_100': m('sugar_100', 'MN9_ipsi'), 'n_active_sugar_100': c['sugar_100']['n_active']}
    act = s100 > max(1.0, 2 * se('sugar_100'))
    ce = b['aBN1_JON_CE'] > max(1.0, 2 * se('JON_CE_100', 'aBN1'))
    calls = {'sugar activates MN9': act,
             'MN9 graded with sugar rate': (diff('sugar_100', 'sugar_50') and diff('sugar_200', 'sugar_100')) if act else None,
             'contra MN9 > ipsi MN9': diff('sugar_100', 'sugar_100', 'MN9_contra', 'MN9_ipsi') if act else None,
             'bitter suppresses (>20% drop)': (diff('sugar_100', 'sugar_bitter_100') and b['bitter_ratio'] < 0.8) if act else None,
             'Ir94e suppresses (>20% drop)': (diff('sugar_100', 'sugar_ir94e_100') and b['ir94e_ratio'] < 0.8) if act else None,
             'JON-CE drives aBN1': ce,
             'JON-F does not (<20% of CE)': (b['aBN1_JON_F'] < 0.2 * b['aBN1_JON_CE']) if ce else None}
    return b, calls


def score_calls(calls, real_calls):
    """Reproduced = same call as the real brain; not testable counts as not reproduced."""
    return {'n_reproduced': int(sum(calls[k] is not None and calls[k] == real_calls[k] for k in real_calls)),
            'n_testable': int(sum(v is not None for v in calls.values())), 'n_calls': len(real_calls)}


def stim_drive(pack):
    """Total |synapses| out of each battery stimulus set in a pack (regrowth keeps these exact by design)."""
    d = PACKS / pack
    ids = np.load(d / 'neuron_ids.npy')
    rp = np.load(d / 'row_ptr.npy').astype(np.int64)
    sc = np.abs(np.load(d / 'signed_counts.npy', mmap_mode='r'))
    pos = {int(r): i for i, r in enumerate(ids)}
    out = {}
    for k, (nids, _) in CONDITIONS.items():
        rows = [pos[int(i)] for i in set(nids)]
        out[k] = int(sum(int(sc[rp[r]:rp[r + 1]].sum()) for r in rows))
    return out


def _r(a, b, responding_only=True):
    """Similarity of two mean response vectors: the correlation over the neurons that respond in either. None (not
    scored) if both are silent or fewer than 3 neurons respond in either, since a correlation needs more; 0 if exactly
    one is silent. With responding_only=False, over all neurons (shared silence
    included)."""
    if a.max() == 0 and b.max() == 0:
        return None
    if a.max() == 0 or b.max() == 0:
        return 0.0
    u = (a > 0) | (b > 0) if responding_only else np.ones(len(a), bool)
    if u.sum() < 3:
        return None
    if a[u].std() == 0 or b[u].std() == 0:
        return 0.0
    return float(np.corrcoef(a[u], b[u])[0, 1])


def _pool(X, groups):
    """Sum spike counts within column groups (X holds sqrt counts) and return sqrt of the sums."""
    _, g = np.unique(groups, return_inverse=True)
    S = np.zeros((len(g), g.max() + 1), np.float32)
    S[np.arange(len(g)), g] = 1
    return np.sqrt((X ** 2) @ S)


def fidelity(ref, oth, cols=None, groups=None):
    """Per-probe similarity of sqrt mean DN vectors (responding DNs only, as in analyze.py) and transfer decoding
    (centroids from ref, single trials from oth). cols: restrict to these DN columns; groups: pool the columns within
    each group label first (e.g. the left and right cell of a DN type)."""
    if cols is not None:
        ref, oth = ref[..., cols], oth[..., cols]
    if groups is not None:
        ref, oth = _pool(ref, groups), _pool(oth, groups)
    R, O = ref.mean(1), oth.mean(1)
    per, per_all = {}, {}
    for p in range(len(R)):
        r = _r(R[p], O[p])
        if r is None:
            continue
        per[p], per_all[p] = r, _r(R[p], O[p], False)
    q = oth.reshape(-1, oth.shape[2])
    true = np.repeat(np.arange(len(R)), oth.shape[1])
    d2 = (q ** 2).sum(1)[:, None] + (R ** 2).sum(1)[None, :] - 2 * q @ R.T
    acc = float((d2.argmin(1) == true).mean())
    med = lambda d: float(np.median(list(d.values()))) if d else math.nan
    return {'median_r': med(per), 'median_r_incl_shared_silence': med(per_all), 'n_probes_compared': len(per),
            'transfer_decoding_acc': acc, 'chance': 1 / len(R), 'per_probe_r': {int(k): v for k, v in per.items()}}


MODALITY = [('taste', ('LB',)), ('mechanosensory (JO, bristles)', ('JO-', 'BM_')), ('olfactory', ('ORN_',)),
            ('thermo/hygro', ('TRN_', 'HRN_')), ('visual projection', ('LC', 'LPLC'))]


def modality(sid):
    t = sid.split(':')[1]
    for name, pre in MODALITY:
        if t.startswith(pre):
            return name
    return 'other'


def by_modality(per_probe_r, probes):
    out = {}
    for p, r in per_probe_r.items():
        out.setdefault(modality(str(probes[int(p)])), []).append(r)
    return {k: {'median_r': float(np.median(v)), 'n': len(v)} for k, v in out.items()}


def dn_pairs(dn_ids):
    """(left, right) column pairs of DN types with exactly one neuron per side."""
    a = annotations().set_index('root_id').reindex(dn_ids)
    ct, side = a.cell_type.astype(str).values, a.side.astype(str).values
    return np.array([(np.flatnonzero((ct == t) & (side == 'left'))[0], np.flatnonzero((ct == t) & (side == 'right'))[0])
                     for t in np.unique(ct)
                     if ((ct == t) & (side == 'left')).sum() == 1 and ((ct == t) & (side == 'right')).sum() == 1])


def mirror_floor(ref, probes, dn_ids, n_boot=2000, seed=0):
    """Left- vs right-hemisphere version of the same probe in the ipsi/contra frame (real brain), scored like
    fidelity() (responding DNs only; a pair silent on one side scores 0). Bootstrap CI over probe pairs."""
    pairs = dn_pairs(dn_ids)
    sid = list(probes)
    rs, rs_all = [], []
    for i, s in enumerate(sid):
        if not s.endswith(':left') or s.replace(':left', ':right') not in sid:
            continue
        j = sid.index(s.replace(':left', ':right'))
        L, R = ref[i].mean(0), ref[j].mean(0)
        Lf = np.r_[L[pairs[:, 0]], L[pairs[:, 1]]]
        Rf = np.r_[R[pairs[:, 1]], R[pairs[:, 0]]]
        r = _r(Lf, Rf)
        if r is None:
            continue
        rs.append(r)
        rs_all.append(_r(Lf, Rf, False))
    rs = np.array(rs)
    rng = np.random.default_rng(seed)
    boot = [np.median(rng.choice(rs, len(rs))) for _ in range(n_boot)] if len(rs) else [math.nan]
    return {'median_r': float(np.median(rs)) if len(rs) else math.nan, 'n_type_pairs': int(len(rs)),
            'ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'median_r_incl_shared_silence': float(np.median(rs_all)) if rs_all else math.nan}


def _gamma(x):
    """Elias-gamma code length in bits for positive integers."""
    return 2 * np.floor(np.log2(np.asarray(x, dtype=np.float64))) + 1


def _base():
    comp = pd.read_csv(BASE / 'Completeness_783.csv', index_col=0)
    con = pd.read_parquet(BASE / 'Connectivity_783.parquet', columns=['Presynaptic_Index', 'Postsynaptic_Index',
                                                                      'Connectivity', 'Excitatory'])
    n = len(comp)
    sign = np.ones(n, dtype=np.int8)
    sign[con.Presynaptic_Index.values] = np.sign(con.Excitatory.values).astype(np.int8)
    return comp.index.values, con, sign


def description_exact(level, base=None):
    """Bits of an explicit, decodable recipe for exactly what regrow.py keeps at this level.

    group of every neuron (entropy code) + which group pairs connect (log2 C(T^2, B)) + edge count of every block +
    the weight multiset of every block (each distinct count and its multiplicity, Elias-gamma) + every neuron's in- and
    out-degree + one sign bit per group. 'neuron' = the full wiring in the same code (topology + every weight + signs).
    """
    ids, con, sign = base or _base()
    n, pre, post = len(ids), con.Presynaptic_Index.values.astype(np.int64), con.Postsynaptic_Index.values.astype(np.int64)
    w = con.Connectivity.values.astype(np.int64)
    E = len(w)
    if level == 'neuron':
        topo = E * (math.log2(n * n / E) + 1.443)
        bits = {'topology': topo, 'weights': float(_gamma(w).sum()), 'signs': float(n)}
        return {'level': level, 'groups': n, 'nonzero_blocks': E, 'bits': bits, 'MB': sum(bits.values()) / 8e6}
    code, T = group_labels(level, ids, sign)
    blk = code[pre] * T + code[post]
    ub, m_b = np.unique(blk, return_counts=True)
    B = len(ub)
    _, n_bv = np.unique(blk * (int(w.max()) + 1) + w, return_counts=True)       # multiplicity of each (block, count)
    vals = np.unique(blk * (int(w.max()) + 1) + w) % (int(w.max()) + 1)
    sizes = np.bincount(code, minlength=T)
    kin, kout = np.bincount(post, minlength=n), np.bincount(pre, minlength=n)
    bits = {'group_labels': float((sizes[sizes > 0] * np.log2(n / sizes[sizes > 0])).sum()),
            'which_blocks': B * (math.log2(T * T / B) + 1.443),
            'block_edge_counts': float(_gamma(m_b).sum()),
            'block_weight_multisets': float(_gamma(vals).sum() + _gamma(n_bv).sum()),
            'degrees': float(_gamma(kin + 1).sum() + _gamma(kout + 1).sum()),
            'signs': float(T)}
    return {'level': level, 'groups': int(T), 'nonzero_blocks': int(B), 'bits': bits, 'MB': sum(bits.values()) / 8e6}


def description_wshuffle(level, base=None):
    """Exact topology and signs, plus the weight multiset of every group-pair block."""
    ids, con, sign = base or _base()
    n, E = len(ids), len(con)
    d = description_exact(level, (ids, con, sign))
    bits = {'topology': E * (math.log2(n * n / E) + 1.443), 'group_labels': d['bits']['group_labels'],
            'block_weight_multisets': d['bits']['block_weight_multisets'], 'signs': float(n)}
    return {'level': 'wshuffle_' + level, 'groups': d['groups'], 'nonzero_blocks': d['nonzero_blocks'], 'bits': bits,
            'MB': sum(bits.values()) / 8e6}


def readout_groups(level, base=None):
    """How the readout neurons are grouped at a level: DN identity, and the group sizes of aBN1 and the MN9 cells."""
    from stimsets import ABN1, MN9C, MN9I
    ids, con, sign = base or _base()
    a = annotations().set_index('root_id').reindex(ids)
    sc, ct = a.super_class.astype(str).values, a.cell_type.astype(str).values
    code, T = group_labels(level.replace('wshuffle_', ''), ids, sign)
    size = np.bincount(code, minlength=T)
    dn = sc == 'descending'
    g = np.unique(code[dn])
    per = pd.Series(ct[dn]).groupby(code[dn]).nunique()
    mixed = np.zeros(T, bool)
    mixed[code[~dn]] = True
    return {'DN_groups': int(len(g)), 'DN_types_per_group_median': float(per.median()), 'DN_types_per_group_max': int(per.max()),
            'frac_DN_groups_with_non_DNs': float(mixed[g].mean()),
            'aBN1_group_size': int(size[code[ids == ABN1][0]]),
            'MN9_group_sizes': sorted({int(size[code[ids == m][0]]) for m in (MN9C, MN9I)})}


def main():
    jr, ref, probes, dn_ids = load(REAL)
    out = {'real': {}, 'levels': {}}
    b, real_calls = behaviours(jr)
    out['real'] = {'behaviours': b, 'calls': real_calls, 'drive': stim_drive(REAL), 'probe_median_active': jr['probe_median_active']}
    pc = np.sort(dn_pairs(dn_ids).ravel())                         # the DNs the hemisphere yardstick uses
    an = annotations().set_index('root_id').reindex(dn_ids)
    dn_type, dn_side = an.cell_type.astype(str).values, an.side.astype(str).values
    multi = np.setdiff1d(np.arange(len(dn_ids)), pc)                # DNs of types with several cells per side

    def variants(oth):
        return {'paired_sides_summed': fidelity(ref, oth, pc, dn_type[pc]),
                'multi_DNs_per_neuron': fidelity(ref, oth, multi),
                'multi_DNs_type_x_side_summed': fidelity(ref, oth, multi, np.char.add(np.char.add(dn_type[multi], '|'), dn_side[multi]))}
    if (BAT / f'{REP}.json').is_file():
        jp, rep, _, _ = load(REP)
        out['noise_ceiling'] = fidelity(ref, rep)
        out['noise_ceiling_paired_DNs'] = fidelity(ref, rep, pc)
        out['noise_ceiling_paired_DNs']['by_modality'] = by_modality(out['noise_ceiling_paired_DNs']['per_probe_r'], probes)
        out['noise_ceiling_variants'] = variants(rep)
        out['noise_ceiling_calls'] = score_calls(behaviours(jp)[1], real_calls)
        out['noise_ceiling']['note'] = ('real brain vs itself with independent input noise; regrown brains are compared '
                                        'with common input noise, so a probe whose pathway is unchanged scores r = 1')
    out['hemisphere_floor'] = mirror_floor(ref, probes, dn_ids)
    if (BAT / 'v783.json').is_file():
        jpub, pub, _, _ = load('v783')
        bp, cp = behaviours(jpub)
        out['published_wiring'] = {'fidelity': fidelity(ref, pub), 'fidelity_paired_DNs': fidelity(ref, pub, pc), 'fidelity_variants': variants(pub),
                                   'calls': cp, 'behaviours': bp, 'score': score_calls(cp, real_calls),
                                   'probe_median_active': jpub['probe_median_active']}
    base = _base()
    neuron = description_exact('neuron', base)
    out['neuron_level'] = neuron
    for lv in LEVELS + WSHUFFLE:
        rows = []
        for sd in (0, 1, 2):
            tag = f'regrow_{lv}_s{sd}'
            if not (BAT / f'{tag}.json').is_file():
                continue
            j, oth, _, _ = load(tag)
            b, calls = behaviours(j)
            st = json.loads((ROOT / 'data' / 'wiring' / 'regrow' / f'{lv}_s{sd}' / 'regrow.json').read_text())
            drv = stim_drive(tag)
            fpd = fidelity(ref, oth, pc)
            fpd['by_modality'] = by_modality(fpd['per_probe_r'], probes)
            rows.append({'seed': sd, 'fidelity': fidelity(ref, oth), 'fidelity_paired_DNs': fpd, 'fidelity_variants': variants(oth),
                         'behaviours': b, 'calls': calls, **score_calls(calls, real_calls),
                         'drive_ratio_vs_real': {k: drv[k] / out['real']['drive'][k] for k in drv},
                         'frac_edges_also_in_real': st['frac_edges_also_in_real'],
                         'probe_median_active': j['probe_median_active'], 'regrow_stats': st})
        if rows:
            desc = description_wshuffle(lv.replace('wshuffle_', ''), base) if lv in WSHUFFLE else description_exact(lv, base)
            out['levels'][lv] = {'description': desc, 'brains': rows, 'readout_groups': readout_groups(lv, base),
                                 'series': 'ladder' if lv in LADDER else ('weights_shuffled' if lv in WSHUFFLE else
                                                                         ('readouts_kept' if lv.endswith('_keep_readouts') else
                                                                          ('fallback_supertype' if lv == 'hemilineage_else_supertype' else 'sides_pooled')))}
            f = [r['fidelity_paired_DNs']['median_r'] for r in rows]
            print(f"{lv:22s} groups {desc['groups']:6d} rules {desc['MB']:.2f} MB  fidelity r {np.mean(f):.3f} "
                  f"transfer {np.mean([r['fidelity']['transfer_decoding_acc'] for r in rows]):.2f}  reproduced "
                  f"{[r['n_reproduced'] for r in rows]} of {len(real_calls)} (testable {[r['n_testable'] for r in rows]})  "
                  f"edges kept {np.mean([r['frac_edges_also_in_real'] for r in rows]):.2f}  "
                  f"min drive ratio {min(min(r['drive_ratio_vs_real'].values()) for r in rows):.3f}", flush=True)
    out['neuron_level_MB'], out['genome_MB'] = neuron['MB'], GENOME_MB
    out['n_probes'] = int(len(probes))
    out['total_synapses'] = int(base[1].Connectivity.sum())
    (ROOT / 'results' / 'expression' / 'compression.json').write_text(json.dumps(out, indent=1, default=float))
    print('real calls:', out['real']['calls'])
    brief = lambda d: {k: v for k, v in (d or {}).items() if k in ('median_r', 'transfer_decoding_acc', 'n_probes_compared')}
    print('noise ceiling:', brief(out.get('noise_ceiling_paired_DNs')), '| hemisphere floor:', out['hemisphere_floor'])
    if 'published_wiring' in out:
        print('published wiring:', brief(out['published_wiring']['fidelity_paired_DNs']), out['published_wiring']['score'])


if __name__ == '__main__':
    main()
