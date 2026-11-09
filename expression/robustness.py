"""Robustness checks behind the main DN-census claims (part 2).

1. Is the low participation ratio of the DN output specific to the neck? DNs vs non-DN neurons matched in response
   breadth (number of stimulus x rate conditions that drive a neuron at >= 1 Hz) and peak rate.
2. The ignition family (analyze.ignition_flags): which stimuli are ignited from the lowest rate, the per-trial
   all-or-none onset of the others, and how many of the 18 are still told apart.
3. Silence vs drive size: silent share by within-class tercile of the driven cells' output synapses, and how much of
   the brain the DN-silent stimuli activate at all.
4. Published vs corrected wiring on the same stimuli and trials: silent share by class, and the runaway by class.
5. Information estimators: plug-in MI, the Panzeri-Treves term, the partition estimate at stricter criteria, and the
   plug-in value extrapolated from half the trials (same predictions subsampled; bias ~ 1/trials).
6. DN hubs: cell types per DN (sides merged), with behaviour labels where published.
7. Hemispheres: the same-side split-half floor of the left/right comparison, and the denominators of the medians.
8. Dose shapes: where the largest step of the switch-like pairs falls.
9. Braun sets: recruits inside and outside the GNG, and the 0.5 Hz criterion by drive rate.
10. Dictionary examples on both sides, with the share of DN output on behaviour-labelled DNs.
11. Stimulus types whose driven cells' output synapses differ most between the two sides.
12. Bilateral drive (census.py --bilateral): is a cell type that is DN-silent from either side alone still silent when
    both sides are driven together, and at the same total input (both sides at r vs one side at 2r)?

    .venv-mlx/bin/python expression/robustness.py v783_alfix_all
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import CEN, OUT, PACKS, decode, dn_pairs, ignition_flags, load, mutual_info, pt_bias, step_index  # noqa: E402
from dictionary import KNOWN  # noqa: E402
from rank_saturation import participation_ratio, whole_brain_means  # noqa: E402
from wiring import annotations  # noqa: E402

IGNITED = 300            # active neurons in a trial: ignition-family trials are bimodal, none between 250 and 300
BREADTH_EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1e9])
PEAK_EDGES_HZ = np.array([0, 1, 2, 5, 10, 20, 40, 80, 160, 1e9])
EXAMPLES = ['vpn:LC4', 'vpn:LPLC2', 'vpn:LT81', 'vpn:MeTu4c', 'sensory:JO-A5', 'sensory:JO-ED1', 'vpn:LC16', 'vpn:LCe04']


def _cls(s):
    return s['cell_class'] if s['kind'] == 'sensory' else 'visual_projection'


def matched_pr(Xd, Xr, rng, by_peak=False, reps=20):
    """PR of non-DN sets drawn to match the DNs' breadth (and peak-rate) distribution, bin by bin."""
    def key(X):
        k = np.digitize((X >= 1).sum(0), BREADTH_EDGES)
        return k * 100 + np.digitize(X.max(0) ** 2, PEAK_EDGES_HZ) if by_peak else k
    kd, kr = key(Xd), key(Xr)
    prs, short = [], 0
    for _ in range(reps):
        pick = []
        for b in np.unique(kd):
            k, pool = int((kd == b).sum()), np.flatnonzero(kr == b)
            if len(pool) == 0:
                short += k
                continue
            if len(pool) < k:
                short += k - len(pool)
            pick += list(rng.choice(pool, k, replace=len(pool) < k))
        prs.append(participation_ratio(Xr[:, pick]))
    return {'mean': float(np.mean(prs)), 'sd': float(np.std(prs)), 'frac_unmatched': short / (reps * Xd.shape[1])}


def pr_control(variant, lib, doses, dn, dn_ids, flag):
    ids = np.load(PACKS / variant / 'neuron_ids.npy')
    W, cols = whole_brain_means(variant, lib)                     # [S*D, M] sqrt(mean count)
    is_dn = np.isin(ids[cols], dn_ids)
    D = len(doses)
    out = {}
    for tag, keep in (('all_stimuli', np.ones(len(lib), bool)), ('without_ignition', ~flag)):
        rng = np.random.default_rng(3)
        Wk = W[np.repeat(keep, D)]
        on = Wk.max(0) > 0
        Xd, Xr = Wk[:, on & is_dn], Wk[:, on & ~is_dn]
        bd, br = (Xd >= 1).sum(0), (Xr >= 1).sum(0)
        rand = [participation_ratio(Xr[:, rng.choice(Xr.shape[1], Xd.shape[1], replace=False)]) for _ in range(20)]
        out[tag] = {'n_conditions': int(Wk.shape[0]), 'n_DN': int(Xd.shape[1]), 'n_rest': int(Xr.shape[1]),
                    'pr_DN': participation_ratio(Xd),
                    'pr_rest_random': {'mean': float(np.mean(rand)), 'sd': float(np.std(rand))},
                    'pr_rest_breadth_matched': matched_pr(Xd, Xr, rng),
                    'pr_rest_breadth_peak_matched': matched_pr(Xd, Xr, rng, by_peak=True),
                    'breadth_DN_quartiles': np.percentile(bd, [25, 50, 75]).tolist(),
                    'breadth_rest_quartiles': np.percentile(br, [25, 50, 75]).tolist(),
                    'peak_hz_DN_median': float(np.median(Xd.max(0) ** 2)),
                    'peak_hz_rest_median': float(np.median(Xr.max(0) ** 2))}
    out['note'] = ('breadth = conditions with trial mean >= 1 Hz; matched sets are drawn bin by bin (log2 breadth bins, '
                   'and peak-rate bins) from the responding non-DN neurons, 20 draws')
    return out


def ignition(lib, doses, dn, nact, Y, dn_ids, flag, ann):
    mean = dn.mean(2)
    ndn = (mean >= 1).sum(2)                                      # [S, D]
    olf = np.array([s.get('cell_class') == 'olfactory' for s in lib])
    seed = olf & (ndn[:, 0] >= 30) & (ndn[:, 0] >= 0.75 * ndn.max(1))   # the rule inside analyze.ignition_flags
    di = list(doses).index(100)
    tpl = np.sqrt(mean[:, di])[seed].mean(0)
    a = ann.set_index('root_id').reindex(dn_ids)
    name = [f"{t}({str(s)[0]})" for t, s in zip(a.cell_type.astype(str), a.side.astype(str))]
    med = np.median(nact, 2)                                      # [S, D]
    ign = nact >= IGNITED                                         # [S, D, T]
    rows = []
    for i in np.flatnonzero(flag):
        frac = ign[i].mean(1)
        onset = next((f'{int(doses[d])}Hz' for d in range(len(doses)) if frac[d] >= 0.5), 'never')
        rows.append({'sid': lib[i]['sid'], 'seed': bool(seed[i]), 'onset': onset,
                     'frac_trials_ignited': frac.round(2).tolist(), 'median_active': med[i].tolist(),
                     'DNs_ge1Hz': ndn[i].tolist()})
    s7 = [r for r in rows if r['seed']]
    rest = [r for r in rows if not r['seed']]
    r100, conf = decode(Y[:, di])
    T = Y.shape[2]
    acc = np.diag(conf) / T
    f = np.flatnonzero(flag)
    err = conf[f].sum() - np.diag(conf)[f].sum()
    err_in = conf[np.ix_(f, f)].sum() - np.diag(conf)[f].sum()
    r18, _ = decode(Y[flag, di])
    r18_200, _ = decode(Y[flag, list(doses).index(200)])
    acc200 = np.diag(decode(Y[:, list(doses).index(200)])[1]) / T
    return {'n_flagged': int(flag.sum()), 'n_seeds': len(s7), 'seeds': [r['sid'] for r in s7],
            'seed_DNs_ge1Hz_at_25Hz_range': [min(r['DNs_ge1Hz'][0] for r in s7), max(r['DNs_ge1Hz'][0] for r in s7)],
            'seed_median_active_at_25Hz_range': [min(r['median_active'][0] for r in s7), max(r['median_active'][0] for r in s7)],
            'seed_median_active_at_200Hz_range': [min(r['median_active'][-1] for r in s7), max(r['median_active'][-1] for r in s7)],
            'others_onset_counts': {o: sum(r['onset'] == o for r in rest) for o in [f'{int(d)}Hz' for d in doses] + ['never']},
            'others_DNs_ge1Hz_at_25Hz_range': [min(r['DNs_ge1Hz'][0] for r in rest), max(r['DNs_ge1Hz'][0] for r in rest)] if rest else None,
            'template_top_DNs': [name[j] for j in np.argsort(-tpl)[:6]],
            'decoded_ge50pct_100Hz': int((acc[f] >= 0.5).sum()), 'decoded_10of10_100Hz': int((acc[f] == 1).sum()),
            'mean_acc_100Hz': float(acc[f].mean()), 'decoded_ge50pct_200Hz': int((acc200[f] >= 0.5).sum()),
            'frac_errors_within_family_100Hz': float(err_in / err) if err else math.nan,
            'family_only_decoding_bits_100Hz': r18['MI_plugin_bits'], 'family_only_max_bits': r18['max_bits'],
            'family_only_decoding_bits_200Hz': r18_200['MI_plugin_bits'],
            'trial_criterion': f'ignited trial = >= {IGNITED} active neurons; onset = lowest rate with >= half the trials ignited',
            'rows': rows}


def silence(lib, doses, dn, nact):
    S = len(lib)
    di = list(doses).index(100)
    silent = dn[:, di].reshape(S, -1).max(1) == 0
    med = np.median(nact[:, di], 1)
    syn = np.array([s['syn_out'] for s in lib], dtype=float)
    cls = np.array([_cls(s) for s in lib])
    terc = {}
    for c in ('olfactory', 'visual_projection', 'mechanosensory', 'gustatory'):
        idx = np.flatnonzero(cls == c)
        parts = np.array_split(idx[np.argsort(syn[idx], kind='stable')], 3)
        terc[c] = {'n': int(len(idx)), 'silent_by_tercile': [float(silent[p].mean()) for p in parts],
                   'syn_out_tercile_medians': [float(np.median(syn[p])) for p in parts]}
    allp = np.array_split(np.argsort(syn, kind='stable'), 3)
    return {'n_silent_100Hz': int(silent.sum()), 'silent_median_active_neurons': float(np.median(med[silent])),
            'silent_le20_active': int((med[silent] <= 20).sum()), 'silent_le5_active': int((med[silent] <= 5).sum()),
            'silent_ge100_active': int((med[silent] >= 100).sum()),
            'silent_ge100_sids': [lib[i]['sid'] for i in np.flatnonzero(silent & (med >= 100))],
            'by_class_tercile_of_output_synapses': terc,
            'all_stimuli_silent_by_tercile': [float(silent[p].mean()) for p in allp],
            'median_cells_and_syn_out': {c: [float(np.median([s['n_driven'] for s, k in zip(lib, cls) if k == c])),
                                             float(np.median(syn[cls == c]))] for c in ('olfactory', 'visual_projection')}}


def published(pub='v783_published100', ref='v783_alfix_all', trials=5):
    common = {s['sid'] for s in json.loads((CEN / pub / 'library.json').read_text())} & \
             {s['sid'] for s in json.loads((CEN / ref / 'library.json').read_text())}
    out = {'trials': trials, 'dose_Hz': 100}
    for v in (pub, ref):
        lib, doses, dn, nact, _ = load(v, sids=common, trials=trials)
        di = list(doses).index(100)
        S = len(lib)
        cls = np.array([_cls(s) for s in lib])
        silent = dn[:, di].reshape(S, -1).max(1) == 0
        med = np.median(nact[:, di], 1)
        run = med > 3000
        m = dn[:, di].mean(1)
        o = {'n': S, 'silent_by_class': {c: [float(silent[cls == c].mean()), int((cls == c).sum())] for c in sorted(set(cls))},
             'n_runaway': int(run.sum()),
             'runaway_by_class': {c: [int((run & (cls == c)).sum()), int((cls == c).sum())] for c in sorted(set(cls))}}
        if run.any():
            o['runaway_median_active'] = float(np.median(med[run]))
            o['runaway_DNs_ge1Hz_median_min_max'] = [float(np.median((m[run] >= 1).sum(1))), int((m[run] >= 1).sum(1).min()),
                                                     int((m[run] >= 1).sum(1).max())]
            o['runaway_DNs_any_spike_median'] = float(np.median((m[run] > 0).sum(1)))
            o['runaway_non_olfactory'] = [lib[i]['sid'] for i in np.flatnonzero(run & ~np.isin(cls, ['olfactory', 'thermosensory', 'hygrosensory']))]
            o['olfactory_not_runaway'] = [lib[i]['sid'] for i in np.flatnonzero(~run & (cls == 'olfactory'))]
        out[v] = o
    return out


def _pred(Yc):
    """The T-fold nearest-centroid predictions of analyze.decode, [C, T]."""
    C, T, F = Yc.shape
    S = Yc.sum(1)
    pred = np.zeros((C, T), dtype=np.int64)
    for k in range(T):
        cen = (S - Yc[:, k]) / (T - 1)
        q = Yc[:, k]
        pred[:, k] = ((q ** 2).sum(1)[:, None] + (cen ** 2).sum(1)[None, :] - 2 * q @ cen.T).argmin(1)
    return pred


def estimators(Y, doses):
    def part(conf, Yc, crit):
        C, T = conf.shape[0], Yc.shape[1]
        silent = Yc.reshape(C, -1).max(1) == 0
        per = np.diag(conf) / T
        ok = ~silent & (per >= crit)
        b = np.array([1.0] * int(ok.sum()) + [float(silent.sum()), float((~silent & ~ok).sum())])
        b = b[b > 0]
        return float((b / C * np.log2(C / b)).sum())

    def one(Yc, reps=20):
        r, conf = decode(Yc)
        C, T = Yc.shape[0], Yc.shape[1]
        pred = _pred(Yc)
        rng = np.random.default_rng(4)
        half = []
        for _ in range(reps):                                     # same predictions, half the trials of every class
            keep = np.argsort(rng.random((C, T)), 1)[:, :T // 2]
            c2 = np.zeros((C, C))
            np.add.at(c2, (np.repeat(np.arange(C), T // 2), np.take_along_axis(pred, keep, 1).ravel()), 1)
            half.append(mutual_info(c2))
        mi = r['MI_plugin_bits']
        return {'plugin': mi, 'pt_term_subtracted': pt_bias(conf), 'plugin_half_trials': float(np.mean(half)),
                'plugin_extrapolated': float(2 * mi - np.mean(half)), 'max_bits': r['max_bits'],
                'partition': {f'ge{int(c * 100)}pct': part(conf, Yc, c) for c in (0.5, 0.8, 0.9, 1.0)}}
    S, D, T, F = Y.shape
    out = {f'{int(d)}Hz': one(Y[:, i]) for i, d in enumerate(doses) if int(d) in (100, 200)}
    out['stimulus_x_rate'] = one(Y.reshape(S * D, T, F))
    return out


def hubs(lib, doses, dn, dn_ids, ann, top=10):
    di = list(doses).index(100)
    m = dn[:, di].mean(1)                                         # [S, F]
    types = np.array([f"{s['kind']}:{s['cell_type']}" for s in lib])
    ut, ti = np.unique(types, return_inverse=True)
    a = ann.set_index('root_id').reindex(dn_ids)
    ct, sd = a.cell_type.astype(str).values, a.side.astype(str).values

    def count(thr):
        on = m >= thr
        n = np.zeros(m.shape[1], int)
        for j in range(len(ut)):
            n += on[ti == j].any(0)
        return n
    n1, n5 = count(1.0), count(5.0)
    order = np.argsort(-n1, kind='stable')[:top]
    return {'n_cell_types': int(len(ut)),
            'top': [{'DN': f'{ct[j]}({sd[j][0]})', 'cell_types_ge1Hz': int(n1[j]), 'cell_types_ge5Hz': int(n5[j]),
                     'behaviour': KNOWN.get(ct[j], (None,))[0]} for j in order]}


def mirror_floor(lib, doses, Y, dn_ids, ann):
    """Split-half (trials 0-4 vs 5-9) similarity of the same side, against the same statistic across sides."""
    di = list(doses).index(100)
    pairs = dn_pairs(ann, dn_ids)
    by = {}
    for i, s in enumerate(lib):
        by.setdefault((s['kind'], s['cell_type']), {})[s['side']] = i
    same_r, same_j, cross_r, cross_j = [], [], [], []

    def sim(a, b, u):
        r = float(np.corrcoef(a[u], b[u])[0, 1]) if u.sum() >= 3 and a[u].std() > 0 and b[u].std() > 0 else math.nan
        x, y = a >= 1, b >= 1
        return r, (float((x & y).sum() / (x | y).sum()) if (x | y).any() else math.nan)
    for d in by.values():
        if not {'left', 'right'} <= d.keys():
            continue
        L, R = Y[d['left'], di], Y[d['right'], di]
        Lf = np.concatenate([L[:, pairs[:, 0]], L[:, pairs[:, 1]]], 1)
        Rf = np.concatenate([R[:, pairs[:, 1]], R[:, pairs[:, 0]]], 1)
        if Lf.max() == 0 or Rf.max() == 0:
            continue                                               # both sides must respond for a floor
        h = Lf.shape[0] // 2
        u = (Lf.mean(0) > 0) | (Rf.mean(0) > 0)
        for X in (Lf, Rf):
            r, j = sim(X[:h].mean(0), X[h:2 * h].mean(0), u)
            same_r.append(r)
            same_j.append(j)
        for a, b in ((Lf[:h], Rf[h:2 * h]), (Lf[h:2 * h], Rf[:h])):
            r, j = sim(a.mean(0), b.mean(0), u)
            cross_r.append(r)
            cross_j.append(j)
    rows = json.loads((OUT / 'mirror_rows_v783_alfix_all.json').read_text())
    nc = np.array([r['mirror_r_noise_corrected'] for r in rows], dtype=float)
    raw = np.array([r['mirror_r'] for r in rows], dtype=float)
    ok = np.isfinite(nc)
    sm = np.array([r['size_matched'] for r in rows])
    return {'n_types_both_sides_respond': len(same_r) // 2,
            'same_side_split_half_median_r': float(np.nanmedian(same_r)), 'same_side_split_half_median_jaccard': float(np.nanmedian(same_j)),
            'cross_side_half_median_r': float(np.nanmedian(cross_r)), 'cross_side_half_median_jaccard': float(np.nanmedian(cross_j)),
            'all_types': {'n': len(rows), 'n_reliability_defined': int(ok.sum()), 'raw_median_r_all': float(np.median(raw)),
                          'raw_median_r_defined': float(np.median(raw[ok])), 'corrected_median_r_defined': float(np.median(nc[ok]))},
            'size_matched': {'n': int(sm.sum()), 'n_reliability_defined': int((ok & sm).sum()),
                             'raw_median_r_all': float(np.median(raw[sm])), 'raw_median_r_defined': float(np.median(raw[ok & sm])),
                             'corrected_median_r_defined': float(np.median(nc[ok & sm]))}}


def dose_steps(mean, peak_hz=5.0, active_hz=1.0):
    where = np.zeros(mean.shape[1], int)                          # largest step: rest->d1, d1->d2, ...
    for s in range(mean.shape[0]):
        for f in np.flatnonzero(mean[s].max(0) >= peak_hz):
            m = mean[s, :, f]
            if (m >= active_hz).sum() < 2 or not step_index(m) >= 0.8:
                continue
            where[int(np.argmax(np.diff(np.concatenate([[0.0], m]))))] += 1
    return {'n_switch_like': int(where.sum()), 'largest_step_counts': where.tolist(),
            'frac_largest_step_from_rest': float(where[0] / where.sum()) if where.sum() else math.nan}


def braun_detail(ann, dn_ids):
    import pandas as pd
    from braun import SETS, select
    rows = json.loads((OUT / 'braun_recruitment_v783_alfix_all.json').read_text())
    t = ann.set_index('root_id').reindex(dn_ids).cell_type.astype(str).values
    out = {'n_DN_gng': int(sum(len(x) > 2 and x[2] == 'g' for x in t)), 'n_DN': int(len(t)), 'sets': {}}
    con = pd.read_parquet(OUT.parents[1] / 'external/Drosophila_brain_model/Connectivity_783.parquet',
                          columns=['Presynaptic_ID', 'Postsynaptic_ID', 'Connectivity'])
    dset = set(int(i) for i in dn_ids)
    out['strongest_edge_to_a_DN_partner'] = {}
    for name, sel, _, _ in SETS:
        if name in ('DNa01', 'DNb02'):
            ids = set(select(ann, sel))
            e = con[con.Presynaptic_ID.isin(ids) & con.Postsynaptic_ID.isin(dset) & ~con.Postsynaptic_ID.isin(ids)]
            out['strongest_edge_to_a_DN_partner'][name] = {'max_synapses': int(e.Connectivity.max()),
                                                          'partners_ge5': int(e[e.Connectivity >= 5].Postsynaptic_ID.nunique())}
    for r in rows:
        if r['set'] in ('DNp09', 'aDN2', 'MDN'):
            out['sets'][r['set']] = {'all_200Hz': r['recruited_200Hz'], 'gng_200Hz': r['recruited_gng_200Hz'],
                                     'non_gng_200Hz': r['recruited_200Hz'] - r['recruited_gng_200Hz'],
                                     'gng_thr0.5_by_rate': [r[f'recruited_gng_{d}Hz_thr0.5'] for d in (25, 50, 100, 200)]}
    return out


def _labelled(m, ct, sd):
    """Rate of the strongest cell of each behaviour-labelled DN type and side (>= 1 Hz), strongest first."""
    out = {}
    for j in np.argsort(-m):
        if m[j] >= 1 and ct[j] in KNOWN:
            out.setdefault(f'{ct[j]}({sd[j][0]})', round(float(m[j]), 1))
    return out


def syn_asymmetry(min_ratio=3.5):
    """Stimulus types whose driven cells' output synapses differ most between the two sides (mirror rows)."""
    rows = json.loads((OUT / 'mirror_rows_v783_alfix_all.json').read_text())
    out = []
    for r in rows:
        a, b = r['syn_out_left'] or 0, r['syn_out_right'] or 0
        if min(a, b) > 0 and max(a, b) / min(a, b) >= min_ratio:
            out.append({'type': r['cell_type'], 'syn_out_left': a, 'syn_out_right': b, 'ratio': round(max(a, b) / min(a, b), 1),
                        'weaker': 'left' if a < b else 'right'})
    return sorted(out, key=lambda x: -x['ratio'])


def examples(lib, doses, dn, dn_ids, ann):
    di = list(doses).index(100)
    a = ann.set_index('root_id').reindex(dn_ids)
    ct, sd = a.cell_type.astype(str).values, a.side.astype(str).values
    pos = {s['sid']: i for i, s in enumerate(lib)}
    d = json.loads((OUT / 'dictionary_v783_alfix_all.json').read_text())
    by = {r['stimulus']: r for r in d['rows']}
    out = {}
    for t in EXAMPLES:
        for side in ('left', 'right'):
            r = by.get(f'{t}:{side}')
            if r is None:
                continue
            tot = max(r['total_DN_rate_Hz'], 1e-9)
            out[f'{t}:{side}'] = {'n_DN': r['n_DN_responding'], 'total_Hz': round(r['total_DN_rate_Hz'], 1),
                                  'top_DNs': r['top_DNs'][:3], 'labelled_share': round(r['known_behaviour_share'], 2),
                                  'behaviours': {k: round(v / tot, 2) for k, v in list(r['behaviours'].items())[:3]},
                                  'labelled_DNs_Hz': _labelled(dn[pos[f'{t}:{side}'], di].mean(0), ct, sd)}
    return out


def bilateral_check(variant='v783_alfix_all'):
    """Silence and capacity when each cell type is driven on both sides at once (needs the full bilateral census)."""
    bv = f'{variant}_bilateral'
    lib_b = json.loads((CEN / bv / 'library.json').read_text())
    if not all((CEN / bv / (x['sid'].replace(':', '__').replace('/', '_') + '.npz')).is_file() for x in lib_b):
        return None
    lib1, doses, dn1, _, _ = load(variant)
    libb, dosesb, dnb, nactb, _ = load(bv)
    assert list(doses) == list(dosesb)
    S1, Sb = len(lib1), len(libb)
    sil1 = dn1.reshape(S1, len(doses), -1).max(2) == 0             # [S, D]
    silb = dnb.reshape(Sb, len(doses), -1).max(2) == 0
    pos = {s['sid']: i for i, s in enumerate(lib1)}
    L = np.array([pos[f"{s['kind']}:{s['cell_type']}:left"] for s in libb])
    R = np.array([pos[f"{s['kind']}:{s['cell_type']}:right"] for s in libb])
    cls = np.array([_cls(s) for s in libb])
    both_alone = sil1[L] & sil1[R]                                  # silent from either side alone
    out = {'n_cell_types': Sb, 'doses': [int(d) for d in doses], 'by_dose': {}, 'by_class_100Hz': {}}
    for i, d in enumerate(doses):
        k = f'{int(d)}Hz'
        row = {'silent_bilateral': float(silb[:, i].mean()), 'silent_either_side_alone': float(both_alone[:, i].mean()),
               'silent_one_sided_stimuli': float(sil1[:, i].mean()),
               'of_types_silent_from_either_side_still_silent_bilateral':
                   float((silb[:, i] & both_alone[:, i]).sum() / max(both_alone[:, i].sum(), 1)),
               'n_types_silent_from_either_side': int(both_alone[:, i].sum()),
               'median_DNs_ge1Hz_bilateral': float(np.median((dnb[:, i].mean(1) >= 1).sum(1))),
               'median_active_bilateral': float(np.median(np.median(nactb[:, i], 1)))}
        j = list(doses).index(2 * d) if 2 * d in doses else None
        if j is not None:                                           # same total input: both sides at r, one side at 2r
            row['silent_one_side_at_double_rate'] = float(sil1[np.r_[L, R], j].mean())
        r, _ = decode(np.sqrt(dnb[:, i]))
        row['decode'] = {x: r[x] for x in ('accuracy_non_silent', 'distinguishable_patterns', 'MI_plugin_bits',
                                           'partition_bits', 'max_bits', 'frac_silent')}
        out['by_dose'][k] = row
    i100 = list(doses).index(100)
    for c in sorted(set(cls), key=lambda c: -(cls == c).sum()):
        m = cls == c
        out['by_class_100Hz'][c] = {'n': int(m.sum()), 'silent_bilateral': float(silb[m, i100].mean()),
                                    'silent_either_side_alone': float(both_alone[m, i100].mean()),
                                    'silent_one_sided_stimuli': float(np.r_[sil1[L[m], i100], sil1[R[m], i100]].mean())}
    return out


def main(variant='v783_alfix_all'):
    ann = annotations()
    lib, doses, dn, nact, dn_ids = load(variant)
    Y = np.sqrt(dn)
    flag, _ = ignition_flags(lib, nact, doses, dn)
    res = {'variant': variant,
           'ignition': ignition(lib, doses, dn, nact, Y, dn_ids, flag, ann),
           'silence': silence(lib, doses, dn, nact),
           'published_vs_corrected_by_class': published(),
           'estimators': estimators(Y, doses),
           'hubs_100Hz': hubs(lib, doses, dn, dn_ids, ann),
           'mirror_floor_100Hz': mirror_floor(lib, doses, Y, dn_ids, ann),
           'dose_steps': dose_steps(dn.mean(2)),
           'braun': braun_detail(ann, dn_ids),
           'dictionary_examples_100Hz': examples(lib, doses, dn, dn_ids, ann),
           'syn_asymmetry_ge3.5x': syn_asymmetry()}
    print(json.dumps({k: v for k, v in res.items() if k != 'ignition'}, indent=1, default=float)[:6000], flush=True)
    bl = bilateral_check(variant) if (CEN / f'{variant}_bilateral' / 'library.json').is_file() else None
    if bl:
        res['bilateral'] = bl
    res['participation_ratio_control'] = pr_control(variant, lib, doses, dn, dn_ids, flag)
    (OUT / f'robustness_{variant}.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res['participation_ratio_control'], indent=1))
    print(json.dumps({k: v for k, v in res['ignition'].items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['v783_alfix_all']))
