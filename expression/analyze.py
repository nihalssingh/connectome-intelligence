"""DN census analysis (part 2): rank, vocabulary, capacity, nuance and hemispheric 'voice' of the descending-neuron channel.

Input: the census (expression/census.py). Statistics use sqrt(spike counts) in 1 s (variance-stabilised), with trial
splits as noise floors. Rank and bits are lower bounds that grow with the size of the stimulus set
(expression/rank_saturation.py), not properties of the channel alone.

    .venv-mlx/bin/python expression/analyze.py v783_alfix_all
    .venv-mlx/bin/python expression/analyze.py v783_published100     # also compares with the corrected wiring
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wiring import annotations  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CEN = ROOT / 'results' / 'expression' / 'census'
OUT = ROOT / 'results' / 'expression'
PACKS = ROOT / 'external' / 'drosophila-brain-mlx' / 'data' / 'pack'
REFERENCE = 'v783_alfix_all'


def _fname(sid):
    return sid.replace(':', '__').replace('/', '_') + '.npz'


def out_strength(variant):
    """Total |synapses| each neuron sends (pack order), from the MLX pack's source-major CSR."""
    d = PACKS / variant
    rp = np.load(d / 'row_ptr.npy').astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(np.abs(np.load(d / 'signed_counts.npy')).astype(np.int64))])
    return cs[rp[1:]] - cs[rp[:-1]]


def load(variant, sids=None, trials=None):
    """Census of one wiring variant -> lib, doses, dn [S, D, T, nDN] counts, n_active [S, D, T], dn_ids.
    Unreadable files (e.g. a write interrupted by a kill) are reported and skipped. Each lib entry gains n_driven
    (neurons actually driven) and syn_out (their total output synapses)."""
    d = CEN / variant
    lib = json.loads((d / 'library.json').read_text())
    rd = np.load(d / 'readouts.npz')
    strength = out_strength(variant) if (PACKS / variant).is_dir() else None
    rows, keep, bad, doses = [], [], [], None
    for i, s in enumerate(lib):
        if sids is not None and s['sid'] not in sids:
            continue
        f = d / _fname(s['sid'])
        if not f.is_file():
            continue
        try:
            z = np.load(f, allow_pickle=True)
            dn, na, drv, doses_i = z['dn'], z['n_active'], z['driven'], z['doses']
        except Exception as e:  # noqa: BLE001
            bad.append(f'{s["sid"]}: {type(e).__name__}')
            continue
        if trials:
            dn, na = dn[:, :trials], na[:, :trials]
        rows.append((dn, na))
        keep.append(i)
        doses = doses_i
        s['n_driven'] = int(len(drv))
        s['syn_out'] = int(strength[drv].sum()) if strength is not None else None
    if bad:
        print('unreadable census files skipped:', bad, flush=True)
    lib = [lib[i] for i in keep]
    dn = np.stack([r[0] for r in rows]).astype(np.float32)        # [S, D, T, nDN]
    nact = np.stack([r[1] for r in rows])                          # [S, D, T]
    return lib, doses, dn, nact, rd['dn_ids']


def mutual_info(conf):
    p = conf / conf.sum()
    ps, pq = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (ps @ pq)[nz])).sum())


def pt_bias(conf):
    """Panzeri-Treves (1996) first-order bias of plug-in MI (bits), with naive counts of occupied response bins."""
    N = conf.sum()
    rs = (conf > 0).sum(1)
    r = (conf.sum(0) > 0).sum()
    return float(((rs - 1).sum() - (r - 1)) / (2 * N * math.log(2)))


def fano_bits(acc, C):
    """Fano lower bound on I(S; S_hat) for C equiprobable classes decoded with accuracy acc."""
    pe = 1 - acc
    if pe <= 0:
        return float(np.log2(C))
    h = -pe * np.log2(pe) - (1 - pe) * np.log2(1 - pe) if pe < 1 else 0.0
    return float(max(np.log2(C) - h - pe * np.log2(C - 1), 0.0))


def decode(Y, seed=0):
    """T-fold nearest-centroid decoding (trial k of every class held out at once). Y: [C, T, F].

    Reports accuracy, the Fano lower bound on bits (tight only if errors spread evenly), plug-in MI and its
    Panzeri-Treves bias-corrected value (still inflated when large groups of mutually confused classes are sampled with
    few trials, e.g. a runaway state), counts that do not depend on Fano (classes that never evoke a spike tie into one
    'silent' word; accuracy on the others; classes decoded on >= half the trials) and a partition estimate of bits."""
    C, T, F = Y.shape
    S = Y.sum(1)
    pred = np.zeros((C, T), dtype=np.int64)
    for k in range(T):
        cen = (S - Y[:, k]) / (T - 1)
        q = Y[:, k]
        d2 = (q ** 2).sum(1)[:, None] + (cen ** 2).sum(1)[None, :] - 2 * q @ cen.T
        pred[:, k] = d2.argmin(1)
    true = np.repeat(np.arange(C), T)
    conf = np.zeros((C, C))
    np.add.at(conf, (true, pred.ravel()), 1)
    correct = pred == np.arange(C)[:, None]
    acc = float(correct.mean())
    silent = Y.reshape(C, -1).max(1) == 0
    per_class = correct.mean(1)
    mi = mutual_info(conf)
    ok50 = ~silent & (per_class >= 0.5)
    n50 = int(ok50.sum())
    # partition estimate: each stimulus decoded on >= half its trials is its own word, all silent stimuli share one
    # word, and the remaining (mutually confused) stimuli share another. Robust where plug-in MI is undersampled.
    blocks = np.array([1.0] * n50 + [float(silent.sum()), float((~silent & ~ok50).sum())])
    blocks = blocks[blocks > 0]
    part = float((blocks / C * np.log2(C / blocks)).sum())
    return {'n_classes': C, 'accuracy': acc, 'chance': 1 / C, 'fano_bits_lower_bound': fano_bits(acc, C),
            'partition_bits': part,
            'vocabulary_lower_bound_fano': float(2 ** fano_bits(acc, C)),
            'MI_plugin_bits': mi, 'MI_pt_corrected_bits': mi - pt_bias(conf), 'max_bits': float(np.log2(C)),
            'n_silent_classes': int(silent.sum()), 'frac_silent': float(silent.mean()),
            'accuracy_non_silent': float(per_class[~silent].mean()) if (~silent).any() else math.nan,
            'n_decoded_ge50pct_non_silent': n50,
            'distinguishable_patterns': n50 + int(silent.any())}, conf


def cvpca(Y, n_null=200, seed=0):
    """Cross-validated PCA (Stringer et al. 2019) over conditions. Y: [C, T, F].
    Symmetrised over the two trial halves; the null permutes conditions of one half (applied to the projections)."""
    T = Y.shape[1]
    A, B = Y[:, : T // 2].mean(1), Y[:, T // 2: 2 * (T // 2)].mean(1)
    mu = (A.mean(0) + B.mean(0)) / 2
    A, B = A - mu, B - mu
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    _, _, vtb = np.linalg.svd(B, full_matrices=False)
    PA, PB = A @ vt.T, B @ vt.T                                    # PCs of half A
    QB, QA = B @ vtb.T, A @ vtb.T                                  # PCs of half B
    cv = ((PA * PB).mean(0) + (QB * QA).mean(0)) / 2
    rng = np.random.default_rng(seed)
    null = np.empty((n_null, len(cv)))
    for i in range(n_null):
        p = rng.permutation(len(B))
        null[i] = ((PA * PB[p]).mean(0) + (QB[p] * QA).mean(0)) / 2
    thr = np.percentile(null, 95, axis=0)
    above = cv > thr
    rank = int(np.argmin(above)) if not above.all() else len(cv)  # leading run of components above null
    pos = np.clip(cv, 0, None)
    pr = float(pos.sum() ** 2 / (pos ** 2).sum()) if pos.sum() > 0 else 0.0
    tot = pos.sum()
    n90 = int(np.searchsorted(np.cumsum(pos) / tot, 0.9) + 1) if tot > 0 else 0
    return {'cv_variance': cv.tolist(), 'null95': thr.tolist(), 'rank_above_null': rank,
            'n_components_above_null_total': int(above.sum()),
            'participation_ratio': pr, 'n_components_90pct_reliable_variance': n90,
            'reliable_variance_fraction_of_total': float(tot / ((A ** 2).mean(0).sum() + 1e-12))}


def step_index(m):
    """m: [D] mean rates at increasing doses (baseline 0 before dose 1). Largest single step / total range.
    A response linear in dose over 4 doses scores 0.5; an all-or-none jump scores 1."""
    r = np.concatenate([[0.0], m])
    d = np.diff(r)
    rng = r.max() - r.min()
    return float(d.max() / rng) if rng > 0 else np.nan


def nuance_shapes(mean, peak_hz=5.0, active_hz=1.0):
    """Dose-response shape of every responsive (stimulus, DN) pair; mean: [S, D, F] Hz.
    'top-dose only' pairs (active only at the highest drive) cannot show whether they plateau, so they are their
    own category; shapes are classified only for pairs active at >= 2 doses."""
    cats = {'top_dose_only': 0, 'other_single_dose': 0, 'switch_like': 0, 'graded': 0, 'intermediate': 0}
    steps = []
    for s in range(mean.shape[0]):
        for f in np.flatnonzero(mean[s].max(0) >= peak_hz):
            m = mean[s, :, f]
            act = m >= active_hz
            if act.sum() < 2:
                cats['top_dose_only' if act[-1] else 'other_single_dose'] += 1
                continue
            si = step_index(m)
            steps.append(si)
            cats['switch_like' if si >= 0.8 else ('graded' if si <= 0.5 else 'intermediate')] += 1
    n = sum(cats.values())
    return {'n_responsive_pairs': n, **{f'frac_{k}': v / n if n else math.nan for k, v in cats.items()},
            'n_shape_classified': len(steps),
            'frac_switch_like_of_classified': cats['switch_like'] / len(steps) if steps else math.nan,
            'frac_graded_of_classified': cats['graded'] / len(steps) if steps else math.nan,
            'step_index_median_classified': float(np.median(steps)) if steps else math.nan}, np.array(steps)


def _sb_reliability(X):
    """Spearman-Brown split-half reliability of the trial mean of X [T, F] (odd T: the last trial is dropped)."""
    h = X.shape[0] // 2
    a, b = X[:h].mean(0), X[h:2 * h].mean(0)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    r = np.corrcoef(a, b)[0, 1]
    return 2 * r / (1 + r) if np.isfinite(r) else np.nan


def dn_pairs(ann, dn_ids):
    a = ann.set_index('root_id').reindex(dn_ids)
    ct, side = a.cell_type.astype(str).values, a.side.astype(str).values
    pairs = []
    for t in np.unique(ct):
        l, r = np.flatnonzero((ct == t) & (side == 'left')), np.flatnonzero((ct == t) & (side == 'right'))
        if len(l) == 1 and len(r) == 1:
            pairs.append((l[0], r[0]))
    return np.array(pairs)


def mirror(lib, Y, dn_ids, ann, dose_i, n_perm=5000, seed=0):
    """Left- vs right-hemisphere 'voices': the same cell type driven on the left or on the right, compared in an
    ipsi/contra frame over the DN types with one neuron per side.

    Similarity is computed over the DNs that respond to either side (shared silence would inflate it); a type that is
    silent on one side scores 0 and is counted. Hemisphere decoding is T-fold (trial k of both sides held out), so
    chance is 0.5; the effect size is the distance between the two centroids over the trial-to-trial spread."""
    pairs = dn_pairs(ann, dn_ids)
    by = {}
    for i, s in enumerate(lib):
        by.setdefault((s['kind'], s['cell_type']), {})[s['side']] = i
    rows, diffs, groups, matched = [], [], [], []
    for (kind, t), d in by.items():
        if not {'left', 'right'} <= d.keys():
            continue
        L, R = Y[d['left'], dose_i], Y[d['right'], dose_i]          # [T, nDN] sqrt counts
        Lf = np.concatenate([L[:, pairs[:, 0]], L[:, pairs[:, 1]]], 1)   # [ipsi, contra] for a left stimulus
        Rf = np.concatenate([R[:, pairs[:, 1]], R[:, pairs[:, 0]]], 1)   # [ipsi, contra] for a right stimulus
        mL, mR = Lf.mean(0), Rf.mean(0)
        if mL.max() == 0 and mR.max() == 0:
            continue                                                 # silent on both sides: nothing to compare
        one_silent = mL.max() == 0 or mR.max() == 0
        u = (mL > 0) | (mR > 0)
        aL, aR = mL >= 1, mR >= 1                                    # active: mean sqrt count >= 1
        jac = float((aL & aR).sum() / (aL | aR).sum()) if (aL | aR).any() else math.nan
        if one_silent or u.sum() < 3 or mL[u].std() == 0 or mR[u].std() == 0:
            r, rl, rr = 0.0, np.nan, np.nan
        else:
            r = float(np.corrcoef(mL[u], mR[u])[0, 1])
            rl, rr = _sb_reliability(Lf[:, u]), _sb_reliability(Rf[:, u])
        rnc = 0.0 if one_silent else (r / math.sqrt(rl * rr) if rl > 0 and rr > 0 else np.nan)
        T = Lf.shape[0]
        ok = 0
        for k in range(T):                                           # T-fold: hold out trial k of both sides
            m = np.ones(T, bool)
            m[k] = False
            c0, c1 = Lf[m].mean(0), Rf[m].mean(0)
            ok += int(((Lf[k] - c0) ** 2).sum() < ((Lf[k] - c1) ** 2).sum())
            ok += int(((Rf[k] - c1) ** 2).sum() < ((Rf[k] - c0) ** 2).sum())
        within = math.sqrt((((Lf - mL) ** 2).sum(1).mean() + ((Rf - mR) ** 2).sum(1).mean()) / 2)
        between = float(np.sqrt(((mL - mR) ** 2).sum()))
        sL, sR = lib[d['left']], lib[d['right']]
        nl, nr = sL.get('n_driven', sL.get('n', 0)), sR.get('n_driven', sR.get('n', 0))
        yl, yr = sL.get('syn_out'), sR.get('syn_out')
        count_matched = abs(nl - nr) <= 0.1 * max(nl, nr)
        syn_matched = yl is not None and yr is not None and abs(yl - yr) <= 0.2 * max(yl, yr, 1)
        rows.append({'kind': kind, 'cell_type': t, 'cell_class': sL.get('cell_class', ''), 'n_left': nl, 'n_right': nr,
                     'syn_out_left': yl, 'syn_out_right': yr, 'count_matched': bool(count_matched),
                     'size_matched': bool(count_matched and syn_matched), 'one_side_silent': bool(one_silent),
                     'n_dn_features_responding': int(u.sum()), 'jaccard_active': jac,
                     'mirror_r': r, 'rel_left': float(rl), 'rel_right': float(rr), 'mirror_r_noise_corrected': float(rnc),
                     'hemisphere_decoding_acc': ok / (2 * T),
                     'hemisphere_effect_size': between / within if within > 0 else math.inf,
                     'dn_spikes_left_stim': float((L ** 2).sum(1).mean()), 'dn_spikes_right_stim': float((R ** 2).sum(1).mean())})
        dv = mL - mR
        if np.linalg.norm(dv) > 0:
            diffs.append(dv / np.linalg.norm(dv))                    # pattern only, not loudness
            groups.append('visual_projection' if kind == 'vpn' else (sL.get('cell_class') or 'unknown'))   # stimulus family
            matched.append(bool(count_matched and syn_matched))
    # one consistent 'hemisphere voice'? leave-one-type-out sign test vs a sign-flip null grouped by cell class
    def voice(D):
        tot = D.sum(0)
        return float(np.mean([(D[j] @ (tot - D[j])) > 0 for j in range(len(D))]))
    out = {'n_types': len(rows), 'n_dn_pairs': int(len(pairs))}

    def voice_test(D, grp, tag):
        if len(D) <= 2:
            return {}
        obs = voice(D)
        g = np.unique(grp, return_inverse=True)[1]
        rng = np.random.default_rng(seed)
        null = [voice(D * rng.choice([-1, 1], g.max() + 1)[g][:, None]) for _ in range(n_perm)]
        return {f'consistent_voice_loo_acc{tag}': obs, f'consistent_voice_p_grouped_signflip{tag}':
                float((np.sum(np.array(null) >= obs) + 1) / (n_perm + 1)), f'n_groups{tag}': int(g.max() + 1),
                f'n_types_voice{tag}': len(D)}
    D, G, M = np.array(diffs), np.array(groups), np.array(matched)
    out.update(voice_test(D, G, ''))
    if M.any():
        out.update(voice_test(D[M], G[M], '_size_matched'))
    return rows, out


def _summ(rows):
    if not rows:
        return {'n_types': 0}
    f = lambda k: [r[k] for r in rows]
    nc = [x for x in f('mirror_r_noise_corrected') if np.isfinite(x)]
    return {'n_types': len(rows), 'n_one_side_silent': int(sum(f('one_side_silent'))),
            'median_mirror_r': float(np.median(f('mirror_r'))),
            'median_mirror_r_noise_corrected': float(np.median(nc)) if nc else math.nan,
            'n_noise_corrected': len(nc),
            'median_jaccard_active': float(np.nanmedian(f('jaccard_active'))),
            'median_hemisphere_decoding_acc': float(np.median(f('hemisphere_decoding_acc'))),
            'frac_types_hemisphere_decodable_gt_0.8': float(np.mean([x > 0.8 for x in f('hemisphere_decoding_acc')])),
            'median_hemisphere_effect_size': float(np.median(f('hemisphere_effect_size')))}


def ignition_flags(lib, nact, doses, dn=None, r_min=0.8):
    """Stimuli that tip into the dose-insensitive LAL/VES/CRE state some single glomeruli ignite ('ignition' in result
    keys; about 550-740 active neurons and 51-69 DNs already at 25 Hz, up to about 1,150 active at 200 Hz). Seeds:
    olfactory stimuli with >= 30 DNs at >= 1 Hz at the lowest rate and at least 75% of their maximum DN count there.
    Flagged: every stimulus whose DN pattern at 100 Hz (or the top rate) correlates at r >= r_min with the seeds' mean
    pattern. Without dn (single-rate census), the whole-brain runaway rule (> 3,000 active) is used."""
    if dn is None or len(doses) < 2:
        flag = np.median(nact, 2)[:, 0] > 3000
        return flag, [lib[i]['sid'] for i in np.flatnonzero(flag)]
    mean = dn.mean(2)                                                 # [S, D, F]
    ndn = (mean >= 1).sum(2)                                          # [S, D]
    olf = np.array([s.get('cell_class') == 'olfactory' for s in lib])
    seed = olf & (ndn[:, 0] >= 30) & (ndn[:, 0] >= 0.75 * ndn.max(1))
    flag = seed.copy()
    if seed.any():
        di = list(doses).index(100) if 100 in doses else len(doses) - 1
        X = np.sqrt(mean[:, di])
        tpl = X[seed].mean(0)
        ok = X.std(1) > 0
        r = np.zeros(len(X))
        r[ok] = [np.corrcoef(x, tpl)[0, 1] for x in X[ok]]
        flag |= r >= r_min
    return flag, [lib[i]['sid'] for i in np.flatnonzero(flag)]


def compare_variants(a, b=REFERENCE, dose=100.0, trials=5):
    """Capacity of two wiring variants on their common stimuli, same dose and trial count."""
    la = {s['sid'] for s in json.loads((CEN / a / 'library.json').read_text())
          if (CEN / a / _fname(s['sid'])).is_file()}
    lb = {s['sid'] for s in json.loads((CEN / b / 'library.json').read_text())
          if (CEN / b / _fname(s['sid'])).is_file()}
    common = la & lb
    res = {'a': a, 'b': b, 'dose_Hz': dose, 'trials': trials, 'n_common_stimuli': len(common)}
    for v in (a, b):
        lib, doses, dn, nact, _ = load(v, sids=common, trials=trials)
        di = list(doses).index(dose)
        r, _ = decode(np.sqrt(dn[:, di]))
        res[v] = r | {'median_active': float(np.median(nact[:, di])),
                      'frac_runaway_gt3000': float((np.median(nact[:, di], 1) > 3000).mean())}
    (OUT / 'published_vs_corrected.json').write_text(json.dumps(res, indent=1))
    return res


def main(variant='v783_alfix_all'):
    ann = annotations()
    lib, doses, dn, nact, dn_ids = load(variant)
    Y = np.sqrt(dn)
    S, D, T, F = Y.shape
    res = {'variant': variant, 'n_stimuli': S, 'doses': doses.tolist(), 'n_trials': T, 'n_DN': F,
           'note': 'rank and bits are lower bounds that grow with the number of stimuli (rank_saturation.py)'}
    mean = dn.mean(2)                                               # [S, D, F] Hz
    res['responding_DNs'] = {f'{int(d)}Hz': {'median': float(np.median((mean[:, i] >= 1).sum(1))),
                                             'frac_stimuli_with_any_DN': float(((mean[:, i] >= 1).sum(1) > 0).mean()),
                                             'union_DNs_ever': int((mean[:, i] >= 1).any(0).sum())}
                             for i, d in enumerate(doses)}
    res['union_DNs_any_dose'] = int((mean >= 1).any((0, 1)).sum())
    # which senses reach the neck: share of stimuli with no DN spike, by cell class (visual projection types pooled)
    cls = np.array([s['cell_class'] if s['kind'] == 'sensory' else 'visual_projection' for s in lib])
    silent = dn.reshape(S, D, -1).max(2) == 0                       # [S, D]
    res['silent_by_class'] = {c: {'n_stimuli': int((cls == c).sum()),
                                  **{f'{int(d)}Hz': float(silent[cls == c, i].mean()) for i, d in enumerate(doses)}}
                              for c in sorted(set(cls))}
    res['runaway_conditions'] = int((np.median(nact, 2) > 3000).sum())
    flag, flagged = ignition_flags(lib, nact, doses, dn)
    rule = ('whole-brain runaway: median active neurons > 3,000 (single-rate census)' if dn is None or len(doses) < 2
            else 'DN pattern at 100 Hz correlates at r >= 0.8 with the mean of dose-insensitive olfactory responses')
    res['ignition'] = {'n_flagged': int(flag.sum()), 'flagged': flagged, 'rule': rule}
    # rank of the channel
    res['cvpca_all'] = cvpca(Y.reshape(S * D, T, F))
    res['cvpca_all']['rank_over_null_seeds'] = [cvpca(Y.reshape(S * D, T, F), seed=k)['rank_above_null'] for k in (1, 2)]
    res['cvpca_by_dose'] = {f'{int(d)}Hz': {k: v for k, v in cvpca(Y[:, i]).items() if not isinstance(v, list)}
                            for i, d in enumerate(doses)}
    # capacity
    res['decode_stimulus'] = {f'{int(d)}Hz': decode(Y[:, i])[0] for i, d in enumerate(doses)}
    if D > 1:
        res['decode_stimulus_x_dose'] = decode(Y.reshape(S * D, T, F))[0]
    if flag.any():
        k = ~flag
        res['without_ignition'] = {'n_stimuli': int(k.sum()),
                                   'decode_stimulus': {f'{int(d)}Hz': decode(Y[k, i])[0] for i, d in enumerate(doses)},
                                   'cvpca_all_rank': cvpca(Y[k].reshape(int(k.sum()) * D, T, F))['rank_above_null']}
    # nuance: dose decoding within stimulus, and dose-response shapes
    steps = np.array([])
    if D > 1:
        dose_bits = [decode(Y[s])[0]['fano_bits_lower_bound'] for s in range(S) if mean[s].max() >= 1]
        shapes, steps = nuance_shapes(mean)
        res['nuance'] = {'dose_bits_median': float(np.median(dose_bits)) if dose_bits else np.nan,
                         'dose_bits_max_possible': float(np.log2(D)), 'n_stimuli_with_response': len(dose_bits), **shapes}
    # monosemantic vs polysemantic DNs, per stimulus (cell type x side) and per cell type (sides merged)
    types = np.array([f"{s['kind']}:{s['cell_type']}" for s in lib])
    ut, ti = np.unique(types, return_inverse=True)
    res['polysemanticity'] = {}
    for i, d in enumerate(doses):
        m = mean[:, i]                                              # [S, F]
        on = m >= 1
        n_stim = on.sum(0)
        n_type = np.zeros(F, int)
        for j in range(len(ut)):
            n_type += on[ti == j].any(0)
        used = n_stim > 0
        resp = on.any(1)                                            # stimuli that drive any DN
        r = m[resp][:, used]
        ls = (1 - (r.mean(0) ** 2) / np.maximum((r ** 2).mean(0), 1e-12)) / (1 - 1 / max(r.shape[0], 2))
        res['polysemanticity'][f'{int(d)}Hz'] = {
            'DNs_used': int(used.sum()), 'frac_used_driven_by_1_stimulus': float((n_stim[used] == 1).mean()),
            'frac_used_driven_by_1_cell_type': float((n_type[used] == 1).mean()),
            'frac_used_driven_by_2to5_cell_types': float(((n_type[used] >= 2) & (n_type[used] <= 5)).mean()),
            'frac_used_driven_by_gt5_cell_types': float((n_type[used] > 5).mean()),
            'median_cell_types_per_used_DN': float(np.median(n_type[used])), 'max_cell_types_per_DN': int(n_type.max()),
            'median_lifetime_sparseness_over_responsive_stimuli': float(np.median(ls)),
            'n_responsive_stimuli': int(resp.sum())}
    # hemispheres
    rows = []
    if 100 in doses:
        i100 = list(doses).index(100)
        rows, fp = mirror(lib, Y, dn_ids, ann, i100)
        lr = np.array([np.log2((r['dn_spikes_right_stim'] + 1) / (r['dn_spikes_left_stim'] + 1)) for r in rows])
        res['mirror_100Hz'] = fp | _summ(rows) | {'side_bias': {
            'median_log2_right_over_left_DN_spikes': float(np.median(lr)), 'frac_types_right_louder': float((lr > 0).mean())}}
        res['mirror_100Hz_count_matched'] = _summ([r for r in rows if r['count_matched']])
        res['mirror_100Hz_size_matched'] = _summ([r for r in rows if r['size_matched']])
        res['mirror_100Hz_size_matched']['criterion'] = 'driven cells within 10% and their output synapses within 20%'
    (OUT / f'expression_{variant}.json').write_text(json.dumps(res, indent=1))
    (OUT / f'mirror_rows_{variant}.json').write_text(json.dumps(rows, indent=1))
    np.savez_compressed(OUT / f'expression_arrays_{variant}.npz', steps=steps, mean=mean, nact=nact,
                        sids=np.array([s['sid'] for s in lib]))
    if variant != REFERENCE and (CEN / REFERENCE).is_dir():
        res['compare_with_reference'] = compare_variants(variant)
    brief = {k: v for k, v in res.items() if k not in ('cvpca_all',)}
    brief['cvpca_all'] = {k: v for k, v in res['cvpca_all'].items() if not isinstance(v, list) or k == 'rank_over_null_seeds'}
    print(json.dumps(brief, indent=1))


if __name__ == '__main__':
    main(*(sys.argv[1:2] or ['v783_alfix_all']))
