"""Figures for parts 2 and 3, the DN census and the regrowth study (palette and style from stage1/figures.py)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage1'))
from figures import AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, _title, plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results' / 'expression'
FIG = RES / 'figures'


def _title_lines(ax, text, sub, fontsize=8):
    """Title with a subtitle of any number of lines (stage1 _title assumes one)."""
    ax.set_title(text, loc='left', fontweight='semibold', pad=4 + 1.35 * fontsize * (sub.count('\n') + 1))
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=fontsize, va='bottom')


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches='tight')
    plt.close(fig)


def rank_and_capacity(e):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={'wspace': 0.32})
    cv = np.array(e['cvpca_all']['cv_variance'])
    null = np.array(e['cvpca_all']['null95'])
    r = e['cvpca_all']['rank_above_null']
    kmax = int(min(len(cv), max(3 * r, 20)))
    cv, null = cv[:kmax], null[:kmax]
    k = np.arange(1, kmax + 1)
    pos = cv > cv.max() * 1e-5
    a1.loglog(k[pos], cv[pos], color=BLUE, lw=2, label='reliable variance (cross-validated)')
    a1.loglog(k, np.clip(null, cv.max() * 1e-5, None), color=MUTED, lw=1.2, ls='--', label='95% shuffle null')
    a1.set_ylim(cv.max() * 1e-5, cv.max() * 3)
    a1.axvline(r, color=AXIS, lw=0.8)
    a1.text(r * 1.08, cv[pos].max() * 0.6, f'{r} reliable dimensions', color=INK2, fontsize=8)
    a1.set_xlabel('principal component')
    a1.set_ylabel('variance shared by independent trial halves')
    a1.legend(loc='lower left')
    _title(a1, 'How many dimensions does the neck use?',
           f"{e['n_DN']} descending neurons, {e['n_stimuli']} stimuli x {len(e['doses'])} rates (cross-validated PCA)")
    doses = e['doses']
    dec = [e['decode_stimulus'][f'{int(d)}Hz'] for d in doses]
    mi = [x['MI_plugin_bits'] for x in dec]
    part = [x['partition_bits'] for x in dec]
    fb = [x['fano_bits_lower_bound'] for x in dec]
    a2.plot(doses, mi, 'o-', color=BLUE, lw=2, ms=7, mec=SURFACE, mew=1.2, label='information (plug-in)', zorder=3)
    a2.plot(doses, part, 'o:', color=BLUE, lw=1.2, ms=4, mec=SURFACE, mew=1, alpha=0.75,
            label='partition count (stimuli decoded on >= half the trials)', zorder=3)
    a2.plot(doses, fb, 'o--', color=MUTED, lw=1.5, ms=5, mec=SURFACE, mew=1, label='Fano lower bound', zorder=3)
    for j, (d, b, x) in enumerate(zip(doses, mi, dec)):
        a2.annotate(f"{x['distinguishable_patterns']} patterns\n{x['frac_silent']:.0%} silent", (d, b),
                    textcoords='offset points', xytext=[(2, 36), (10, -20)][j] if j < 2 else (-14, 11), fontsize=7,
                    color=INK2, ha='left' if j < 2 else 'center', bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.5),
                    arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.7, shrinkA=0, shrinkB=4) if j == 0 else None)
    mx = np.log2(e['n_stimuli'])
    a2.axhline(mx, color=MUTED, lw=1, ls=':')
    a2.text(doses[0], mx + 0.12, f'{e["n_stimuli"]} stimuli = {mx:.1f} bits', color=INK2, fontsize=7.5)
    a2.set_xscale('log', base=2)
    a2.set_xticks(doses)
    a2.set_xticklabels([f'{int(d)}' for d in doses])
    a2.set_ylim(0, mx + 0.9)
    a2.set_xlabel('sensory drive (Hz)')
    a2.set_ylabel('bits per 1-s trial')
    a2.legend(loc='upper left', bbox_to_anchor=(0.0, 0.93))
    a2.grid(axis='y')
    sub = (f"single-trial decoding of which stimulus was given\n"
           f"corrected wiring, {e['n_stimuli']} stimuli, {e['n_trials']} trials each")
    _title_lines(a2, 'How much does one second of output say?', sub, fontsize=7.5)
    _save(fig, 'expression_rank_capacity.png')


DOSE_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']   # ordinal blue ramp, light to dark


def saturation(s, rb=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={'wspace': 0.42, 'width_ratios': [1.35, 1]})
    top = 0
    for (k, rows), c in zip(s['saturation_by_dose'].items(), DOSE_RAMP):
        n = np.array([r['n_stimuli'] for r in rows])
        m = np.array([r['rank_mean'] for r in rows])
        sd = np.array([r['rank_sd'] for r in rows])
        a1.fill_between(n, m - sd, m + sd, color=c, alpha=0.18, linewidth=0)
        a1.plot(n, m, 'o-', color=c, lw=2, ms=4.5, mec=SURFACE, mew=1, label=k.replace('Hz', ' Hz'))
        a1.annotate(k.replace('Hz', ' Hz'), (n[-1], m[-1]), textcoords='offset points', xytext=(6, -3), fontsize=7.5,
                    color=INK2)
        top = max(top, n[-1])
    a1.plot([0, top], [0, top], color=MUTED, lw=1, ls='--')
    a1.annotate('one dimension\nper stimulus', (top * 0.78, top * 0.78), textcoords='offset points', xytext=(-62, 14),
                fontsize=7.5, color=INK2, ha='right', arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.8))
    a1.set_xlim(0, top * 1.12)
    a1.set_ylim(0, top * 1.05)
    a1.set_xlabel('stimuli in the set (random subsets)')
    a1.set_ylabel('reliable dimensions of the DN output')
    a1.grid(axis='y')
    a1.legend(loc='upper left', title='drive', title_fontsize=7.5)
    _title(a1, 'Does the neck run out of dimensions?', 'cross-validated rank vs stimulus-set size (mean ± sd, 5 subsets)')
    b = s['bottleneck']['without_ignition']
    p = (rb or {}).get('participation_ratio_control', {}).get('without_ignition')
    if p:                                       # DNs vs random, breadth-matched and all other responders
        labels = [f"{p['n_DN']} responding\nDNs", f"random {p['n_DN']}\nother\nresponders", f"{p['n_DN']} others\nas broadly\ntuned",
                  f"all {b['n_rest_responding']:,}\nother\nresponders"]
        vals = [p['pr_DN'], p['pr_rest_random']['mean'], p['pr_rest_breadth_matched']['mean'], b['pr_rest_of_brain']]
        err = [0, p['pr_rest_random']['sd'], p['pr_rest_breadth_matched']['sd'], 0]
        cols = [BLUE, MUTED, DOSE_RAMP[0], MUTED]
    else:
        labels = [f"{b['n_DN_responding']} responding\nDNs", f"random {b['n_DN_responding']}\nother responders",
                  f"all {b['n_rest_responding']:,}\nother responders"]
        vals = [b['pr_DN'], b['pr_random_rest_matched_mean'], b['pr_rest_of_brain']]
        err = [0, b['pr_random_rest_matched_sd'], 0]
        cols = [BLUE, MUTED, MUTED]
    xs = range(len(vals))
    a2.bar(xs, vals, width=0.6, color=cols, edgecolor=SURFACE, linewidth=2)
    for i, (v, e_) in enumerate(zip(vals, err)):
        if e_:
            a2.errorbar([i], [v], yerr=[e_], color=INK2, capsize=3, lw=1)
        a2.text(i, v + e_ + max(vals) * 0.02, f'{v:.1f}', ha='center', va='bottom', fontsize=8, color=INK)
    a2.set_xticks(list(xs))
    a2.set_xticklabels(labels, fontsize=7)
    a2.set_ylim(0, max(vals) * 1.18)
    a2.set_ylabel('participation ratio (effective dimensions)')
    a2.grid(axis='y')
    _title_lines(a2, 'Is the output more compressed than the brain?',
                 'no more than neurons as broadly tuned as the DNs\n'
                 f"trial means, {b['n_stimuli']} stimuli x 4 rates (the {s['bottleneck']['all_stimuli']['n_stimuli'] - b['n_stimuli']} "
                 'odour stimuli that tip into a separate state excluded)')
    _save(fig, 'expression_rank_saturation.png')


def nuance_and_mirror(e, steps, mirror_rows, rb=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.7), gridspec_kw={'wspace': 0.32})
    a1.hist(steps, bins=np.linspace(0, 1, 21) + 1e-9, color=BLUE, edgecolor=SURFACE, linewidth=1.5)   # 0.5 -> graded bin
    a1.set_xlabel('largest single step / full range of a DN dose response')
    a1.set_ylabel('stimulus-DN pairs (active at >= 2 rates)')
    n = e['nuance']
    a1.text(0.02, 0.97, f"graded (<= 0.5): {n['frac_graded_of_classified']:.0%}\n"
                        f"switch-like (>= 0.8): {n['frac_switch_like_of_classified']:.0%}\n"
                        f"plus {n['frac_top_dose_only']:.0%} of all pairs that turn on\nonly at 200 Hz (shape unknown)\n"
                        f"intensity decodable: {n['dose_bits_median']:.1f} of {n['dose_bits_max_possible']:.0f} bits",
            transform=a1.transAxes, va='top', fontsize=7.5, color=INK2)
    _title(a1, 'Nuance: graded or all-or-none?', f"{n['n_shape_classified']} pairs with a measurable shape (peak >= 5 Hz)")
    rows = [r for r in mirror_rows]
    val = [r['mirror_r'] for r in rows]                     # one side silent scores 0 (analyze.mirror)
    o = np.argsort(val)
    y = np.arange(len(rows))
    col = [BLUE if rows[j]['size_matched'] else MUTED for j in o]
    a2.scatter(np.array(val)[o], y, s=12, c=col, zorder=3)
    a2.axvline(1, color=AXIS, lw=0.8)
    a2.axvline(0, color=AXIS, lw=0.8)
    a2.set_yticks([])
    a2.set_xlim(-0.6, 1.25)
    a2.set_xlabel('left-vs-right similarity of the DN output (r; one side silent = 0)')
    a2.set_ylabel('stimulus type (sorted)')
    m = e['mirror_100Hz']
    a2.text(0.02, 0.97, f"median r {m['median_mirror_r']:.2f}\n{m['n_one_side_silent']} of {m['n_types']} types: one side silent\n"
                        f"active DNs shared: {m['median_jaccard_active']:.0%} (median)\n"
                        f"side decodable from 1 trial: {m['frac_types_hemisphere_decodable_gt_0.8']:.0%} of types\n"
                        + (f"same side, split trials: r {rb['mirror_floor_100Hz']['same_side_split_half_median_r']:.3f}\n"
                           if rb and 'mirror_floor_100Hz' in rb else '') +
                        "blue: cells and synapses matched L/R", transform=a2.transAxes, va='top', ha='left',
            bbox=dict(facecolor=SURFACE, edgecolor='none', pad=2),
            fontsize=7.5, color=INK2)
    _title(a2, 'Two hemispheres, two voices', 'same cell type driven on the left or on the right, 100 Hz')
    _save(fig, 'expression_nuance_hemispheres.png')


def braun(rows, base=None):
    """Primary readout: GNG DNs (third letter 'g'), Braun et al.'s connectome proxy for the Dfd+ GNG DNs they imaged."""
    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        c = BLUE if r['braun_class'] == 'broadcaster' else ORANGE
        ax.plot([i, i], [r['recruited_gng_50Hz'], r['recruited_gng_200Hz']], color=c, lw=2, alpha=0.45)
        ci = r.get('recruited_gng_200Hz_ci95')
        if ci:
            ax.errorbar([i], [r['recruited_gng_200Hz']], yerr=[[r['recruited_gng_200Hz'] - ci[0]], [ci[1] - r['recruited_gng_200Hz']]],
                        color=c, capsize=2.5, lw=1, zorder=3)
        ax.scatter([i], [r['recruited_gng_200Hz']], s=46, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        ax.scatter([i + 0.22], [r['recruited_200Hz']], s=14, color=MUTED, zorder=3)
        ax.scatter([i - 0.25], [r['braun_downstream_DNs']], s=40, facecolor='none', edgecolor=INK2, linewidth=1.2, zorder=5)
        ax.scatter([i - 0.25], [r['connectome_downstream_DNs_ge5']], s=20, marker='D', color=AXIS, edgecolor=SURFACE,
                   linewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([r['set'] for r in rows], rotation=40, ha='right')
    ax.set_ylabel('descending neurons')
    ax.grid(axis='y')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=BLUE, label='model: GNG DNs recruited at 200 Hz (bar from 50 Hz; 95% CI), broadcaster'),
                       Line2D([], [], marker='o', ls='', color=ORANGE, label='model: same, standalone'),
                       Line2D([], [], marker='o', ls='', color=MUTED, ms=4, label='model: all DNs recruited at 200 Hz'),
                       Line2D([], [], marker='o', ls='', mfc='none', mec=INK2, label='Braun et al.: monosynaptic DN partners (v630)'),
                       Line2D([], [], marker='D', ls='', color=AXIS, ms=5, label='this wiring: DN partners at >= 5 synapses')],
              loc='upper right', fontsize=7)
    by = {r['set']: r for r in rows}
    o = [by[k] for k in ('DNp09', 'aDN2', 'MDN')]
    sub = (f"Braun imaged Dfd+ GNG DNs: DNp09 > aDN2 > MDN. Model ('g'-named proxy, 200 Hz): {o[0]['recruited_gng_200Hz']} / "
           f"{o[1]['recruited_gng_200Hz']} / {o[2]['recruited_gng_200Hz']}: MDN weakest, top two tied or swapped")
    if base is not None:
        rho = base['spearman_LIF_vs_baselines']
        sub += (f"\nmodel vs v783 DN-partner counts: Spearman {rho['all >=5-synapse DN partners']['gng_200Hz']:.2f}; with inhibitory sets as 0 "
                f"{rho['signed count']['gng_200Hz']:.2f} (GNG, 200 Hz). DNb02: glutamate = inhibitory here; DNa01: weak synapses")
    ax.set_title('Command neurons: does the model recruit DN networks like flies do?', loc='left', fontweight='semibold',
                 pad=8 + 12 * (sub.count(chr(10)) + 1))
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=7.5, va='bottom')
    _save(fig, 'expression_braun_recruitment.png')


def collapse(cor_nact, pub_nact):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, (lab, v, c) in enumerate([('published wiring', pub_nact, ORANGE), ('corrected wiring', cor_nact, BLUE)]):
        jit = np.random.default_rng(i).uniform(-0.25, 0.25, len(v))
        ax.scatter(i + jit, np.maximum(v, 1), s=8, color=c, alpha=0.6, linewidths=0)
        ax.text(i, 2.2e4, f"{(v > 3000).sum()} of {len(v)} stimuli\nrun away", ha='center', fontsize=8, color=INK2)
    ax.axhline(3000, color=MUTED, lw=1, ls='--')
    ax.set_yscale('log')
    ax.set_ylim(1, 6e4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['published wiring', 'corrected wiring'])
    ax.set_ylabel('active neurons (median over trials)')
    _title(ax, 'One stuck state swallows many stimuli', 'every census stimulus at 100 Hz; dashed: 3,000 neurons')
    _save(fig, 'expression_runaway_collapse.png')


LABEL = {'type_x_side': 'cell type x side', 'supertype': 'supertype', 'hemilineage': 'hemilineage',
         'cell_class': 'cell class', 'super_class': 'super class', 'type_sides_pooled': 'cell type, sides pooled',
         'wshuffle_type_x_side': 'real wiring, counts shuffled', 'supertype_keep_readouts': 'supertype, readouts kept',
         'hemilineage_keep_readouts': 'hemilineage, readouts kept', 'hemilineage_else_supertype': 'hemilineage, else supertype',
         'type_sides_pooled_keep_readouts': 'sides pooled, readouts kept'}
GREEN = '#1baf7a'                                          # third categorical colour, after blue and orange
MARK = {'ladder': 'o', 'sides_pooled': 'o', 'weights_shuffled': 'D', 'readouts_kept': '^', 'fallback_supertype': 's'}
EDGE = {'ladder': None, 'sides_pooled': BLUE, 'weights_shuffled': ORANGE, 'readouts_kept': GREEN, 'fallback_supertype': GREEN}


def compression(c):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={'wspace': 0.34, 'width_ratios': [1.25, 1]})
    lv = list(c['levels'])
    xs, ys, placed = [], [], []
    for k in lv:
        d = c['levels'][k]
        x = d['description']['MB']
        f = [b['fidelity_paired_DNs']['median_r'] for b in d['brains']]
        ser = d['series']
        mk = MARK[ser]
        fc = BLUE if ser == 'ladder' else 'none'
        ec = SURFACE if ser == 'ladder' else EDGE[ser]
        a1.scatter([x] * len(f), f, s=28, marker=mk, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
        if ser == 'ladder':
            xs.append(x); ys.append(np.mean(f))
        close = any(abs(math.log(x / x2)) < 0.1 for x2 in placed)                # e.g. cell class vs super class
        off = {'weights_shuffled': (-6, 9), 'sides_pooled': (7, 6), 'readouts_kept': (-7, 6),
               'fallback_supertype': (7, -10)}.get(ser, {'cell_class': (7, -13), 'super_class': (7, 6)}.get(k, (7, -3)))
        ha = {'weights_shuffled': 'center', 'readouts_kept': 'right'}.get(ser, 'left')
        off, ha = {'type_sides_pooled_keep_readouts': ((7, -3), 'left'), 'hemilineage_keep_readouts': ((0, 10), 'center')}.get(k, (off, ha))
        a1.annotate('counts shuffled' if ser == 'weights_shuffled' else LABEL.get(k, k), (x, np.mean(f)),
                    textcoords='offset points', xytext=off, fontsize=7, color=INK2, ha=ha)
        placed.append(x)
    o = np.argsort(xs)
    a1.plot(np.array(xs)[o], np.array(ys)[o], color=BLUE, lw=1.5, alpha=0.6, zorder=2)
    nc = c.get('noise_ceiling_paired_DNs', {}).get('median_r', 1.0)
    a1.scatter([c['neuron_level_MB']], [nc], marker='s', s=36, color=INK, zorder=4)
    a1.annotate('real wiring', (c['neuron_level_MB'], nc), textcoords='offset points', xytext=(7, -3), fontsize=7,
                color=INK2, ha='left')
    a1.axhline(nc, color=MUTED, lw=1, ls=':')
    a1.text(0.01, nc + 0.015, 'same brain, new input noise', fontsize=7, color=INK2, transform=a1.get_yaxis_transform())
    hf = c['hemisphere_floor']
    a1.axhspan(hf['ci95'][0], hf['ci95'][1], color=ORANGE, alpha=0.12, lw=0)
    a1.axhline(hf['median_r'], color=ORANGE, lw=1, ls='--')
    a1.text(0.01, hf['median_r'] + 0.015, 'left vs right hemisphere (95% CI)', fontsize=7, color=INK2,
            transform=a1.get_yaxis_transform())
    a1.set_xscale('log')
    a1.set_xlim(1.2, c['neuron_level_MB'] * 2.2)
    a1.set_xticks([1.5, 2, 3, 5, 10, 20, 30])
    a1.set_xticklabels(['1.5', '2', '3', '5', '10', '20', '30'])
    a1.minorticks_off()
    a1.set_ylim(-0.32, 1.1)
    a1.set_xlabel('description of the wiring rules (MB, log scale)')
    a1.set_ylabel('DN output vs real brain (median r over probes)')
    a1.grid(axis='y')
    _title_lines(a1, 'Regrow a fly from its parts list', 'one marker per regrown brain; filled: side-aware ladder; open circles: sides\n'
                                                         'pooled; diamonds: real wiring, counts shuffled; green: controls')
    n_calls = len(c['real']['calls'])
    order = sorted(lv, key=lambda k: -c['levels'][k]['description']['MB'])
    for i, k in enumerate(order):
        d = c['levels'][k]
        n = [b['n_reproduced'] for b in d['brains']]
        ser = d['series']
        jit = np.linspace(-0.12, 0.12, len(n))
        a2.scatter(i + jit, n, s=30, marker=MARK[ser], facecolor=BLUE if ser == 'ladder' else 'none',
                   edgecolor=SURFACE if ser == 'ladder' else EDGE[ser], linewidth=1.2, zorder=3)
    a2.axhline(n_calls, color=INK, lw=1, ls='--')
    a2.text(1.0, n_calls + 0.2, f'real brain: {n_calls} of {n_calls}', fontsize=7.5, color=INK2, ha='right',
            transform=a2.get_yaxis_transform())
    a2.set_xticks(range(len(order)))
    a2.set_xticklabels([LABEL.get(k, k) for k in order], rotation=32, ha='right', fontsize=7.5)
    a2.set_ylim(-0.3, n_calls + 0.8)
    a2.set_ylabel('validated behaviour calls reproduced')
    a2.grid(axis='y')
    _title_lines(a2, 'Which behaviours survive?', 'sugar, graded, contra>ipsi, bitter, Ir94e, JON-CE, JON-F;\nfinest rules on the left')
    _save(fig, 'compression_regrow.png')


CAT5 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # categorical colours 1-5


def _spread(vals, gap):
    """Nudge label positions apart so neighbours are at least `gap` apart (order kept)."""
    o = np.argsort(vals)
    pos = np.array(vals, float)[o]
    for j in range(1, len(pos)):
        pos[j] = max(pos[j], pos[j - 1] + gap)
    pos -= max(0, pos[-1] - max(vals)) / 2
    out = np.empty_like(pos)
    out[o] = pos
    return out


def compression_modality(c):
    """What is lost first: fidelity of the DN output per probe modality along the side-aware ladder."""
    lad = sorted([k for k, d in c['levels'].items() if d['series'] == 'ladder'],
                 key=lambda k: -c['levels'][k]['description']['MB'])
    mods = ['taste', 'mechanosensory (JO, bristles)', 'olfactory', 'thermo/hygro', 'visual projection']
    nc = c.get('noise_ceiling_paired_DNs', {}).get('by_modality', {})
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    xs = np.arange(len(lad) + 1)
    ends = {}
    for m, col, mk in zip(mods, CAT5, ['o', 's', 'D', '^', 'v']):
        y = [nc[m]['median_r'] if m in nc else np.nan]
        for k in lad:
            v = [b['fidelity_paired_DNs']['by_modality'][m]['median_r'] for b in c['levels'][k]['brains']
                 if m in b['fidelity_paired_DNs'].get('by_modality', {})]
            y.append(np.mean(v) if v else np.nan)
        ax.plot(xs, y, marker=mk, color=col, lw=2, ms=6, mec=SURFACE, mew=1, label=m)
        ends[m] = y[-1]
    ax.set_xticks(xs)
    ax.set_xticklabels(['real wiring\n(new noise)'] + [f"{LABEL.get(k, k)}\n{c['levels'][k]['description']['MB']:.1f} MB"
                                                       for k in lad], fontsize=7)
    ax.set_xlim(-0.3, xs[-1] + 0.3)
    ax.set_ylim(-0.35, 1.08)
    ax.set_ylabel('DN output vs real brain (median r)')
    ax.grid(axis='y')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.16), ncol=3, fontsize=7)
    _title(ax, 'What is lost first?', 'fidelity of each sensory channel as the wiring rules get coarser (mean of 3 brains per level)')
    _save(fig, 'compression_by_modality.png')


def main():
    e = json.loads((RES / 'expression_v783_alfix_all.json').read_text())
    arr = np.load(RES / 'expression_arrays_v783_alfix_all.npz')
    pubf = RES / 'expression_v783_published100.json'
    pub = json.loads(pubf.read_text()) if pubf.is_file() else None
    rank_and_capacity(e)
    rbf = RES / 'robustness_v783_alfix_all.json'
    rb = json.loads(rbf.read_text()) if rbf.is_file() else None
    sf = RES / 'rank_saturation_v783_alfix_all.json'
    if sf.is_file():
        saturation(json.loads(sf.read_text()), rb)
    nuance_and_mirror(e, arr['steps'], json.loads((RES / 'mirror_rows_v783_alfix_all.json').read_text()), rb)
    bf = RES / 'braun_recruitment_v783_alfix_all.json'
    bb = RES / 'braun_baseline_v783_alfix_all.json'
    if bf.is_file():
        braun(json.loads(bf.read_text()), json.loads(bb.read_text()) if bb.is_file() else None)
    if pub is not None:
        pa = np.load(RES / 'expression_arrays_v783_published100.npz')
        ci = e['doses'].index(100.0)
        collapse(np.median(arr['nact'][:, ci], 1), np.median(pa['nact'][:, 0], 1))
    cf = RES / 'compression.json'
    if cf.is_file():
        cj = json.loads(cf.read_text())
        compression(cj)
        if any('by_modality' in b['fidelity_paired_DNs'] for d in cj['levels'].values() for b in d['brains']):
            compression_modality(cj)


if __name__ == '__main__':
    main()
