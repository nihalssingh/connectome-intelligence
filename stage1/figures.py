"""Stage 1 figures: stimulation + raster + rates, frequency response, reproduction check."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

# palette (light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE = '#2a78d6', '#eb6834'

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
    'legend.frameon': False, 'legend.fontsize': 8, 'lines.solid_capstyle': 'round',
})


def _title(ax, text, sub=None):
    ax.set_title(text, loc='left', fontweight='semibold', pad=14 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=8, va='bottom')


def _end_labels(ax, x_end, ends, labels, min_gap_frac=0.09):
    """Direct labels at line ends; labels pushed apart to a minimum gap get a thin leader line."""
    lo, hi = min(ends.values()), max(ends.values())
    gap = min_gap_frac * max(hi - lo, 1e-9) if len(ends) > 1 else 0
    gap = max(gap, min_gap_frac * (ax.get_ylim()[1] - ax.get_ylim()[0]))
    order = sorted(ends, key=ends.get)
    placed = {}
    for k in order:
        y = ends[k]
        if placed:
            y = max(y, max(placed.values()) + gap)
        placed[k] = y
    shift = max(0, max(placed.values()) - hi - gap)  # keep the stack centred on the data
    for k in order:
        y_lab = placed[k] - shift / 2
        ax.annotate(labels[k], xy=(x_end, ends[k]), xytext=(x_end + 30, y_lab), textcoords='data',
                    color=INK2, fontsize=8, va='center', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.8, shrinkA=0, shrinkB=2))


def _fmt_p(p, n_perm=None):
    if n_perm and p <= 1 / (n_perm + 1) + 1e-12:
        return f'p < {1 / n_perm:.0e}'.replace('e-0', 'e-')
    return f'p = {p:.3f}' if p < 0.1 else f'p = {p:.2f}'


def sugar_response(df_trial, inputs, responders, labels, highlight, rates, rate_labels, out, r_poi):
    """One trial: input raster, responder raster (latency order), and 30-trial rates of key neurons.

    df_trial: spikes of a single trial; inputs/responders: ordered ID lists; labels: id -> y label;
    highlight: ids drawn in the accent colour; rates: PSTH DataFrame (index ms, columns ids).
    """
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True,
                                     gridspec_kw={'height_ratios': [1.1, 3.6, 1.5], 'hspace': 0.42})
    t_ms = df_trial['t'].to_numpy() * 1000
    fid = df_trial['flywire_id'].to_numpy()

    def raster(ax, ids, accent):
        trains = [t_ms[fid == i] for i in ids]
        colors = [ORANGE if i in accent else INK2 for i in ids]
        ax.eventplot(trains, colors=colors, linelengths=0.75, linewidths=0.7)
        ax.set_ylim(len(ids) - 0.5, -0.5)

    raster(a1, inputs, set())
    a1.set_yticks([])
    a1.set_ylabel(f'{len(inputs)} GRNs')
    _title(a1, 'Input: 21 labellar sugar GRNs', f'Poisson stimulation at {r_poi} Hz, one trial (repo calls them "right"; FlyWire: fly\'s left)')

    raster(a2, responders, set(highlight))
    a2.set_yticks(range(len(responders)))
    a2.set_yticklabels([labels.get(i, f'…{str(i)[-6:]}') for i in responders], fontsize=6.5)
    for tick, i in zip(a2.get_yticklabels(), responders):
        if i in highlight:
            tick.set_color(INK)
            tick.set_fontweight('semibold')
    _title(a2, 'Response: most active downstream neurons', 'same trial; ordered by median first-spike latency (30 trials)')

    ends = {}
    for i, color in zip(rates.columns, [ORANGE, BLUE, INK2, MUTED]):
        a3.plot(rates.index, rates[i], color=color, lw=2, label=rate_labels[i])
        ends[i] = rates[i].iloc[-1]
    _end_labels(a3, rates.index[-1], ends, rate_labels)
    a3.set_ylabel('rate (Hz)')
    a3.set_xlabel('time from stimulus onset (ms)')
    a3.set_ylim(bottom=0)
    a3.grid(axis='y')
    a3.set_xlim(0, 1000)
    _title(a3, 'Firing rate', '30-trial average, 20 ms bins')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def frequency_response(freqs, curves, out, published=None):
    """MN9 rate vs. stimulation frequency.

    curves: {label: (mean array, sd array, colour)}; published: {label: (freqs, mean, sd, colour)}
    plotted as open markers with SD whiskers.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, (m, sd, color) in curves.items():
        ax.fill_between(freqs, m - sd, m + sd, color=color, alpha=0.10, lw=0)
        ax.plot(freqs, m, color=color, lw=2, label=f'{label} (re-run, mean ± SD)')
    if published:
        for label, (f, m, sd, color) in published.items():
            ax.errorbar(f, m, yerr=sd, fmt='o', ms=6, mfc=SURFACE, mec=color, mew=1.6, ecolor=color,
                        elinewidth=1, capsize=0, label=label, zorder=3)
    ax.set_xlabel('sugar GRN stimulation rate (Hz)')
    ax.set_ylabel('MN9 firing rate (Hz)')
    ax.set_xlim(0, max(freqs) + 5)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y')
    ax.legend(loc='upper left')
    _title(ax, 'Sugar GRN rate → MN9 (proboscis motor neuron)', '21 sugar GRNs, 30 trials × 1 s per point, FlyWire v630; contra/ipsi relative to the stimulated side')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def replicate_calibration(pairs, mn9_rows, out):
    """Left: MN9 vs frequency for the authors' four archived sweeps and our re-run(s).
    Right: ECDF of trial-permutation p-values, authors-vs-authors (the model's own null) vs ours-vs-authors."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={'wspace': 0.32})
    authors = [m for m in mn9_rows if not m['run'].startswith('ours')]
    a1.scatter([m['freq'] for m in authors], [m['MN9_L_mean'] for m in authors], s=22, color=MUTED,
               edgecolor=SURFACE, linewidth=0.6, zorder=2, label="authors' 4 archived sweeps")
    ours = sorted((m for m in mn9_rows if m['run'] == 'ours'), key=lambda m: m['freq'])
    a1.plot([m['freq'] for m in ours], [m['MN9_L_mean'] for m in ours], color=ORANGE, lw=2, zorder=3, label='this re-run')
    a1.scatter([m['freq'] for m in ours], [m['MN9_L_mean'] for m in ours], s=36, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    reps = [m for m in mn9_rows if m['run'].startswith('ours_rep')]
    if reps:
        a1.scatter([m['freq'] + 4 for m in reps], [m['MN9_L_mean'] for m in reps], s=30, facecolor=SURFACE, edgecolor=ORANGE,
                   linewidth=1.4, zorder=4, label='our 200 Hz repeats')
    a1.set_xlabel('sugar GRN stimulation rate (Hz)')
    a1.set_ylabel('MN9 (contra) mean rate (Hz, 30 trials)')
    a1.set_ylim(bottom=0)
    a1.set_xlim(0, 215)
    a1.grid(axis='y')
    a1.legend(loc='upper left')
    _title(a1, 'MN9 response vs. the model’s own run-to-run spread', 'FlyWire v630, 21 sugar GRNs, default parameters')

    xs = np.linspace(0, 1, 101)
    a2.plot(xs, xs, color=AXIS, lw=1, zorder=1)
    for kind, color, label in (('authors-authors', MUTED, 'authors vs. authors'), ('ours-authors', ORANGE, 'this re-run vs. authors')):
        p = np.sort([r['perm_p'] for r in pairs if r['kind'] == kind])
        if len(p):
            a2.step(np.r_[0, p], np.r_[0, np.arange(1, len(p) + 1) / len(p)], where='post', color=color, lw=2,
                    label=f'{label} (n = {len(p)} pairs)')
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1)
    a2.set_xlabel('trial-permutation test p-value')
    a2.set_ylabel('fraction of run pairs')
    a2.legend(loc='lower right')
    a2.text(0.03, 0.97, 'diagonal = runs statistically exchangeable', transform=a2.transAxes, va='top', color=INK2, fontsize=8)
    _title(a2, 'Are runs exchangeable?', 'all pairwise comparisons at 20, 40, …, 200 Hz')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def version_comparison(rows, out):
    """MN9 (contra, ipsi) vs. stimulation rate on FlyWire v630 and v783, same 21 GRNs and parameters."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    f = [r['freq'] for r in rows]
    for side, color in (('contra', ORANGE), ('ipsi', BLUE)):
        for ver, style in (('v630', '-'), ('v783', '--')):
            m = np.array([r[f'MN9_{side}_{ver}']['mean_Hz'] for r in rows])
            sd = np.array([r[f'MN9_{side}_{ver}']['sd_Hz'] for r in rows])
            if ver == 'v630':
                ax.fill_between(f, m - sd, m + sd, color=color, alpha=0.10, lw=0)
                ax.plot(f, m, color=color, lw=2, label=f'MN9 {side}, v630')
            else:
                ax.plot(f, m, ls='none', marker='o', ms=6, mfc=SURFACE, mec=color, mew=1.6, label=f'MN9 {side}, v783')
    ax.set_xlabel('sugar GRN stimulation rate (Hz)')
    ax.set_ylabel('MN9 firing rate (Hz)')
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, 210)
    ax.grid(axis='y')
    ax.legend(loc='upper left')
    _title(ax, 'Dataset version: FlyWire v630 (paper) vs. v783 (public)', 'same 21 GRNs (one ID remapped), same parameters; line = v630 mean ± SD, circles = v783 mean')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def grn_set_dotplot(rows, out):
    """Dot plot: MN9 (contra) response for each annotation-defined GRN set at 50/100/200 Hz (mean ± SD)."""
    order = []
    for r in rows:
        if r['set'] not in order:
            order.append(r['set'])
    nice = {
        'shiu21': "Shiu's 21 'sugar' GRNs", 'shiu21_sugar_typed': "  of which sugar-typed (LB3b/c)",
        'shiu21_non_sugar_typed': "  of which not sugar-typed (LB3d/LB4b)", 'sugar_class_all': 'all sugar-typed GRNs (LB3b + LB3c)',
        'sugar_class_15': '15 random sugar-typed GRNs', 'high_salt_LB3d': 'high-salt GRNs (LB3d)',
    }
    n = {r['set']: r['n_grns'] for r in rows}
    ramp = {50: '#86b6ef', 100: '#3987e5', 200: '#1c5cab'}  # sequential blue: ordinal stimulation rate
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(order) + 1.4))
    for y, name in enumerate(order):
        for k, fr in enumerate((50, 100, 200)):
            hit = [r for r in rows if r['set'] == name and r['freq'] == fr]
            if hit:
                m, sd = hit[0]['MN9_contra']['mean_Hz'], hit[0]['MN9_contra']['sd_Hz']
                ax.errorbar(m, y + (k - 1) * 0.18, xerr=sd, fmt='o', ms=7, color=ramp[fr], mec=SURFACE, mew=1.2,
                            elinewidth=1, capsize=0, label=f'{fr} Hz' if y == 0 else None, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{nice.get(s, s)}  (n={n[s]})" for s in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel('MN9 (contra) firing rate, mean ± SD over 30 trials (Hz)')
    ax.set_xlim(left=0)
    ax.grid(axis='x')
    # below the axis, so it can never cover a data point
    ax.legend(title='GRN stimulation', loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    _title(ax, 'Which annotated GRN types drive MN9?', 'FlyWire v783, GRN types from flywire_annotations v3.2.0 (Tastekin et al. 2026); stimulated side only')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def reproduction_check(table, summary, t_null, out, title_sub):
    """Per-neuron mean rates (saved vs re-run) and the distribution of Welch z-scores vs N(0,1)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={'wspace': 0.3})
    floor = 0.02
    x = table['mean_ref'].clip(lower=floor)
    y = table['mean_new'].clip(lower=floor)
    lim = (floor * 0.8, 400)
    a1.plot(lim, lim, color=AXIS, lw=1, zorder=1)
    a1.scatter(x, y, s=16, color=BLUE, edgecolor=SURFACE, linewidth=0.5, zorder=2)
    a1.set_xscale('log')
    a1.set_yscale('log')
    a1.set_xlim(lim)
    a1.set_ylim(lim)
    a1.set_xlabel('saved output: mean rate (Hz)')
    a1.set_ylabel('re-run: mean rate (Hz)')
    a1.text(0.04, 0.96, f"{summary['n_neurons_compared']} neurons active in either run\n"
                        f"Pearson r = {summary['pearson_r_mean_rates']:.4f}",
            transform=a1.transAxes, va='top', color=INK2, fontsize=8)
    _title(a1, 'Per-neuron mean rate', title_sub)

    z = table['z'].to_numpy()
    bins = np.linspace(-4.5, 4.5, 37)
    a2.hist(z, bins=bins, color=BLUE, alpha=0.85, rwidth=0.88, density=True)
    xs = np.linspace(-4.5, 4.5, 200)
    a2.plot(xs, norm.pdf(xs), color=INK2, lw=1.5)
    a2.annotate('N(0, 1)', xy=(1.25, norm.pdf(1.25)), xytext=(2.3, 0.3), color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.8))
    a2.set_xlabel('Welch z (re-run − saved)')
    a2.set_ylabel('density')
    a2.set_ylim(0, max(0.62, a2.get_ylim()[1] * 1.25))
    a2.text(0.98, 0.97, f"|z| > 1.96: {100 * summary['frac_abs_z_gt_1.96']:.1f}% of neurons (5% expected)\n"
                        f"trial-permutation test: {_fmt_p(summary['perm_p'], summary['n_perm'])}\n"
                        f"neurons with FDR q < 0.05: {summary['n_mw_q_lt_0.05']}",
            transform=a2.transAxes, va='top', ha='right', color=INK2, fontsize=8)
    a2.grid(axis='y')
    _title(a2, 'Differences vs. trial-to-trial noise', 'if the runs are exchangeable, z ~ N(0, 1)')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
