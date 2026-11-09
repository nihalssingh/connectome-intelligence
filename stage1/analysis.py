"""Analysis for Stage 1: rates, run-vs-run reproduction test, PSTHs, latencies, hop distance.

Spike files are the reference model's output format: one row per spike with columns
t (s), trial, flywire_id, exp_name (see external/Drosophila_brain_model/model.py).
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import false_discovery_control, mannwhitneyu

T_RUN = 1.0  # s, default_params['t_run']
N_RUN = 30   # default_params['n_run']


def load_spikes(path):
    df = pd.read_parquet(path, columns=['t', 'trial', 'flywire_id'])
    df['t'] = df['t'].astype(float)
    df['trial'] = df['trial'].astype(int)
    return df


def rate_matrix(df, n_run=N_RUN, t_run=T_RUN):
    """Firing rate (Hz) per neuron and trial; trials without spikes count as 0 Hz."""
    counts = df.groupby(['flywire_id', 'trial']).size().unstack(fill_value=0)
    return counts.reindex(columns=range(n_run), fill_value=0) / t_run


def _welch_z(a, b, eps):
    """Per-row Welch z of mean(b) - mean(a); eps (Hz^2) floors the variance of near-constant rows."""
    va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    return (b.mean(axis=1) - a.mean(axis=1)) / np.sqrt((va + eps) / a.shape[1] + (vb + eps) / b.shape[1])


def compare_runs(ref, new, n_perm=5000, seed=0, eps=1.0):
    """Are two runs of the same stochastic experiment statistically exchangeable?

    ref, new: rate matrices (neuron x trial). Every neuron active in either run is compared.
    Global test: T = sum_i z_i^2 over neurons, with the null distribution from permuting whole
    trials between the runs (keeps the within-trial correlations between neurons intact).
    Also reports per-neuron Mann-Whitney tests with Benjamini-Hochberg FDR.
    """
    ids = ref.index.union(new.index)
    a = ref.reindex(ids, fill_value=0).to_numpy(float)
    b = new.reindex(ids, fill_value=0).to_numpy(float)
    z = _welch_z(a, b, eps)
    t_obs = float(np.sum(z ** 2))

    pooled = np.hstack([a, b])
    na = a.shape[1]
    rng = np.random.default_rng(seed)
    t_null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(pooled.shape[1])
        t_null[k] = np.sum(_welch_z(pooled[:, p[:na]], pooled[:, p[na:]], eps) ** 2)
    p_perm = (1 + np.sum(t_null >= t_obs)) / (1 + n_perm)

    p_mw = np.array([
        mannwhitneyu(x, y).pvalue if np.ptp(np.r_[x, y]) > 0 else 1.0 for x, y in zip(a, b)
    ])
    q_mw = false_discovery_control(p_mw)

    table = pd.DataFrame({
        'mean_ref': a.mean(1), 'sd_ref': a.std(1, ddof=1),
        'mean_new': b.mean(1), 'sd_new': b.std(1, ddof=1),
        'z': z, 'p_mw': p_mw, 'q_mw': q_mw,
    }, index=pd.Index(ids, name='flywire_id'))
    m_a, m_b = table.mean_ref.to_numpy(), table.mean_new.to_numpy()
    summary = {
        'n_neurons_compared': int(len(ids)),
        'n_active_ref': int((a.sum(1) > 0).sum()),
        'n_active_new': int((b.sum(1) > 0).sum()),
        'n_active_both': int(((a.sum(1) > 0) & (b.sum(1) > 0)).sum()),
        'max_rate_only_in_one_run_Hz': float(max(m_a[b.sum(1) == 0].max(initial=0), m_b[a.sum(1) == 0].max(initial=0))),
        'pearson_r_mean_rates': float(np.corrcoef(m_a, m_b)[0, 1]),
        'median_abs_diff_Hz': float(np.median(np.abs(m_b - m_a))),
        'max_abs_diff_Hz': float(np.max(np.abs(m_b - m_a))),
        'frac_abs_z_gt_1.96': float(np.mean(np.abs(z) > 1.96)),
        'perm_T_obs': t_obs,
        'perm_T_null_mean': float(t_null.mean()),
        'perm_T_null_95pct': float(np.quantile(t_null, 0.95)),
        'perm_p': float(p_perm),
        'n_perm': n_perm,
        'n_mw_q_lt_0.05': int(np.sum(q_mw < 0.05)),
    }
    return table, summary, t_null


def psth(df, ids, n_run=N_RUN, t_run=T_RUN, bin_ms=10):
    """Trial-averaged firing rate (Hz) in bins of bin_ms; index = bin centre (ms)."""
    edges = np.arange(0, t_run * 1000 + bin_ms / 2, bin_ms)
    t_ms = df['t'].to_numpy() * 1000
    out = {}
    for i in ids:
        h, _ = np.histogram(t_ms[df['flywire_id'].to_numpy() == i], edges)
        out[i] = h / (n_run * bin_ms / 1000)
    return pd.DataFrame(out, index=edges[:-1] + bin_ms / 2)


def first_spike_latency(df):
    """Median (over trials in which it fired) first-spike time in ms, and fraction of trials with a spike."""
    first = df.groupby(['flywire_id', 'trial'])['t'].min() * 1000
    g = first.groupby(level='flywire_id')
    return pd.DataFrame({'latency_ms': g.median(), 'frac_trials': g.size() / df['trial'].nunique()})


def input_drive(path_con, target, rates, w_syn=0.275, tau_s=5e-3):
    """Mean synaptic drive onto `target` by presynaptic partner: signed synapses * w_syn * rate * tau (mV).

    A mean-field attribution (ignores spike timing); `rates` maps FlyWire ID -> mean rate (Hz).
    Returns partners sorted by |drive| with the cumulative share of the total excitatory drive.
    """
    con = pd.read_parquet(path_con, columns=['Presynaptic_ID', 'Postsynaptic_ID', 'Excitatory x Connectivity'],
                          filters=[('Postsynaptic_ID', '==', target)])
    d = con.set_index('Presynaptic_ID')['Excitatory x Connectivity'].rename('signed_synapses').to_frame()
    d['rate_Hz'] = rates.reindex(d.index).fillna(0).to_numpy()
    d['drive_mV'] = d['signed_synapses'] * w_syn * d['rate_Hz'] * tau_s
    d = d[d['rate_Hz'] > 0].sort_values('drive_mV', key=np.abs, ascending=False)
    d['cum_share_excitatory'] = d['drive_mV'].clip(lower=0).cumsum() / d['drive_mV'].clip(lower=0).sum()
    return d


def hop_distance(path_comp, path_con, sources, excitatory_only=False):
    """Fewest synaptic hops from any neuron in `sources` (FlyWire IDs) to every neuron.

    Returns a Series indexed by FlyWire ID (inf = unreachable). Edges follow the model's
    connectivity table; with excitatory_only, only positive-sign connections are used.
    """
    comp = pd.read_csv(path_comp, index_col=0)
    con = pd.read_parquet(path_con, columns=['Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory'])
    if excitatory_only:
        con = con[con['Excitatory'] > 0]
    n = len(comp)
    adj = csr_matrix((np.ones(len(con), dtype=np.int8), (con['Presynaptic_Index'], con['Postsynaptic_Index'])), shape=(n, n))
    flyid2i = {f: i for i, f in enumerate(comp.index)}
    d = shortest_path(adj, directed=True, unweighted=True, indices=[flyid2i[s] for s in sources])
    return pd.Series(d.min(axis=0), index=comp.index, name='hops')
