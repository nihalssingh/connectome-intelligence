"""Thin runner over the MLX port (Kisame76/drosophila-brain-mlx) for many short whole-brain experiments.

The port's run_exp writes one parquet per experiment; here we need thousands of small conditions, so we call its
engine directly and keep per-trial spike counts. Same model, same constants, same Poisson input: the stimulus is the
port's own core.make_stimulus_for (drive at rate r on the target neurons, no refractoriness for targets), optionally
switched off after `on_s` seconds.
"""
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import numpy as np

from lif import core, engine_fused

ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / 'external' / 'drosophila-brain-mlx' / 'data' / 'pack'


@lru_cache(maxsize=4)
def pack(name='v783'):
    p = core.load_pack(PACKS / name)
    return p


@lru_cache(maxsize=4)
def index(name='v783'):
    p = pack(name)
    return {int(n): i for i, n in enumerate(p.neuron_ids)}


def to_idx(ids, name='v783'):
    ix = index(name)
    out = [ix[int(i)] for i in ids if int(i) in ix]
    return np.array(sorted(set(out)), dtype=np.int32)


def stimulus(p, targets, rate_hz, n_ticks, seed, on_ticks=None):
    """Poisson drive on `targets` (indices) at rate_hz; zero after on_ticks if given."""
    stim = core.make_stimulus_for(p, targets, float(rate_hz), n_ticks, seed)
    if on_ticks is not None and on_ticks < n_ticks:
        d = np.asarray(stim.draws).copy()
        d[on_ticks:] = False
        stim = core.Stimulus(targets=stim.targets, draws=mx.array(d), n_ticks=n_ticks, rate_hz=stim.rate_hz,
                             seed=stim.seed)
    return stim


def run_counts(name, targets, rate_hz, n_trials=10, t_s=1.0, seed=0, silenced=None, edge_split=1,
               on_s=None, record=False):
    """Per-trial spike counts for every neuron: int32[n_trials, N]; with record=True also the events per trial."""
    p = pack(name)
    n_ticks = int(round(t_s * 1e4))
    on_ticks = None if on_s is None else int(round(on_s * 1e4))
    counts = np.zeros((n_trials, p.n_neurons), dtype=np.int32)
    events = []
    for k in range(n_trials):
        stim = stimulus(p, targets, rate_hz, n_ticks, seed + k, on_ticks)
        r = engine_fused.run(p, stim, silenced=silenced, warmup=0, edge_split=edge_split, record=record)
        counts[k] = r.spike_counts
        if record:
            events.append(np.asarray(r.events))
    return (counts, events) if record else counts
