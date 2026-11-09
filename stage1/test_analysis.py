"""Checks for the Stage 1 statistics and the input-rate formula.

    .venv/bin/python -m pytest stage1/test_analysis.py -q   # pass/fail guards
    .venv/bin/python stage1/test_analysis.py                # re-measures the false-positive rate and power curve
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import compare_runs  # noqa: E402


def synthetic_run(rng, base, n_trials=30, shift=0.0):
    """Spike-count rates for neurons that share a per-trial input fluctuation (like GRN-driven circuits)."""
    common = rng.normal(0, 0.015, size=n_trials)  # ~1.5% trial-to-trial gain noise shared by all neurons
    lam = np.outer(base * (1 + shift), 1 + common).clip(min=0)
    return pd.DataFrame(rng.poisson(lam).astype(float), index=np.arange(len(base)))


@pytest.fixture(scope='module')
def base():
    return np.random.default_rng(1).lognormal(mean=2.0, sigma=1.5, size=400).clip(0.05, 250)


def test_permutation_test_is_calibrated_under_null(base):
    rng = np.random.default_rng(2)
    p = [compare_runs(synthetic_run(rng, base), synthetic_run(rng, base), n_perm=400, seed=k)[1]['perm_p'] for k in range(120)]
    rejection = np.mean(np.array(p) < 0.05)
    assert 0.0 <= rejection <= 0.12, rejection  # nominal 5%; binomial 95% band for 120 runs is ~1-9%


def test_permutation_test_detects_meaningful_differences(base):
    """Guard the 5% operating point and a pathway-specific change (run this file directly for the power curve)."""
    rng = np.random.default_rng(3)
    p = [compare_runs(synthetic_run(rng, base), synthetic_run(rng, base, shift=-0.05), n_perm=300, seed=k)[1]['perm_p'] for k in range(20)]
    assert np.mean(np.array(p) < 0.05) > 0.9
    p = []
    for k in range(20):
        changed = base.copy()
        changed[rng.choice(len(base), 20, replace=False)] *= 0.8
        p.append(compare_runs(synthetic_run(rng, base), synthetic_run(rng, changed), n_perm=300, seed=k)[1]['perm_p'])
    assert np.mean(np.array(p) < 0.05) > 0.8


def test_input_rate_formula_matches_brian2():
    """A Poisson event landing on the step a neuron spikes is erased by the reset: rate = r / (1 + r*dt)."""
    brian2 = pytest.importorskip('brian2')
    ref = Path(__file__).resolve().parents[1] / 'external' / 'Drosophila_brain_model'
    sys.path.insert(0, str(ref))
    from model import default_params as P
    from report import expected_input_rate
    n, t = 1000, 5 * brian2.second
    neu = brian2.NeuronGroup(n, model=P['eqs'], method='linear', threshold=P['eq_th'], reset=P['eq_rst'],
                             refractory='rfc', namespace=P)
    neu.v = P['v_0']
    neu.rfc = 0 * brian2.ms
    poi = brian2.PoissonInput(target=neu, target_var='v', N=1, rate=200 * brian2.Hz, weight=P['w_syn'] * P['f_poi'])
    mon = brian2.SpikeMonitor(neu)
    brian2.Network(neu, poi, mon).run(t)
    rate = mon.num_spikes / n / 5.0
    assert abs(rate - expected_input_rate(200)) < 4 * np.sqrt(expected_input_rate(200) / (n * 5.0))


def calibration(n_rep=200, n_perm=400):
    """False-positive rate and power of the per-rate permutation test on synthetic runs of 30 trials (400 neurons with
    shared trial-to-trial gain noise), as quoted in the README. Fixed seeds, so the output repeats exactly."""
    base_rates = np.random.default_rng(1).lognormal(mean=2.0, sigma=1.5, size=400).clip(0.05, 250)
    cases = [('no change (false positives)', 0.0, None), ('all rates 1% lower', -0.01, None),
             ('all rates 2% lower', -0.02, None), ('all rates 3% lower', -0.03, None),
             ('all rates 5% lower', -0.05, None), ('5% of neurons 20% lower', 0.0, 20)]
    for k, (label, shift, n_changed) in enumerate(cases):
        rng = np.random.default_rng(100 + k)
        p = []
        for i in range(n_rep):
            other = base_rates.copy()
            if n_changed:
                other[rng.choice(len(base_rates), n_changed, replace=False)] *= 0.8
            p.append(compare_runs(synthetic_run(rng, base_rates), synthetic_run(rng, other, shift=shift),
                                  n_perm=n_perm, seed=i)[1]['perm_p'])
        print(f'{label:30s} p < 0.05 in {np.mean(np.array(p) < 0.05):.0%} of {n_rep} pairs of runs', flush=True)


if __name__ == '__main__':
    calibration()
