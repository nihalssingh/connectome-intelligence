"""Stimulus and readout sets of the behaviour battery in v783 root IDs (no MLX import, so regrow.py can use them).

Shiu et al.'s sets come from their figures.ipynb (data/shiu_figures_ids.json, v630 root IDs) and are mapped to v783
with stage1/neurons.REMAP_630_783 (successors by supervoxel ID). Every set is complete in v783 after mapping.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'stage1'))
from neurons import MN9_L, MN9_R, SUGAR_R, to_v783  # noqa: E402

_IDS = json.loads((ROOT / 'data' / 'shiu_figures_ids.json').read_text())
SUGAR = to_v783(SUGAR_R)
BITTER = to_v783(_IDS['neu_bitter'])
IR94E = to_v783(_IDS['neu_ir94e'])
WATER = to_v783(_IDS['neu_water'])
JON_CE = to_v783(_IDS['neu_JON_CE'])
JON_F = to_v783(_IDS['neu_JON_F'])
ABN1 = to_v783(_IDS['aBN1'])[0]
MN9C, MN9I = MN9_L, to_v783([MN9_R])[0]          # contralateral / ipsilateral to the stimulated sugar GRNs
CONDITIONS = {
    'sugar_50': (SUGAR, 50), 'sugar_100': (SUGAR, 100), 'sugar_200': (SUGAR, 200),
    'sugar_bitter_100': (SUGAR + BITTER, 100), 'sugar_ir94e_100': (SUGAR + IR94E, 100),
    'bitter_100': (BITTER, 100), 'water_100': (WATER, 100),
    'JON_CE_100': (JON_CE, 100), 'JON_F_100': (JON_F, 100),
}
STIMULATED = sorted({i for ids, _ in CONDITIONS.values() for i in ids})   # every neuron the battery drives
