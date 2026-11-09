"""Neuron IDs used in the Stage 1 experiments.

All IDs are FlyWire root IDs for materialization v630, copied verbatim from
external/Drosophila_brain_model/figures.ipynb (Shiu et al.).
"""
import pickle
from pathlib import Path

REF = Path(__file__).resolve().parents[1] / 'external' / 'Drosophila_brain_model'

# labellar sugar-sensing gustatory receptor neurons, right hemisphere (figures.ipynb, Fig. 1D)
SUGAR_R = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

# proboscis motor neuron MN9 (figures.ipynb, Fig. 2: "left and right"); Fig. 1 reads out the first
MN9_L = 720575940660219265
MN9_R = 720575940645521262

# second-stimulus sets of Figs. 3a, 3b and 4e (figures.ipynb); at 0 Hz they get no input, but model.poi()
# still sets their refractory period to 0, so those sweeps are exact replicates only if these stay silent
BITTER_R = [
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
    720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
    720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
    720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
]
IR94E_R = [
    720575940614211295, 720575940638218173, 720575940628832256, 720575940626016017, 720575940621375231,
    720575940612920386, 720575940614273292, 720575940628198503, 720575940626241636, 720575940619387814,
    720575940624604560, 720575940615274425, 720575940610683315, 720575940627265265, 720575940624079544,
    720575940629211607, 720575940615089369, 720575940631082124,
]
WATER_R = [
    720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053, 720575940622902535,
    720575940616177458, 720575940660292225, 720575940622486922, 720575940613786774, 720575940629852866,
    720575940625861168, 720575940613996959, 720575940617857694, 720575940644965399, 720575940625203504,
    720575940630553415, 720575940635172191, 720575940634796536,
]

# neurons silenced in the repo's saved example runs (results/example/sugarR-<id>.parquet, 100 Hz)
EXAMPLE_SILENCED = [720575940621754367, 720575940622695448, 720575940617937543]


# v630 -> v783 root IDs for the neurons above that were edited between releases (issue #7 of the
# reference repo, confirmed by anchor-supervoxel matching); all other IDs used here are unchanged
REMAP_630_783 = {720575940620900446: 720575940639259967, MN9_R: 720575940618238523,
                 # successors by supervoxel_id (v1.1.0 -> v3.2.0) for two cells of Shiu's figure sets
                 720575940618600651: 720575940619072513,   # bitter GRN -> LB1c
                 720575940626307902: 720575940625054647}   # JON-CE (JO-EVM) -> JO-EV3


def to_v783(ids):
    return [REMAP_630_783.get(i, i) for i in ids]


def grn_sets_783(seed=0):
    """Stimulus sets on the stimulated side (FlyWire side 'left'), from the GRN retyping in
    flywire_annotations v3.2.0 (Tastekin et al., Cell 2026). v783 root IDs."""
    import numpy as np
    ann = annotations()
    lb = ann[(ann['cell_class'] == 'gustatory') & ann['cell_type'].fillna('').str.startswith('LB') & (ann['side'] == 'left')]
    sugar_class = sorted(lb.index[lb['cell_sub_class'].isin(['sugar', 'sugar/low_salt'])])
    shiu = to_v783(SUGAR_R)
    return {
        'shiu21': shiu,
        'shiu21_sugar_typed': [i for i in shiu if i in set(sugar_class)],
        'shiu21_non_sugar_typed': [i for i in shiu if i not in set(sugar_class)],
        'sugar_class_all': sugar_class,  # LB3b + LB3c
        'high_salt_LB3d': sorted(lb.index[lb['cell_type'] == 'LB3d']),
        'sugar_class_15': sorted(np.random.default_rng(seed).choice(sugar_class, 15, replace=False).tolist()),  # size-matched to LB3d
    }


def sez_types():
    """Named SEZ cell types -> list of root IDs (sez_neurons.pickle shipped with the reference repo)."""
    with open(REF / 'sez_neurons.pickle', 'rb') as f:
        return pickle.load(f)


# Side labels: the repo calls the stimulated GRNs "right" and MN9_L "left", but FlyWire's corrected
# annotations (the FAFB volume was mirrored) put the GRNs on the fly's left and MN9_L on the right.
# Relative to the stimulus, MN9_L is contralateral and MN9_R ipsilateral, as in Nature Fig. 1c.
MN9_LABEL = {MN9_L: 'MN9 contra', MN9_R: 'MN9 ipsi'}

# feeding-circuit names used in Shiu et al., Nature 2024, Fig. 1b (v630 IDs; sez_neurons.pickle has
# older bioRxiv-era names for some, e.g. 'tinctoria' -> MN11)
NATURE_NAMES = {
    720575940629888530: 'Zorro', 720575940620874757: 'G2N-1', 720575940638103349: 'Rattle',
    720575940632648612: 'Usnea', 720575940655014049: 'Clavicle', 720575940614763666: 'FMIn',
    720575940616103218: 'Phantom', 720575940632365905: 'Sternum', 720575940631997032: 'Fdg',
    720575940623211725: 'Roundup', 720575940607272649: 'Roundup', 720575940632252743: 'Roundtree',
    720575940627847752: 'Rounddown', 720575940628826128: 'MN6', 720575940623352063: 'MN8',
    720575940618165019: 'MN11',
}


def id2name():
    """Root ID -> readable label for the neurons above and the named SEZ types."""
    names = {i: f'sugarGRN_{k + 1}' for k, i in enumerate(SUGAR_R)}
    for t, ids in sez_types().items():
        for k, i in enumerate(ids):
            names[i] = t if len(ids) == 1 else f'{t}_{k + 1}'
    names.update(NATURE_NAMES)
    names.update(MN9_LABEL)
    return names


DATA = Path(__file__).resolve().parents[1] / 'data'
# flywire_annotations main branch (v3.2.0, GRNs retyped per Tastekin et al. 2026): root IDs are v783
ANNOTATIONS = DATA / 'flywire_annotations' / 'Supplemental_file1_neuron_annotations.tsv'
# flywire_annotations v1.1.0: root IDs are v630, the release the paper and Stage 1 use
ANNOTATIONS_630 = DATA / 'flywire_annotations_v1.1.0' / 'Supplemental_file1_annotations.tsv'


def annotations(version='783'):
    """FlyWire cell-type annotations (Schlegel et al. 2024 + updates), indexed by root ID of `version`."""
    import pandas as pd
    cols = ['root_id', 'super_class', 'cell_class', 'cell_sub_class', 'cell_type', 'top_nt', 'side']
    path = ANNOTATIONS_630 if version == '630' else ANNOTATIONS
    return pd.read_csv(path, sep='\t', usecols=cols, low_memory=False).set_index('root_id')


def id2label(version='630'):
    """Root ID -> figure label: 'Nature name (cell type)', else the cell type, else nothing.

    The bioRxiv-era names in sez_neurons.pickle are left out (some were renamed in the paper);
    placeholder types that embed a root ID (e.g. 'AN7205759...') count as unnamed.
    """
    ann = annotations(version)
    types = {i: t for i, t in ann['cell_type'].dropna().items() if '72057594' not in str(t)}
    labels = dict(types)
    for i, n in {**NATURE_NAMES, **MN9_LABEL}.items():
        labels[i] = f'{n} ({types[i]})' if i in types else n
    return labels
