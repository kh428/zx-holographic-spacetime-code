"""Assemble deep-block decoding matrices from a labelled K=2 export by
round stacking (seconds, any even K).

Usage: python scripts/assemble_deep.py <n> <K> [data_dir]
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from spacetime.assemble import assemble

n, K = int(sys.argv[1]), int(sys.argv[2])
DATA = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else \
    pathlib.Path(__file__).resolve().parent.parent / 'data' / 'matrices'
H, O, wires, cls = assemble(DATA, n, K)
out = DATA / f'newcon_n{n}_K{K}_stacked_mats.npz'
np.savez_compressed(out, H=H, O=O, n_wires=len(wires))
print(f'n={n} K={K}: {len(wires)} wires | H {H.shape} -> {out}')
