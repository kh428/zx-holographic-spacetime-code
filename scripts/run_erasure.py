"""Erasure-decoding grid (BP+OSD-0) for a stored matrix file.

Usage: python scripts/run_erasure.py <matrices.npz> [shots]
Writes fail counts per erasure rate to stdout and a JSON next to the npz.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from spacetime.simulate import run_point

GRID = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07]
path = pathlib.Path(sys.argv[1])
shots = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
z = np.load(path)
H, O, nw = z['H'], z['O'], int(z['n_wires'])
points = {}
for i, p in enumerate(GRID):
    fails = run_point(H, O, nw, p, shots=shots, seed=1000 + i)
    points[str(p)] = [int(fails), shots]
    print(f'p={p}: {fails}/{shots}', flush=True)
out = path.with_name(path.stem + '_erasure.json')
json.dump({'points': points}, open(out, 'w'), indent=1)
print('->', out)
