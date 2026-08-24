"""Build the bond-extended spacetime decoding matrices from scratch.

Constructs the rim-adapted check basis, foliates it for K rounds with the
scheduled builder, and saves the detector matrix H, the logical correlator O,
and the wire labels needed by the round-stacking assembler.

The fold is pure Python; n=1 takes seconds, n=2 at K=2 takes tens of minutes.
Deeper blocks should be assembled from the K=2 export with
``scripts/assemble_deep.py`` instead of folded directly.

Usage: python scripts/build_new_matrices.py <n> <K> [output_dir]
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from spacetime.rim_basis import rim_basis_code
from spacetime import sched_block as SB

n, K = int(sys.argv[1]), int(sys.argv[2])
OUT = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else \
    pathlib.Path(__file__).resolve().parent.parent / 'data' / 'matrices'

t0 = time.perf_counter()
code2 = rim_basis_code(n)
fb = SB.foliate_and_fold_scheduled(code2, [list(range(code2['m']))] * K)
dec, res = fb['dec'], fb['res']
nw = dec['n_wires_after']
rev = {v: kk for kk, v in res['vmapF'].items()}
labels = [[list(map(str, rev.get(u))) if rev.get(u) else None,
           list(map(str, rev.get(v))) if rev.get(v) else None]
          for (u, v) in dec['wires']]
out = OUT / f'labelled_n{n}_K{K}.npz'
np.savez_compressed(out, H=dec['H_det'], O=dec['O_central'], n_wires=nw)
json.dump(labels, open(out.with_suffix('.json'), 'w'))
print(f'n={n} K={K}: wires={nw} H {dec["H_det"].shape} '
      f'[{time.perf_counter()-t0:.0f}s] -> {out}')
