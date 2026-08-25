"""Depolarising-noise decoding of a stored spacetime block with BP+OSD.

Noise model: every wire suffers X, Y or Z with probability p/3 each. The
decoder is BP+OSD (product-sum belief propagation, 50 iterations,
combination-sweep OSD) with static channel priors of 2p/3 per column; unlike
the erasure runs, the decoder receives no location information.

Shots are run in batches and written to the output JSON after every batch,
so partial results are always on disk.

Usage: python scripts/run_pauli.py <matrices.npz> [osd_order] [shots_per_batch] [batches]
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

GRID = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02]


def run_batch(H, O, n_wires, p, shots, seed, osd_order):
    import ldpc
    from scipy.sparse import csr_matrix
    q = 2.0 * p / 3.0
    bpd = ldpc.BpOsdDecoder(csr_matrix(H), error_channel=[q] * H.shape[1],
                            bp_method="product_sum", max_iter=50,
                            osd_method="osd_cs", osd_order=osd_order)
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(shots):
        hit = np.flatnonzero(rng.random(n_wires) < p)
        if hit.size == 0:
            continue
        e = np.zeros(H.shape[1], np.uint8)
        pauli = rng.integers(3, size=hit.size)
        for w, pa in zip(hit, pauli):
            if pa in (0, 2):
                e[2 * w] = 1
            if pa in (1, 2):
                e[2 * w + 1] = 1
        syndrome = ((H @ e) % 2).astype(np.uint8)
        correction = np.asarray(bpd.decode(syndrome), dtype=np.uint8)
        if ((O @ (e ^ correction)) % 2).any():
            fails += 1
    return fails


if __name__ == '__main__':
    path = pathlib.Path(sys.argv[1])
    osd_order = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    batches = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    z = np.load(path)
    H = z['H'].astype(np.uint8)
    O = z['O'].astype(np.uint8)
    nw = int(z['n_wires'])
    out = path.with_name(path.stem + '_pauli.json')
    points = json.load(open(out))['points'] if out.exists() else {}
    for b in range(batches):
        for i, p in enumerate(GRID):
            fails = run_batch(H, O, nw, p, shots, 1000 * (b + 1) + i, osd_order)
            a = points.setdefault(str(p), [0, 0])
            a[0] += fails
            a[1] += shots
            print(f'batch {b+1}/{batches} p={p}: +{fails}/{shots} '
                  f'-> {a[0]}/{a[1]}', flush=True)
            json.dump({'points': points}, open(out, 'w'), indent=1)
    print('->', out)
