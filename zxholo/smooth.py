"""Stabiliser-basis smoothing heuristic from the paper.

Algorithm (paper, section on stabiliser row reductions):
  - Up to `passes` iterations (default 8000 per seed).
  - Each iteration: pick the currently heaviest row i.
  - Sample up to `sample_js` (default 1200) other rows j uniformly.
  - If any XOR H_i XOR H_j strictly reduces the weight of row i, apply
    the best such XOR.
  - Terminate early once every row has weight <= `stop_weight`
    (default 10).

`pick_best_smoothed_basis` runs the above across multiple random seeds
and picks the one minimising the triple score:
  (max column-degree on logical support,
   max row weight,
   mean column-degree on logical support)
— which balances BP friendliness against narrow check supports.

Note: the random-subsampling heuristic is load-bearing. A greedy
"lightest partner first" shortcut converges to a different basis and
can change BP LER by ~10%.
"""
from __future__ import annotations

import numpy as np


def smooth_stabiliser_basis(
    H: np.ndarray,
    passes: int = 8000,
    sample_js: int = 1200,
    stop_weight: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """One random-seeded pass of the paper's smoothing heuristic.

    Parameters
    ----------
    H : (m, 2N) binary matrix
        Symplectic stabiliser check matrix, interleaved (x₀, z₀, …) layout.
    passes, sample_js, stop_weight : ints
        As in paper / notebook.
    seed : int
        RNG seed for candidate sampling.

    Returns
    -------
    H_smoothed : (m, 2N) binary matrix in the same row-space as `H`.
    """
    H = (H % 2).astype(np.uint8, copy=True)
    m, _ = H.shape
    rng = np.random.default_rng(seed)

    for _ in range(passes):
        row_w = H.sum(axis=1)
        i = int(np.argmax(row_w))
        wi = int(row_w[i])
        if wi <= stop_weight:
            break

        js = rng.choice(m, size=min(sample_js, m), replace=False)
        best_w = wi
        best_j = None
        ri = H[i].copy()
        for j in js:
            if j == i:
                continue
            w_new = int(np.bitwise_xor(ri, H[j]).sum())
            if w_new < best_w:
                best_w = w_new
                best_j = int(j)

        if best_j is not None:
            H[i] ^= H[best_j]

    return H


def basis_score(H: np.ndarray, lx_arr: np.ndarray):
    """Triple score lower-is-better:
       (max col-degree on logical support,
        max row weight,
        mean col-degree on logical support)."""
    H = (H % 2).astype(np.uint8)
    row_max = int(H.sum(axis=1).max())
    col_deg = H.sum(axis=0)
    Lsupp = np.where(((lx_arr[0] | lx_arr[1]) % 2) == 1)[0]
    if Lsupp.size == 0:
        return (0, row_max, 0.0)
    degL_max = int(col_deg[Lsupp].max())
    degL_mean = float(col_deg[Lsupp].mean())
    return (degL_max, row_max, degL_mean)


def pick_best_smoothed_basis(
    H: np.ndarray,
    lx_arr: np.ndarray,
    seeds=range(20),
    passes: int = 8000,
    sample_js: int = 1200,
    stop_weight: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """Run `smooth_stabiliser_basis` across `seeds` and return the best by
    `basis_score`."""
    best_sc, best_H, best_seed = None, None, None
    for seed in seeds:
        Hs = smooth_stabiliser_basis(
            H, passes=passes, sample_js=sample_js,
            stop_weight=stop_weight, seed=seed,
        )
        sc = basis_score(Hs, lx_arr)
        if verbose:
            print(f"  seed {seed}: score {sc}")
        if best_sc is None or sc < best_sc:
            best_sc, best_H, best_seed = sc, Hs, seed
    if verbose:
        print(f"  BEST seed {best_seed}: score {best_sc}")
    return best_H
