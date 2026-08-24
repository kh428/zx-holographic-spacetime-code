"""SCHEDULED foliated block — the verified circuit-level machinery extended
to per-round active-check lists (RG cadences), minimal diff from
foliated_general.build_general_foliated_block.

Same node/edge/fault model (all ZX wires), same downstream pipeline
(single_open_zx -> pyzx webs -> fold_and_classify -> id_simp_decoding).
Detector gap rule (verified to reproduce the uniform builder for pure-Z,
pure-X and Y supports): the web for check c between consecutive ACTIVE
rounds k1 < k2 covers, for each q in supp(c), every data slot t with
first_coupling(k1, q) < t < last_coupling(k2, q), where a round-k coupling
touches slot 2k (Z part) and 2k+1 (X part).

Validation gate: an all-active schedule must reproduce the uniform builder
node-for-node, edge-for-edge, detector-for-detector.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Dict, List, Sequence

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
STAGE = HERE.parent

from .core import foliated_general as FG
from .core import wt02_pipeline as W
from . import e45
from . import e45_bondext as BE


def build_scheduled_block(sx, sz, active: Sequence[Sequence[int]]):
    """Like FG.build_general_foliated_block but round k measures only
    active[k] (sorted check indices)."""
    sx = np.asarray(sx, dtype=np.uint8) % 2
    sz = np.asarray(sz, dtype=np.uint8) % 2
    m, n = sx.shape
    K = len(active)
    sym = (sx @ sz.T + sz @ sx.T) % 2
    assert not sym.any()

    nodes = []
    for k in range(K):
        for q in range(n):
            for dt in (0, 1):
                nodes.append(('data', q, 2 * k + dt))
        for c in active[k]:
            nodes.append(('ancilla', c, 2 * k))
    node_index = {key: i for i, key in enumerate(nodes)}

    edge_set = set()

    def add(u, v):
        e = frozenset((u, v))
        assert e not in edge_set
        edge_set.add(e)

    for q in range(n):
        for t in range(2 * K - 1):
            add(('data', q, t), ('data', q, t + 1))
    B = (sx @ sz.T) % 2
    for k in range(K):
        acts = sorted(active[k])
        for c in acts:
            for q in range(n):
                if sz[c, q]:
                    add(('ancilla', c, 2 * k), ('data', q, 2 * k))
                if sx[c, q]:
                    add(('ancilla', c, 2 * k), ('data', q, 2 * k + 1))
        for ai, c in enumerate(acts):
            for c2 in acts[ai + 1:]:
                if B[c, c2]:
                    add(('ancilla', c, 2 * k), ('ancilla', c2, 2 * k))

    adjacency = {key: set() for key in nodes}
    for e in edge_set:
        u, v = tuple(e)
        adjacency[u].add(v)
        adjacency[v].add(u)

    anc_basis = ['X' if int((sx[c] & sz[c]).sum()) % 2 == 0 else 'Y'
                 for c in range(m)]

    rounds_of = [[k for k in range(K) if c in set(active[k])]
                 for c in range(m)]
    raw_dets = []
    for c in range(m):
        ks = rounds_of[c]
        for j in range(len(ks) - 1):
            k1, k2 = ks[j], ks[j + 1]
            det = [('ancilla', c, 2 * k1), ('ancilla', c, 2 * k2)]
            for q in range(n):
                if not (sx[c, q] or sz[c, q]):
                    continue
                first = 2 * k1 if sz[c, q] else 2 * k1 + 1
                last = 2 * k2 + 1 if sx[c, q] else 2 * k2
                det += [('data', q, t) for t in range(first + 1, last)]
            raw_dets.append({'c': c, 'k': k1, 'nodes': det})
    detectors = sorted(raw_dets, key=lambda d: (d['k'], d['c']))  # FG order
    uniform = all(len(set(a)) == m for a in active)
    if not uniform:
        detectors = None          # scheduled: derived from detecting regions

    return FG.GeneralFoliatedBlock(
        rounds=K, sx=sx, sz=sz, nodes=nodes, node_index=node_index,
        edges=sorted(edge_set, key=lambda e: sorted(node_index[x] for x in e)),
        adjacency=adjacency,
        in_legs=[('data', q, 0) for q in range(n)],
        out_legs=[('data', q, 2 * K - 1) for q in range(n)],
        anc_basis=anc_basis, detectors=detectors)


def rg_schedule(code: Dict, rounds: int) -> List[List[int]]:
    """cadence(check) = 2**(maxr - innermost ring touched); active lists."""
    P = e45.patch_layers(code["n"])
    maxr = int(P["ring"].max())
    nq = code["nq"]
    S = code["code_rows"]
    ring_q = np.array([int(P["ring"][q // 4]) for q in range(nq)])
    cad = []
    for c in range(code["m"]):
        supp = np.flatnonzero(S[c, :nq] | S[c, nq:])
        cad.append(2 ** (maxr - int(ring_q[supp].min())))
    return [[c for c in range(code["m"]) if t % cad[c] == 0]
            for t in range(rounds)], cad


def foliate_and_fold_scheduled(code: Dict, active) -> Dict:
    """Scheduled analogue of st_code.foliate_and_fold with generalized census."""
    blk = build_scheduled_block(code["sx"], code["sz"], active)
    assert all(b == "X" for b in blk.anc_basis)
    g1, vm1, rg1, st1 = W.single_open_zx(blk)
    res = W.fold_and_classify(blk, g1, vm1, rg1, st1, code["reps"],
                              code["central"], code["code_rows"],
                              weight_passes=(2 if blk.detectors is not None
                                             else 0))
    d = res["decomposition"]
    m, k = code["m"], code["k"]
    if blk.detectors is not None:
        exp_per_copy = 2 * len(blk.detectors)
    else:                          # physics cross-check: Σ_c (E_c - 1) per copy
        E = [sum(1 for a in active if c in set(a)) for c in range(m)]
        exp_per_copy = 2 * sum(e - 1 for e in E)
    assert d["per_copy_detectors"] == exp_per_copy, \
        (d["per_copy_detectors"], exp_per_copy)
    assert d["seam_in"] == m and d["seam_out"] == m and d["logical"] == 2 * k
    assert d["total"] == d["d_all"], "closed-web decomposition not exhaustive"
    dec = W.id_simp_decoding(res, W.decoding(res, blk), verbose=False)
    assert dec["n_silent_all"] == 0, "silent wires"
    return dict(blk=blk, res=res, dec=dec)
