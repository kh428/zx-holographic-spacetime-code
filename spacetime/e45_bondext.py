"""Algebraic BOND-EXTENDED {4,5} ZX-holographic code (tenet-compliant).

Direct port of src/s74_bondext.py to the r4 tile with the paper's H-edge bond
convention. Qubits = ALL corner leg-ends (4 per tile, n=1: 52). Generators:
  * per-bond H-TWISTED Bell rows X_a Z_b and Z_a X_b (the bond H-edge maps
    X <-> Z across the join) — STABILISERS, never open legs;
  * lifted tile checks: products of per-tile generators whose leg Paulis
    H-agree across every bond (x_a = z_b AND z_a = x_b) = left kernel of the
    twisted mismatch matrix Phi.
Tile groups on own legs (from the paper's r4 stabilisers):
  central (bulk open):  ZZYY, XIXI, IXIX          logical X = IIXX, Z = ZIZX
  gauged  (bulk |+>):   ZZYY, XIXI, IXIX, IIXX
Central logical: the contracted code's central reps carried onto rim leg-ends.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
STAGE = HERE.parent

from . import e45
from .core import wt02_pipeline as W
from .core import wt03_bondext as B3
from .webs_to_checks import code_stabiliser_basis

_TILE = {}


def tile_gens():
    """Per-tile generator strings in THIS builder's slot convention, derived
    by EXHAUSTIVE tensor enumeration of the capped single tile (256 Paulis;
    exact, sign-blind symplectically). X-gauge (|+>) promotes the IZXZ coset;
    ZZZZ belongs to the Z-gauge — the paper's labels are H-swapped on the
    bulk leg relative to this builder. Cached."""
    if _TILE:
        return _TILE
    from itertools import product as _prod
    from . import e45 as _e45
    from pyzx.utils import VertexType as _VT
    X = np.array([[0, 1], [1, 0]]); Z = np.array([[1, 0], [0, -1]])
    Y = np.array([[0, -1j], [1j, 0]]); I2 = np.eye(2)
    PM = {"I": I2, "X": X, "Z": Z, "Y": Y}

    def capped(cap):
        g, _, _ = _e45.build_zx(0, gauge=False)
        b = list(g.inputs())[0]
        g.set_type(b, cap)
        g.set_inputs(())
        psi = g.to_tensor().flatten()
        return psi / np.linalg.norm(psi)

    def group(psi):
        out = set()
        for tup in _prod("IXZY", repeat=4):
            st = "".join(tup)
            M = PM[st[0]]
            for ch in st[1:]:
                M = np.kron(M, PM[ch])
            v = M @ psi
            if np.allclose(v, psi) or np.allclose(v, -psi):
                out.add(st)
        return out

    gx = group(capped(_VT.X))
    gz = group(capped(_VT.Z))
    common = sorted((gx & gz) - {"IIII"},
                    key=lambda t: (4 - t.count("I"), t))
    sp = W.Gf2Span()
    stabs = [t for t in common
             if sp.add(np.concatenate(W.pauli_xz(t)).astype(np.uint8))]
    xbar = sorted(gx - gz, key=lambda t: (4 - t.count("I"), t))[0]
    zbar = sorted(gz - gx, key=lambda t: (4 - t.count("I"), t))[0]
    _TILE.update(central=stabs, gauged=stabs + [xbar],
                 xbar=xbar, zbar=zbar)
    return _TILE


def build(n: int) -> dict:
    P = e45.patch_layers(n)
    T = P["T"]
    nq = 4 * T
    q = lambda i, s: 4 * i + s

    TG = tile_gens()
    rows, names = [], []
    for i in range(T):
        gens = TG["central"] if i == 0 else TG["gauged"]
        for gs in gens:
            x, z = W.pauli_xz(gs)
            r = np.zeros(2 * nq, dtype=np.uint8)
            for leg in range(4):
                r[q(i, leg)] = x[leg]
                r[nq + q(i, leg)] = z[leg]
            rows.append(r)
            names.append((i, gs))
    S_tile = np.array(rows, dtype=np.uint8)

    bonds = []
    for (i, s), j in P["nbr"].items():
        if j >= 0 and j > i:
            s2 = next(s2 for s2 in range(4) if P["nbr"].get((j, s2)) == i)
            bonds.append((q(i, s), q(j, s2)))
    nb = len(bonds)

    # H-twisted mismatch: x_a must equal z_b, z_a must equal x_b
    Phi = np.zeros((S_tile.shape[0], 2 * nb), dtype=np.uint8)
    for k, (a, b) in enumerate(bonds):
        Phi[:, 2 * k] = S_tile[:, a] ^ S_tile[:, nq + b]
        Phi[:, 2 * k + 1] = S_tile[:, nq + a] ^ S_tile[:, b]
    ker = B3.left_nullspace(Phi) if Phi.shape[1] else \
        [np.eye(S_tile.shape[0], dtype=np.uint8)[i]
         for i in range(S_tile.shape[0])]
    lift = np.array([np.bitwise_xor.reduce(S_tile[v.astype(bool)], axis=0)
                     for v in ker], dtype=np.uint8) if len(ker) else \
        np.zeros((0, 2 * nq), dtype=np.uint8)
    for k, (a, b) in enumerate(bonds):
        assert not (lift[:, a] ^ lift[:, nq + b]).any()
        assert not (lift[:, nq + a] ^ lift[:, b]).any()

    bells = []
    for (a, b) in bonds:
        xz = np.zeros(2 * nq, np.uint8); xz[a] = 1; xz[nq + b] = 1
        zx = np.zeros(2 * nq, np.uint8); zx[nq + a] = 1; zx[b] = 1
        bells += [xz, zx]
    bells = np.array(bells, dtype=np.uint8) if bells else \
        np.zeros((0, 2 * nq), dtype=np.uint8)

    S_ext = np.vstack([lift, bells]) if len(bells) else lift
    sym = (S_ext[:, :nq] @ S_ext[:, nq:].T
           + S_ext[:, nq:] @ S_ext[:, :nq].T) % 2
    assert not sym.any(), "S_ext is not abelian"
    span, keep = W.Gf2Span(), []
    for idx, r in enumerate(S_ext):
        if span.add(r.copy()):
            keep.append(idx)
    S_ind = S_ext[keep]
    m = S_ind.shape[0]
    k_log = nq - m

    # central reps from the contracted (gauged) diagram, carried to rim legs
    gV, _, _ = e45.build_zx(n, gauge=True)
    basis, strings = code_stabiliser_basis(gV)
    repsC = W.logical_reps(gV, strings)
    open_slots = [(i, s) for i in range(T) for s in range(4)
                  if P["nbr"].get((i, s), -1) < 0]
    assert len(open_slots) == len(strings[0])
    c = repsC["central"]
    reps = []
    for idx in (2 * c, 2 * c + 1):
        nm, x, z = repsC["reps"][idx]
        r = np.zeros(2 * nq, dtype=np.uint8)
        for jq, (i, s) in enumerate(open_slots):
            r[q(i, s)] = x[jq]
            r[nq + q(i, s)] = z[jq]
        for srow in S_ind:
            assert (int(r[:nq] @ srow[nq:]) + int(r[nq:] @ srow[:nq])) % 2 \
                == 0, f"{nm} does not centralise S_ext"
        assert not span.contains(r.copy()), f"{nm} is a stabiliser"
        reps.append((nm, r[:nq].copy(), r[nq:].copy()))

    return dict(n=n, nq=nq, m=m, k=k_log, S=S_ind, reps=reps, bonds=bonds,
                lift_dim=int(lift.shape[0]), n_bells=int(len(bells)),
                layoutP=P)


def leg_positions(n: int, scale: float = 4.0) -> np.ndarray:
    import math
    P = e45.patch_layers(n)
    T = len(P["cells"])
    xy = P["xy"] * scale
    rho = 0.30 * P["d_edge"] * scale
    pos = np.zeros((4 * T, 2))
    for i in range(T):
        for s in range(4):
            th = 2 * math.pi * (s + 0.5) / 4
            dx, dy = math.cos(th), math.sin(th)
            if s < len(P["slots"][i]):
                dx, dy = P["slots"][i][s]
            pos[4 * i + s] = xy[i] + rho * np.array([dx, dy])
    return pos


def code_for_foliation(n: int) -> dict:
    """Light-first basis + Y-parity gate, foliation-ready (mirrors s74_be_view)."""
    import collections
    c = build(n)
    nq, S, m = c["nq"], c["S"], c["m"]
    order = sorted(range(m), key=lambda i: int((S[i, :nq] | S[i, nq:]).sum()))
    sp = W.Gf2Span()
    rows = [S[i] for i in order if sp.add(S[i].copy())]
    rows = np.array(rows, dtype=np.uint8)
    yc = [int((r[:nq] & r[nq:]).sum()) for r in rows]
    odd = [i for i, y in enumerate(yc) if y % 2]
    repaired = False
    if odd:
        rows = B3._fix_y_parity(rows)
        assert all(int((r[:nq] & r[nq:]).sum()) % 2 == 0 for r in rows)
        repaired = True
    wts = collections.Counter(int((r[:nq] | r[nq:]).sum()) for r in rows)
    return dict(n=n, nq=nq, m=m, k=c["k"], code_rows=rows,
                sx=rows[:, :nq].copy(), sz=rows[:, nq:].copy(),
                reps=c["reps"], central=0, qpos=leg_positions(n),
                n_odd_y=len(odd), y_repaired=repaired,
                weights=dict(sorted(wts.items())), bonds=c["bonds"],
                lift_dim=c["lift_dim"])
