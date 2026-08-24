"""{4,5} ZX-holographic code (the paper's Evenbly-style family) — STRUCTURE.

Tile = the r4 tensor of arXiv:2601.04467 eq. (r4): the 4-WHEEL graph state —
hub Z-spider (bulk leg) H-connected to four corner Z-spiders which form an
H-edged 4-cycle; corner legs contract tile-to-tile with the Hadamard edge
RETAINED (paper convention). X-gauge: every non-central bulk leg capped with
a phase-0 X spider (|+>); the central bulk leg stays open.

Tiling: hypertiling SRG kernel (GRG is broken for {4,5}) + hyperbolic-metric
edge adjacency, mirroring src/s74.py. {4,5}: 4 edge-neighbours per tile.
"""
from __future__ import annotations

import math
import pathlib
import sys
from typing import Dict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
STAGE = HERE.parent

from hypertiling import HyperbolicTiling          # noqa: E402


def cells_45(L: int):
    T = HyperbolicTiling(4, 5, L, kernel="SRG")
    xy = np.array([[complex(T.get_center(i)).real,
                    complex(T.get_center(i)).imag]
                   for i in range(len(T))])
    lay = np.array([T.get_layer(i) for i in range(len(T))])
    return xy, lay


def patch_layers(n: int) -> Dict:
    """Cells within n native {4,5} layers (paper convention: N_bnd 4/20/76/284)."""
    xy_all, lay_all = cells_45(n + 2)
    N = len(xy_all)
    z2 = (xy_all ** 2).sum(1)
    diff2 = ((xy_all[:, None, :] - xy_all[None, :, :]) ** 2).sum(-1)
    den = np.maximum((1 - z2)[:, None] * (1 - z2)[None, :], 1e-15)
    dh = np.arccosh(np.maximum(1 + 2 * diff2 / den, 1.0))
    np.fill_diagonal(dh, np.inf)
    d1 = float(np.min(dh))
    order = np.argsort(dh, axis=1)
    edge_nb = []
    for i in range(N):
        e = [int(j) for j in order[i, :4] if dh[i, j] < 1.20 * d1]
        edge_nb.append(e)
    ring = {int(t): int(lay_all[t]) for t in range(N) if lay_all[t] <= n}
    cells = sorted(ring, key=lambda t: (ring[t],
                                        math.atan2(*xy_all[t][::-1])))
    idx = {t: i for i, t in enumerate(cells)}
    T = len(cells)
    xy = xy_all[cells]
    rings = np.array([ring[t] for t in cells])
    # slot assignment: for tile i, its (up to 4) edge-neighbours inside the
    # patch, sorted by direction; remaining slots are RIM (open legs)
    nbr = {}
    slots = []
    for i, t in enumerate(cells):
        dirs = []
        for u in edge_nb[t]:
            j = idx.get(u, -1)
            d = xy_all[u] - xy_all[t]
            th = math.atan2(d[1], d[0])
            dirs.append((th, j, d / max(np.hypot(*d), 1e-12)))
        dirs.sort(key=lambda x: x[0])
        slots.append([tuple(d[2]) for d in dirs])
        for s, (_, j, _) in enumerate(dirs):
            nbr[(i, s)] = j
    n_bond = sum(1 for v in nbr.values() if v >= 0) // 2
    n_rim = sum(1 for v in nbr.values() if v < 0)
    return dict(cells=cells, xy=xy, ring=rings, nbr=nbr, slots=slots, T=T,
                d_edge=d1, n_bond=n_bond, n_rim=n_rim)


def build_zx(n: int, scale: float = 4.0, gauge: bool = True,
             lift_bulk: float = 1.3):
    """The {4,5} ZX-holographic diagram: one 4-wheel per tile, H-bonds.

    Returns (g, P, meta). Central bulk leg = the single input; rim corner
    legs = outputs; non-central bulk legs X-gauged (|+>) when gauge=True,
    else left open (extra inputs). Bulk legs are z-lifted for 3D.
    """
    import pyzx as zx
    from pyzx.utils import EdgeType, VertexType

    P = patch_layers(n)
    T, xy = P["T"], P["xy"] * scale
    rho = 0.30 * P["d_edge"] * scale
    g = zx.Graph()
    hub, corner = [], []
    for i in range(T):
        h = g.add_vertex(VertexType.Z, row=xy[i][0], qubit=xy[i][1])
        hub.append(h)
        cs = []
        for s in range(4):
            th = 2 * math.pi * (s + 0.5) / 4
            dx, dy = math.cos(th), math.sin(th)
            if s < len(P["slots"][i]):
                dx, dy = P["slots"][i][s]
            c = g.add_vertex(VertexType.Z, row=xy[i][0] + rho * dx,
                             qubit=xy[i][1] + rho * dy)
            cs.append(c)
            g.add_edge((h, c), EdgeType.HADAMARD)
        corner.append(cs)
        for a in range(4):                       # corner 4-cycle (wheel rim)
            g.add_edge((cs[a], cs[(a + 1) % 4]), EdgeType.HADAMARD)
    # bonds: corner--corner with RETAINED Hadamard edge (paper convention)
    seen = set()
    for (i, s), j in P["nbr"].items():
        if j < 0 or (j, i) in seen:
            continue
        seen.add((i, j))
        s2 = next(s2 for s2 in range(4) if P["nbr"].get((j, s2)) == i)
        g.add_edge((corner[i][s], corner[j][s2]), EdgeType.HADAMARD)
    # rim legs: open boundary on unmatched corner slots
    outs = []
    for (i, s), j in sorted(P["nbr"].items()):
        if j >= 0:
            continue
        c = corner[i][s]
        b = g.add_vertex(VertexType.BOUNDARY,
                         row=1.55 * (g.row(c) - xy[i][0]) + xy[i][0],
                         qubit=1.55 * (g.qubit(c) - xy[i][1]) + xy[i][1])
        g.add_edge((c, b), EdgeType.SIMPLE)
        outs.append(b)
    # bulk legs
    ins = []
    for i in range(T):
        if i == 0:                               # central: open input, lifted
            b = g.add_vertex(VertexType.BOUNDARY, row=xy[i][0],
                             qubit=xy[i][1])
            g.set_vdata(b, "z", lift_bulk)
            g.add_edge((hub[i], b), EdgeType.SIMPLE)
            ins.append(b)
        elif gauge:                              # |+> cap, sits lower
            x = g.add_vertex(VertexType.X, row=xy[i][0], qubit=xy[i][1])
            g.set_vdata(x, "z", 0.45 * lift_bulk)
            g.add_edge((hub[i], x), EdgeType.SIMPLE)
        else:
            b = g.add_vertex(VertexType.BOUNDARY, row=xy[i][0],
                             qubit=xy[i][1])
            g.set_vdata(b, "z", lift_bulk)
            g.add_edge((hub[i], b), EdgeType.SIMPLE)
            ins.append(b)
    g.set_inputs(tuple(ins))
    g.set_outputs(tuple(outs))
    meta = dict(n=n, T=T, n_bond=P["n_bond"], n_rim=P["n_rim"],
                V=g.num_vertices(), E=g.num_edges())
    return g, P, meta


def boxes(n: int, scale: float = 4.0, lift_bulk: float = 1.1):
    """HONEST purple-box view: one Z_BOX per r4 tensor WITH ALL ITS LEGS —
    H-bond tile-tile edges, open rim stubs, central bulk stub lifted out of
    plane, gauged bulks as short capped stubs."""
    import pyzx as zx
    from pyzx.utils import EdgeType, VertexType

    P = patch_layers(n)
    T, xy = P["T"], P["xy"] * scale
    rho = 0.55 * P["d_edge"] * scale
    G = zx.Graph()
    box = []
    for i in range(T):
        v = G.add_vertex(ty=VertexType.Z_BOX)
        G.set_row(v, xy[i][0])
        G.set_qubit(v, xy[i][1])
        box.append(v)
    seen = set()
    outs, ins = [], []
    for (i, s), j in sorted(P["nbr"].items()):
        if j >= 0:
            if (j, i) not in seen and j > i:
                G.add_edge((box[i], box[j]), EdgeType.HADAMARD)
                seen.add((i, j))
        else:
            dx, dy = P["slots"][i][s] if s < len(P["slots"][i]) else (
                math.cos(2 * math.pi * (s + 0.5) / 4),
                math.sin(2 * math.pi * (s + 0.5) / 4))
            b = G.add_vertex(ty=VertexType.BOUNDARY,
                             row=xy[i][0] + rho * dx,
                             qubit=xy[i][1] + rho * dy)
            G.add_edge((box[i], b), EdgeType.SIMPLE)
            outs.append(b)
    for i in range(T):
        if i == 0:
            b = G.add_vertex(ty=VertexType.BOUNDARY, row=xy[i][0],
                             qubit=xy[i][1] + 0.001)
            G.set_vdata(b, "z", lift_bulk)
            G.add_edge((box[i], b), EdgeType.SIMPLE)
            ins.append(b)
        else:
            x = G.add_vertex(ty=VertexType.X, row=xy[i][0],
                             qubit=xy[i][1] + 0.001)
            G.set_vdata(x, "z", 0.4 * lift_bulk)
            G.add_edge((box[i], x), EdgeType.SIMPLE)
    G.set_inputs(tuple(ins))
    G.set_outputs(tuple(outs))
    return G
