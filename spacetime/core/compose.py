"""Composition of foliated [[5,1,3]] perfect-code blocks by leg joins.

Two join semantics for connecting chain-end (leg) nodes of two blocks:

  'plain' : identify the two chain-end nodes.  In ZX this is a cup / plain
            wire between the two open legs, which fuses the two phase-0
            Z-spiders into one; physically the two blocks share that data
            qubit, and (having no dangling leg) it is X-measured.
  'H'     : CZ between the two chain-end nodes, both X-measured -- the
            chain-extension / fusion primitive of arXiv:2607.13784.  In ZX
            this is a Hadamard edge between the two spiders (equivalently a
            plain join with an H on the joining wire: variants (b) and (c)
            of the gluing semantics are the SAME ZX diagram).

Composites of graph-like blocks stay graph-like (all Z-spiders, all H-edges),
so the graph-state Pauli-web analysis applies unchanged: a closed web /
detecting region is a firing set v (subset of measured nodes) with A v = 0
over GF(2), A the composite adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .foliated_block import FoliatedBlock

GNode = Tuple[str, tuple]   # (block_label, node_key)


@dataclass
class Composite:
    blocks: Dict[str, FoliatedBlock]
    rep: Dict[GNode, GNode]              # raw node -> canonical representative
    nodes: List[GNode]                   # canonical nodes
    node_index: Dict[GNode, int]
    edges: List[FrozenSet[GNode]]        # H-edges between canonical nodes
    open_inputs: List[GNode]             # canonical, ordered (5 per listed leg set)
    open_outputs: List[GNode]
    members: Dict[GNode, List[GNode]]    # canonical -> raw members

    @property
    def open_nodes(self) -> set:
        return set(self.open_inputs) | set(self.open_outputs)


def leg(label: str, key: tuple) -> GNode:
    return (label, key)


def in_legs(label: str, block: FoliatedBlock) -> List[GNode]:
    return [(label, k) for k in block.in_legs]


def out_legs(label: str, block: FoliatedBlock) -> List[GNode]:
    return [(label, k) for k in block.out_legs]


def build_composite(
    blocks: Dict[str, FoliatedBlock],
    plain_joins: Sequence[Tuple[GNode, GNode]] = (),
    h_joins: Sequence[Tuple[GNode, GNode]] = (),
    open_inputs: Sequence[GNode] = (),
    open_outputs: Sequence[GNode] = (),
) -> Composite:
    raw = [(lbl, key) for lbl, b in blocks.items() for key in b.nodes]
    parent = {r: r for r in raw}

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in plain_joins:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    members: Dict[GNode, List[GNode]] = {}
    for r in raw:
        members.setdefault(find(r), []).append(r)
    nodes = list(members.keys())
    node_index = {n: i for i, n in enumerate(nodes)}

    ecount: Dict[FrozenSet[GNode], int] = {}
    for lbl, b in blocks.items():
        for e in b.edges_signed:
            u, v = tuple(e)
            cu, cv = find((lbl, u)), find((lbl, v))
            if cu == cv:
                raise ValueError(f"self-loop after fusion at {cu}")
            k = frozenset((cu, cv))
            ecount[k] = ecount.get(k, 0) + 1
    for a, b in h_joins:
        cu, cv = find(a), find(b)
        if cu == cv:
            raise ValueError(f"H-join self-loop at {cu}")
        k = frozenset((cu, cv))
        ecount[k] = ecount.get(k, 0) + 1
    edges = [k for k, c in ecount.items() if c % 2 == 1]

    return Composite(
        blocks=blocks,
        rep={r: find(r) for r in raw},
        nodes=nodes,
        node_index=node_index,
        edges=edges,
        open_inputs=[find(x) for x in open_inputs],
        open_outputs=[find(x) for x in open_outputs],
        members=members,
    )


# ---------------------------------------------------------------- pyzx graph
def composite_to_zx(
    comp: Composite,
    coords=None,
    pi_phases: Sequence[GNode] = (),
):
    """Render the composite as a graph-like pyzx diagram.

    All interior canonical nodes are X-measured with outcome +1 post-selected
    (plain phase-0 Z-spider = <+| absorbed).  pi_phases lists canonical nodes
    whose measurement outcome is flipped to -1 (phase pi on the spider).

    coords: optional callable canonical-node -> (row, qubit, z) for drawing.
    Returns (graph, vmap).
    """
    import pyzx as zx
    from pyzx.utils import EdgeType, VertexType

    flip = {comp.rep.get(n, n) for n in pi_phases}
    g = zx.Graph()
    vmap: Dict[GNode, int] = {}
    for i, n in enumerate(comp.nodes):
        if coords is not None:
            row, qb, zc = coords(n)
        else:
            lbl, (kind, idx, t) = n
            row, qb, zc = t, (idx if kind == "data" else 6 + idx), 0.0
        v = g.add_vertex(VertexType.Z, qubit=qb, row=row,
                         phase=Fraction(1, 1) if n in flip else 0)
        g.set_vdata(v, "z", zc)
        vmap[n] = v

    for e in comp.edges:
        u, v = tuple(e)
        g.add_edge((vmap[u], vmap[v]), EdgeType.HADAMARD)

    inputs, outputs = [], []
    for n in comp.open_inputs:
        if coords is not None:
            row, qb, zc = coords(n)
            row -= 0.5
        else:
            row, qb, zc = -1, n[1][1], 0.0
        b = g.add_vertex(VertexType.BOUNDARY, qubit=qb, row=row)
        g.set_vdata(b, "z", zc)
        g.add_edge((b, vmap[n]), EdgeType.SIMPLE)
        inputs.append(b)
    for n in comp.open_outputs:
        lbl, (kind, idx, t) = n
        if coords is not None:
            row, qb, zc = coords(n)
            row += 0.5
        else:
            row, qb, zc = t + 1, idx, 0.0
        b = g.add_vertex(VertexType.BOUNDARY, qubit=qb, row=row)
        g.set_vdata(b, "z", zc)
        g.add_edge((vmap[n], b), EdgeType.SIMPLE)
        outputs.append(b)
    g.set_inputs(tuple(inputs))
    g.set_outputs(tuple(outputs))
    return g, vmap


# ---------------------------------------------------------------- GF(2) webs
def gf2_rank(rows) -> int:
    if len(rows) == 0:
        return 0
    m = (np.array(rows, dtype=np.uint8) % 2).copy()
    r = 0
    for c in range(m.shape[1]):
        piv = None
        for i in range(r, m.shape[0]):
            if m[i, c]:
                piv = i
                break
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        for i in range(m.shape[0]):
            if i != r and m[i, c]:
                m[i] ^= m[r]
        r += 1
    return r


def gf2_nullspace(A: np.ndarray) -> List[np.ndarray]:
    """Basis of {v : A v = 0 (mod 2)} for A (rows x cols) over GF(2)."""
    A = (A % 2).astype(np.uint8).copy()
    nrows, ncols = A.shape
    pivots = {}
    r = 0
    for c in range(ncols):
        piv = None
        for i in range(r, nrows):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(nrows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        pivots[c] = r
        r += 1
        if r == nrows:
            break
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for fc in free:
        v = np.zeros(ncols, dtype=np.uint8)
        v[fc] = 1
        for c, pr in pivots.items():
            if A[pr, fc]:
                v[c] = 1
        basis.append(v)
    return basis


def adjacency_matrix(comp: Composite) -> np.ndarray:
    n = len(comp.nodes)
    A = np.zeros((n, n), dtype=np.uint8)
    for e in comp.edges:
        u, v = tuple(e)
        i, j = comp.node_index[u], comp.node_index[v]
        A[i, j] = A[j, i] = 1
    return A


def closed_webs(comp: Composite) -> List[np.ndarray]:
    """Basis of closed X-webs (detecting regions): firing sets v supported on
    interior (measured) nodes with A v = 0 on ALL rows.  Vectors are over the
    full canonical node list."""
    A = adjacency_matrix(comp)
    interior = [i for i, n in enumerate(comp.nodes) if n not in comp.open_nodes]
    basis_small = gf2_nullspace(A[:, interior])
    out = []
    for v in basis_small:
        w = np.zeros(len(comp.nodes), dtype=np.uint8)
        for pos, i in enumerate(interior):
            w[i] = v[pos]
        out.append(w)
    return out


def webs_supported_on(comp: Composite, allowed: set) -> List[np.ndarray]:
    """Closed webs whose firing set lies inside `allowed` (canonical nodes)."""
    A = adjacency_matrix(comp)
    cols = [i for i, n in enumerate(comp.nodes)
            if n in allowed and n not in comp.open_nodes]
    basis_small = gf2_nullspace(A[:, cols])
    out = []
    for v in basis_small:
        w = np.zeros(len(comp.nodes), dtype=np.uint8)
        for pos, i in enumerate(cols):
            w[i] = v[pos]
        out.append(w)
    return out


def pyzx_web_counts(comp: Composite, coords=None) -> Tuple[int, int]:
    """(detecting regions, stabiliser webs) from pyzx.web.compute."""
    from pyzx.web.compute import compute_detecting_regions, compute_stabilisers

    g, _ = composite_to_zx(comp, coords=coords)
    return len(compute_detecting_regions(g)), len(compute_stabilisers(g))


# ---------------------------------------------------------------- contraction
def linear_coords(scheme: Dict[str, Tuple[int, float]]):
    """Contraction-order coordinates: per block label a (sign, shift) so that
    row = sign * t + shift for data nodes (ancillas at their round midpoint).
    tensorfy contracts row-by-row, so rows should follow the unfolded time."""
    def coords(n: GNode):
        lbl, (kind, idx, t) = n
        sgn, sh = scheme[lbl]
        row = sgn * t + sh if kind == "data" else sgn * (t + 0.5) + sh
        qb = idx if kind == "data" else 6 + idx
        return row, qb, 0.0
    return coords


def contract(comp: Composite, pi_phases: Sequence[GNode] = (),
             coords=None) -> np.ndarray:
    """Exact tensor of the composite: matrix 2^{n_out} x 2^{n_in} (pyzx
    convention: first listed output = most significant row bit)."""
    from pyzx.tensor import tensor_to_matrix, tensorfy

    g, _ = composite_to_zx(comp, coords=coords, pi_phases=pi_phases)
    t = tensorfy(g)
    return tensor_to_matrix(t, len(comp.open_inputs), len(comp.open_outputs))
