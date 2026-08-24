"""General d=2 foliation of an arbitrary qubit stabilizer code
(Ustun-Devitt-Saied, arXiv:2607.13784, Definition S-1), extending the
verified [[5,1,3]]-specific module `foliated_block.py` to any generator
list given as binary symplectic rows (sx | sz).

Recipe per measurement round k (sub-layers t = 2k, 2k+1):
  * data worldlines: CZ chain (q,t)-(q,t+1) for every qubit q;
  * one ancilla per generator c, coupled to data (q,2k) iff sz[c,q]=1 and
    to (q,2k+1) iff sx[c,q]=1 — a Y-site (sx=sz=1) couples to BOTH;
  * ancilla-ancilla edge within a round iff sx_c . sz_c' is odd
    (symmetric for commuting generators);
  * ancilla measurement basis XZ^{pX.pZ}: plain X iff the generator's
    Y-site count is even, else Y (at d=2);
  * detectors D(c,k), k = 0..K-2:  anc(c,2k), anc(c,2k+2),
    data(q,2k+1) on the Z-support, data(q,2k+2) on the X-support
    (Y-sites appear in both) — pure-X products when all ancilla bases
    are X.

`self_check_against_513(rounds)` rebuilds the [[5,1,3]] foliation from its
unsigned XZZXI generators (in the verified module's 'anc' ordering) and
asserts edge-set equality with `foliated_block.build_foliated_perfect_block`
— anchoring this general builder to the module that was verified against
the paper's ancillary data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

NodeKey = Tuple[str, int, int]  # ('data', q, t) or ('ancilla', c, 2k)


@dataclass
class GeneralFoliatedBlock:
    rounds: int
    sx: np.ndarray                      # (m, n) uint8
    sz: np.ndarray
    nodes: List[NodeKey]
    node_index: Dict[NodeKey, int]
    edges: List[FrozenSet[NodeKey]]
    adjacency: Dict[NodeKey, set]
    in_legs: List[NodeKey]
    out_legs: List[NodeKey]
    anc_basis: List[str]                # 'X' | 'Y' per generator (d=2 rule 3)
    detectors: List[dict]               # {'c','k','nodes'}

    @property
    def n(self) -> int: return self.sx.shape[1]

    @property
    def m(self) -> int: return self.sx.shape[0]


def pauli_strings_to_sxsz(strings: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """['XZZXI', ...] -> (sx, sz) uint8 arrays; 'Y' sets both."""
    m, n = len(strings), len(strings[0])
    sx = np.zeros((m, n), dtype=np.uint8)
    sz = np.zeros((m, n), dtype=np.uint8)
    for c, s in enumerate(strings):
        for q, p in enumerate(s):
            if p in 'XY': sx[c, q] = 1
            if p in 'ZY': sz[c, q] = 1
            if p not in 'IXYZ': raise ValueError(f'bad Pauli {p!r}')
    return sx, sz


def build_general_foliated_block(sx, sz, rounds: int) -> GeneralFoliatedBlock:
    sx = np.asarray(sx, dtype=np.uint8) % 2
    sz = np.asarray(sz, dtype=np.uint8) % 2
    if sx.shape != sz.shape:
        raise ValueError('sx and sz must have equal shapes')
    m, n = sx.shape
    K = rounds
    if K < 1:
        raise ValueError('rounds must be >= 1')
    sym = (sx @ sz.T + sz @ sx.T) % 2
    if sym.any():
        raise ValueError('generators must pairwise commute (symplectic form != 0)')

    nodes: List[NodeKey] = []
    for k in range(K):
        for q in range(n):
            for dt in (0, 1):
                nodes.append(('data', q, 2 * k + dt))
        for c in range(m):
            nodes.append(('ancilla', c, 2 * k))
    node_index = {key: i for i, key in enumerate(nodes)}

    edge_set: set = set()

    def add(u: NodeKey, v: NodeKey) -> None:
        e = frozenset((u, v))
        if e in edge_set:
            raise ValueError(f'duplicate edge {u} {v}')
        edge_set.add(e)

    for q in range(n):                                  # worldline chains
        for t in range(2 * K - 1):
            add(('data', q, t), ('data', q, t + 1))
    for k in range(K):                                  # ancilla-data
        for c in range(m):
            for q in range(n):
                if sz[c, q]: add(('ancilla', c, 2 * k), ('data', q, 2 * k))
                if sx[c, q]: add(('ancilla', c, 2 * k), ('data', q, 2 * k + 1))
    B = (sx @ sz.T) % 2                                 # ancilla-ancilla
    for k in range(K):
        for c in range(m):
            for c2 in range(c + 1, m):
                if B[c, c2]:
                    add(('ancilla', c, 2 * k), ('ancilla', c2, 2 * k))

    adjacency: Dict[NodeKey, set] = {key: set() for key in nodes}
    for e in edge_set:
        u, v = tuple(e)
        adjacency[u].add(v)
        adjacency[v].add(u)

    anc_basis = ['X' if int((sx[c] & sz[c]).sum()) % 2 == 0 else 'Y' for c in range(m)]

    detectors: List[dict] = []
    for k in range(K - 1):
        for c in range(m):
            det = [('ancilla', c, 2 * k), ('ancilla', c, 2 * k + 2)]
            det += [('data', q, 2 * k + 1) for q in range(n) if sz[c, q]]
            det += [('data', q, 2 * k + 2) for q in range(n) if sx[c, q]]
            detectors.append({'c': c, 'k': k, 'nodes': det})

    return GeneralFoliatedBlock(
        rounds=K, sx=sx, sz=sz, nodes=nodes, node_index=node_index,
        edges=sorted(edge_set, key=lambda e: sorted(node_index[x] for x in e)),
        adjacency=adjacency,
        in_legs=[('data', q, 0) for q in range(n)],
        out_legs=[('data', q, 2 * K - 1) for q in range(n)],
        anc_basis=anc_basis, detectors=detectors)


def to_zx(block: GeneralFoliatedBlock, open_in: bool = True, open_out: bool = True):
    """Z-spider per node, Hadamard edge per CZ edge; interior measurements
    absorbed as +1-outcome effects (plain spiders). NOTE: a 'Y'-basis ancilla
    (odd Y-count generator) is represented here by the same plain effect —
    exact only for even-Y-count generator sets; callers must check
    block.anc_basis and caveat otherwise."""
    import pyzx as zx
    from pyzx.utils import EdgeType, VertexType

    g = zx.Graph()
    vmap: Dict[NodeKey, int] = {}
    for key in block.nodes:
        kind, idx, t = key
        vmap[key] = g.add_vertex(VertexType.Z, qubit=idx, row=t)
    for e in block.edges:
        u, v = tuple(e)
        g.add_edge((vmap[u], vmap[v]), EdgeType.HADAMARD)

    inputs, outputs = [], []
    if open_in:
        for key in block.in_legs:
            b = g.add_vertex(VertexType.BOUNDARY, qubit=key[1], row=-1)
            g.add_edge((b, vmap[key]), EdgeType.SIMPLE)
            inputs.append(b)
    if open_out:
        for key in block.out_legs:
            b = g.add_vertex(VertexType.BOUNDARY, qubit=key[1], row=2 * block.rounds)
            g.add_edge((vmap[key], b), EdgeType.SIMPLE)
            outputs.append(b)
    g.set_inputs(tuple(inputs))
    g.set_outputs(tuple(outputs))
    return g, vmap


def embed_prism(g, vmap: Dict[NodeKey, int], block: GeneralFoliatedBlock,
                data_angles: Optional[Sequence[float]] = None,
                r_data: float = 2.4, r_anc: float = 0.9, t_scale: float = 1.5,
                anc_polar: Optional[Dict[int, Tuple[float, float]]] = None) -> None:
    """Prism embedding: data worldlines on a circle at `data_angles`
    (default: uniform n-gon), ancillas on an inner circle, time along row.

    anc_polar: optional {c: (angle, radius)} overriding the uniform inner
    circle — e.g. the holographic placement (support-arc centroid angle,
    radius shrinking with angular spread) from `holographic_anc_polar`."""
    n, m = block.n, block.m
    if data_angles is None:
        data_angles = [2 * math.pi * q / n + math.pi / 2 for q in range(n)]
    for key, v in vmap.items():
        kind, idx, t = key
        if kind == 'data':
            ang, r = data_angles[idx], r_data
        elif anc_polar is not None:
            ang, r = anc_polar[idx]
        else:
            ang, r = 2 * math.pi * idx / m + math.pi / m, r_anc
        g.set_row(v, t * t_scale)
        g.set_qubit(v, r * math.cos(ang))
        g.set_vdata(v, 'z', r * math.sin(ang))
    for b in list(g.inputs()) + list(g.outputs()):
        nb = next(iter(g.neighbors(b)))
        g.set_qubit(b, g.qubit(nb))
        g.set_vdata(b, 'z', g.vdata(nb, 'z', default=0))
        g.set_row(b, -t_scale if b in g.inputs() else 2 * block.rounds * t_scale)


def holographic_anc_polar(block: GeneralFoliatedBlock,
                          data_angles: Sequence[float],
                          r_data: float = 2.4,
                          r_min: float = 0.35,
                          margin: float = 0.6) -> Dict[int, Tuple[float, float]]:
    """Holographic ancilla placement: each ancilla sits at the CIRCULAR MEAN
    angle of its support qubits, at a radius set by the mean resultant length
    Rbar of the support angles (Rbar ~ 1: tight boundary arc -> near the
    boundary; Rbar ~ 0: delocalized support -> deep in the bulk). Checks of
    outer tiles hug the boundary; central-tile checks sink to the center —
    the layout reconstructs bulk depth from boundary support."""
    out: Dict[int, Tuple[float, float]] = {}
    for c in range(block.m):
        supp = [q for q in range(block.n) if block.sx[c, q] or block.sz[c, q]]
        sx_sum = sum(math.cos(data_angles[q]) for q in supp)
        sy_sum = sum(math.sin(data_angles[q]) for q in supp)
        rbar = math.hypot(sx_sum, sy_sum) / max(len(supp), 1)
        ang = math.atan2(sy_sum, sx_sum) if (sx_sum or sy_sum) else 0.0
        out[c] = (ang, r_min + (r_data - margin - r_min) * rbar)
    return out


def self_check_against_513(rounds: int = 2) -> dict:
    """The general builder on the unsigned XZZXI generators (verified 'anc'
    ordering) must reproduce foliated_block.build_foliated_perfect_block
    edge-for-edge."""
    from src.foliated_block import build_foliated_perfect_block, signed_generators
    gens = signed_generators('anc')
    strings = []
    for gsx, gsz in gens:
        strings.append(''.join(
            'Y' if (abs(x) % 2 and abs(z) % 2) else 'X' if abs(x) % 2 else 'Z' if abs(z) % 2 else 'I'
            for x, z in zip(gsx, gsz)))
    sx, sz = pauli_strings_to_sxsz(strings)
    gen_block = build_general_foliated_block(sx, sz, rounds)
    ref = build_foliated_perfect_block(rounds)
    ref_edges = {frozenset(e) for e in ref.edges_signed}
    gen_edges = set(gen_block.edges)
    assert gen_edges == ref_edges, (
        f'edge mismatch: only-general={len(gen_edges - ref_edges)}, '
        f'only-reference={len(ref_edges - gen_edges)}')
    assert gen_block.anc_basis == ['X'] * 4
    det_ref = {frozenset(d['nodes']) for d in ref.detectors}
    det_gen = {frozenset(d['nodes']) for d in gen_block.detectors}
    assert det_gen == det_ref, 'detector node-set mismatch'
    return {'edges': len(gen_edges), 'detectors': len(det_gen), 'match': True}
