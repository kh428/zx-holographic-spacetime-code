"""Time-folded foliated [[5,1,3]] block (the TIME FOLD, user idea 3).

In the open K-round block the time boundaries are unprotected: detectors
D(c,k) exist only between adjacent round pairs (4(K-1) of them), so the
first and last rounds have no partners.  The fold takes TWO copies of the
block and joins them INPUT-to-INPUT and OUTPUT-to-OUTPUT by explicit seam
wires:

    in_q(copy0)  --seam('in')--   in_q(copy1)     q = 0..4
    out_q(copy0) --seam('out')--  out_q(copy1)    q = 0..4

  join='plain'    : the seam wire is a plain (SIMPLE) ZX wire — the cup /
                    Bell contraction <Phi+|.  Tensor-equal to fusing the two
                    phase-0 Z-spiders, but the wire is kept EXPLICIT so it
                    remains a fault location of the error model.
  join='hadamard' : the seam wire carries a Hadamard — equivalently a CZ
                    between the two X-measured chain ends (the
                    chain-extension primitive of arXiv:2607.13784).

The folded diagram is CLOSED (no open legs; every spider is an X
measurement).  Its closed-web space decomposes (machine-verified in
nb22_time_fold_seams.ipynb) into
  * per-copy detectors D(c,k)  — no seam support,
  * seam-spanning webs         — the truncated boundary detectors,
  * logical LOOPS              — aligned in-in/out-out concatenations of the
                                 open block's boundary webs, OUTSIDE the
                                 detector span (the fold's observables).

Error-model export (`decoding_matrices`, user-pinned convention): fault
locations are the EDGES (wires) of the ZX diagram, one per physical wire
INCLUDING the seam wires; each wire carries independent X- and Z-flip
components.  A fault component is tested against the web label on a FIXED
side of the wire — the half-edge at the LOWER-vertex-id endpoint.  Sliding a
fault through a Hadamard conjugates fault and label equally, so
anticommutation is side-invariant and the fixed choice is safe.

    H[r, (w, X)] = 1  iff  label_r(fixed half of w) in {Z, Y}
    H[r, (w, Z)] = 1  iff  label_r(fixed half of w) in {X, Y}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .foliated_block import FoliatedBlock, NodeKey, build_foliated_perfect_block

FoldKey = Tuple[int, str, int, int]   # (copy, kind, idx, t)


# ---------------------------------------------------------------- structure
@dataclass
class FoldedBlock:
    rounds: int
    join: str                                   # 'plain' | 'hadamard'
    block: FoliatedBlock                        # the single copy
    nodes: List[FoldKey]
    node_index: Dict[FoldKey, int]
    intra_edges: List[Tuple[FoldKey, FoldKey]]  # 2x the block's CZ edges
    seam_edges: List[Tuple[FoldKey, FoldKey, str]]  # (copy0 key, copy1 key, 'in'|'out')


def build_folded(rounds: int = 2, join: str = "plain",
                 generator_order: str = "anc") -> FoldedBlock:
    """Two copies of the K-round block, joined in-in and out-out."""
    if join not in ("plain", "hadamard"):
        raise ValueError(f"unknown join {join!r}")
    block = build_foliated_perfect_block(rounds, generator_order)

    nodes: List[FoldKey] = [(ci, *key) for ci in (0, 1) for key in block.nodes]
    node_index = {key: i for i, key in enumerate(nodes)}

    intra_edges: List[Tuple[FoldKey, FoldKey]] = []
    for ci in (0, 1):
        for e in block.edges_signed:
            u, v = tuple(e)
            intra_edges.append(((ci, *u), (ci, *v)))

    seam_edges: List[Tuple[FoldKey, FoldKey, str]] = []
    for key in block.in_legs:
        seam_edges.append(((0, *key), (1, *key), "in"))
    for key in block.out_legs:
        seam_edges.append(((0, *key), (1, *key), "out"))

    return FoldedBlock(rounds=rounds, join=join, block=block, nodes=nodes,
                       node_index=node_index, intra_edges=intra_edges,
                       seam_edges=seam_edges)


def to_zx_folded(fb: FoldedBlock):
    """Render the fold as a CLOSED pyzx diagram (no boundary vertices).

    Z-spider (phase 0) per node, Hadamard edge per CZ edge; seam wires are
    SIMPLE for join='plain' and HADAMARD for join='hadamard'.
    Returns (graph, vmap) with vmap: FoldKey -> vertex id.
    """
    import pyzx as zx
    from pyzx.utils import EdgeType, VertexType

    g = zx.Graph()
    vmap: Dict[FoldKey, int] = {}
    for key in fb.nodes:
        ci, kind, idx, t = key
        qcoord = (idx if kind == "data" else 6 + idx) + 12 * ci
        vmap[key] = g.add_vertex(VertexType.Z, qubit=qcoord, row=t)
    for u, v in fb.intra_edges:
        g.add_edge((vmap[u], vmap[v]), EdgeType.HADAMARD)
    seam_type = EdgeType.SIMPLE if fb.join == "plain" else EdgeType.HADAMARD
    for u, v, _s in fb.seam_edges:
        g.add_edge((vmap[u], vmap[v]), seam_type)
    g.set_inputs(())
    g.set_outputs(())
    return g, vmap


def embed_fold_prisms(g, vmap: Dict[FoldKey, int], r_data: float = 1.8,
                      r_anc: float = 0.65, t_scale: float = 1.5,
                      dx: float = 6.0) -> None:
    """Pentagonal-prism embedding, one prism per copy, offset by dx in the
    qubit direction; seam wires become lateral chords at the prism ends."""
    for key, v in vmap.items():
        ci, kind, idx, t = key
        if kind == "data":
            ang, r = 2 * math.pi * idx / 5 + math.pi / 2, r_data
        else:
            ang, r = 2 * math.pi * idx / 4 + math.pi / 4, r_anc
        g.set_row(v, t * t_scale)
        g.set_qubit(v, r * math.cos(ang) + ci * dx)
        g.set_vdata(v, "z", r * math.sin(ang))


# ---------------------------------------------------------------- GF(2)
def gf2_rref(M: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Reduced row echelon form over GF(2); returns (R, pivot column list)."""
    R = (np.asarray(M, dtype=np.uint8) % 2).copy()
    if R.size == 0:
        return R, []
    pivots: List[int] = []
    r = 0
    for c in range(R.shape[1]):
        piv = next((i for i in range(r, R.shape[0]) if R[i, c]), None)
        if piv is None:
            continue
        R[[r, piv]] = R[[piv, r]]
        for i in range(R.shape[0]):
            if i != r and R[i, c]:
                R[i] ^= R[r]
        pivots.append(c)
        r += 1
        if r == R.shape[0]:
            break
    return R, pivots


def gf2_rank(M: np.ndarray) -> int:
    return len(gf2_rref(M)[1])


def gf2_in_span(B: np.ndarray, v: np.ndarray) -> bool:
    if B.size == 0:
        return not np.any(np.asarray(v) % 2)
    return gf2_rank(np.vstack([B, v])) == gf2_rank(B)


def gf2_nullspace(A: np.ndarray) -> List[np.ndarray]:
    """Basis of {v : A v = 0 (mod 2)}."""
    A = (np.asarray(A, dtype=np.uint8) % 2)
    nrows, ncols = A.shape
    R, pivots = gf2_rref(A)
    pivot_row = {c: i for i, c in enumerate(pivots)}
    free = [c for c in range(ncols) if c not in pivot_row]
    basis = []
    for fc in free:
        v = np.zeros(ncols, dtype=np.uint8)
        v[fc] = 1
        for c, pr in pivot_row.items():
            if pr < R.shape[0] and R[pr, fc]:
                v[c] = 1
        basis.append(v)
    return basis


def gf2_left_nullspace(M: np.ndarray) -> List[np.ndarray]:
    """Basis of {lam : lam M = 0 (mod 2)}."""
    return gf2_nullspace(np.asarray(M, dtype=np.uint8).T)


# ---------------------------------------------------------------- webs
def closed_web_basis(graph):
    """Basis of the closed-web (detecting-region) space of `graph`.

    pyzx's compute_detecting_regions returns [] on a diagram with NO
    boundary vertices: its boundary-restriction step takes the nullspace of
    a 0-row boundary matrix to be empty, when for a closed diagram EVERY
    valid firing assignment is already a detecting region.  For diagrams
    with boundary we defer to pyzx; for closed diagrams we enumerate the
    firing-assignment solutions directly with the same pyzx internals.
    """
    if graph.num_inputs() + graph.num_outputs() > 0:
        from pyzx.web.compute import compute_detecting_regions
        return compute_detecting_regions(graph)

    from pyzx.web.compute import (convert_firing_assignment_to_web_prototype,
                                  create_firing_verification,
                                  determine_ordering, to_red_green_form)

    g = graph.clone()                      # clone preserves vertex ids
    additional = to_red_green_form(g)
    ordering = determine_ordering(g)
    sols = create_firing_verification(g, ordering).nullspace()
    webs = [convert_firing_assignment_to_web_prototype(g, ordering, v)
            for v in sols]
    for w in webs:
        additional.remove_from(g, w)
        w.g = graph
    return webs


def fire_web(g, vertices: Sequence[int]):
    """Vertex-firing web: X on every half-edge at each fired vertex (the far
    half picks up the edge-type conjugate).  For a graph state this is the
    product of graph-state stabilisers K_v over the fired set."""
    from pyzx.pauliweb import PauliWeb

    w = PauliWeb(g)
    for v in vertices:
        for nb in g.neighbors(v):
            w.add_edge((v, nb), "X")
    return w


def canonical_detector_webs(fb: FoldedBlock, g, vmap: Dict[FoldKey, int]):
    """The 2 x 4(K-1) per-copy detectors D(c,k) as firing webs of the fold."""
    out = []
    for ci in (0, 1):
        for det in fb.block.detectors:
            vs = [vmap[(ci, *k)] for k in det["nodes"]]
            out.append((f"copy{ci}:D(c={det['c']},k={det['k']})",
                        fire_web(g, vs)))
    return out


def web_is_closed(g, web) -> bool:
    """Closure check: (ii) the two halves of every edge agree (H-conjugate
    across a Hadamard edge); (i) at every phase-0 spider the labels form a
    (anti-)stabiliser: the fire component (X for a Z-spider, Z for an
    X-spider) is all-or-nothing across the legs, and if absent the passing
    component has even parity.  Boundary vertices are exempt from (i)."""
    from pyzx.pauliweb import h_pauli
    from pyzx.utils import EdgeType, VertexType

    es = web.half_edges()
    for e in g.edges():
        s, t = g.edge_st(e)
        ps, pt = es.get((s, t), "I"), es.get((t, s), "I")
        expect = h_pauli(ps) if g.edge_type(e) == EdgeType.HADAMARD else ps
        if pt != expect:
            return False
    for v in g.vertices():
        ty = g.type(v)
        if ty == VertexType.BOUNDARY:
            continue
        if g.phase(v) != 0:
            raise ValueError(f"web_is_closed assumes phase-0 spiders, got {g.phase(v)} at {v}")
        labels = [es.get((v, nb), "I") for nb in g.neighbors(v)]
        if not labels:
            continue
        fire, passing = ("X", "Z") if ty == VertexType.Z else ("Z", "X")
        fired = [p in (fire, "Y") for p in labels]
        if any(fired) and not all(fired):
            return False
        if not any(fired) and sum(p in (passing, "Y") for p in labels) % 2:
            return False
    return True


def combine_webs(g, webs, coeffs):
    """GF(2) combination of PauliWebs (labels compose via Pauli multiplication)."""
    from pyzx.pauliweb import PauliWeb

    w = PauliWeb(g)
    for wi, c in zip(webs, coeffs):
        if int(c) % 2:
            for he, p in wi.half_edges().items():
                w.add_half_edge(he, p)
    return w


def greedy_weight_reduce(webs, helpers, passes: int = 4):
    """Lower each web's labelled-half-edge count by multiplying with webs
    from `helpers` and with the OTHER entries of `webs` — pure GF(2) row
    operations, so span and linear independence are preserved.  Returns a
    new list (inputs untouched)."""
    cur = [w.copy() for w in webs]
    for _ in range(passes):
        changed = False
        for i in range(len(cur)):
            pool = list(helpers) + [x for j, x in enumerate(cur) if j != i]
            for h in pool:
                trial = cur[i] * h
                if len(trial.half_edges()) < len(cur[i].half_edges()):
                    cur[i] = trial
                    changed = True
        if not changed:
            break
    return cur


# ------------------------------------------------- half-edge vectorisation
def sorted_wires(g) -> List[Tuple[int, int]]:
    """All wires of g as (min_id, max_id) pairs, sorted — the canonical wire
    list; the FIXED side of wire (u, v) is the half-edge (u, v), u < v."""
    return sorted((min(s, t), max(s, t))
                  for s, t in (g.edge_st(e) for e in g.edges()))


def half_edge_cols(g) -> Dict[Tuple[int, int], int]:
    """Column base per half-edge: 4 columns per wire
    [(u,v).x, (u,v).z, (v,u).x, (v,u).z] in sorted_wires order."""
    cols: Dict[Tuple[int, int], int] = {}
    for i, (u, v) in enumerate(sorted_wires(g)):
        cols[(u, v)] = 4 * i
        cols[(v, u)] = 4 * i + 2
    return cols


def web_vec(web, cols: Dict[Tuple[int, int], int]) -> np.ndarray:
    """Web as a GF(2) vector over (half-edge) x (X, Z components)."""
    vec = np.zeros(2 * len(cols), dtype=np.uint8)
    for he, p in web.half_edges().items():
        base = cols[he]
        if p in ("X", "Y"):
            vec[base] = 1
        if p in ("Z", "Y"):
            vec[base + 1] = 1
    return vec


def boundary_wire_cols(g, cols: Dict[Tuple[int, int], int],
                       boundaries: Sequence[int]) -> List[int]:
    """All vector columns living on the leg wires of the given boundary
    vertices (both halves, both components)."""
    out: List[int] = []
    for b in boundaries:
        nb = next(iter(g.neighbors(b)))
        for he in ((b, nb), (nb, b)):
            out.extend((cols[he], cols[he] + 1))
    return out


def boundary_pauli_string(g, web, boundaries: Sequence[int]) -> str:
    """Web labels on the interior halves of the given boundary legs."""
    es = web.half_edges()
    chars = []
    for b in boundaries:
        nb = next(iter(g.neighbors(b)))
        chars.append(es.get((nb, b), "I"))
    return "".join(chars)


# ------------------------------------------------------------- aligned webs
def aligned_fold_web(fb: FoldedBlock, g_fold, vmap_fold: Dict[FoldKey, int],
                     g_single, vmap_single: Dict[NodeKey, int], web):
    """Concatenate an open web of the single block with its aligned copy:
    the SAME web is placed in both copies; on each seam wire the half-edge at
    copy ci inherits the web's label on the interior half of the
    corresponding boundary leg.  Closure of the result depends on the join
    (checked by web_is_closed, not assumed)."""
    from pyzx.pauliweb import PauliWeb

    rev = {v: k for k, v in vmap_single.items()}
    bnd = set(g_single.inputs()) | set(g_single.outputs())
    w = PauliWeb(g_fold)
    for (u, v), p in web.half_edges().items():
        if u in bnd:
            continue                       # the leg's outer half has no fold image
        ku = rev[u]
        for ci in (0, 1):
            fu = vmap_fold[(ci, *ku)]
            fv = (vmap_fold[(1 - ci, *ku)] if v in bnd
                  else vmap_fold[(ci, *rev[v])])
            w.add_half_edge((fu, fv), p)
    return w


# --------------------------------------------------------- decoding export
def decoding_matrices(g, det_webs, log_webs):
    """Check matrix H_det and observable matrix O per the pinned error model.

    Columns: 2 per wire of `sorted_wires(g)` — (wire, X-flip) then
    (wire, Z-flip).  A fault component is tested against the web label on the
    FIXED half-edge (u, v) with u < v: H-edge sliding conjugates fault and
    label equally, so anticommutation is side-invariant.

        H[r, 2i]   = 1  iff  label_r((u,v)) in {Z, Y}   (X-flip anticommutes)
        H[r, 2i+1] = 1  iff  label_r((u,v)) in {X, Y}   (Z-flip anticommutes)

    Returns (wires, H_det, O).
    """
    wires = sorted_wires(g)

    def rows(webs):
        H = np.zeros((len(webs), 2 * len(wires)), dtype=np.uint8)
        for r, w in enumerate(webs):
            es = w.half_edges()
            for i, (u, v) in enumerate(wires):
                p = es.get((u, v), "I")
                if p in ("Z", "Y"):
                    H[r, 2 * i] = 1
                if p in ("X", "Y"):
                    H[r, 2 * i + 1] = 1
        return H

    return wires, rows(det_webs), rows(log_webs)


def wire_table(fb: FoldedBlock, g, vmap_fold: Dict[FoldKey, int]):
    """Per-wire metadata rows aligned with sorted_wires(g):
    (key_u, key_v, edge_type 'H'|'S', seam ''|'in'|'out')."""
    from pyzx.utils import EdgeType

    rev = {v: k for k, v in vmap_fold.items()}
    seam_of = {frozenset((vmap_fold[u], vmap_fold[v])): s
               for u, v, s in fb.seam_edges}

    def key_str(k: FoldKey) -> str:
        ci, kind, idx, t = k
        return f"c{ci}.{'d' if kind == 'data' else 'a'}{idx}.t{t}"

    out = []
    for u, v in sorted_wires(g):
        et = g.edge_type(g.edge(u, v))
        out.append((key_str(rev[u]), key_str(rev[v]),
                    "H" if et == EdgeType.HADAMARD else "S",
                    seam_of.get(frozenset((u, v)), "")))
    return out
