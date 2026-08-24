"""Foliated [[5,1,3]] perfect-code block (Ustun-Devitt-Saied, arXiv:2607.13784).

Reconstructs the foliated perfect-code cluster state as a reusable "block":
5 data worldlines x 2K sub-layers (t = 0..2K-1) as CZ chains, plus one
ancilla per stabilizer generator per round (4 per round), with

  * round k occupying data sub-layers t = 2k (Z-type couplings) and
    t = 2k+1 (X-type couplings);
  * ancilla(c, 2k) -- data(q, 2k)   edge iff sz[c][q] != 0,
    ancilla(c, 2k) -- data(q, 2k+1) edge iff sx[c][q] != 0;
  * ancilla-ancilla edges within a round for pairs with sx_c . sz_c' odd
    (for the perfect code: all 6 pairs, i.e. K4 per round);
  * all measurements in the X basis;
  * detectors D(c,k) = m_anc(c,2k) - m_anc(c,2k+2)
                       + sum_q sz[c][q] m_data(q,2k+1)
                       - sum_q sx[c][q] m_data(q,2k+2)   (mod d),
    for k = 0..K-2 (4 per adjacent round pair).

Sign conventions (qudit, dimension d; all vanish mod 2 for qubits):
  * generator c (anc ordering) = cyclic right-shift by (3-c) of the signed
    base generator  X^{-1} Z Z^{-1} X I ;
  * chain edge (q,t)-(q,t+1) has weight (-1)^(t+1);
  * ancilla-data edge weight = the corresponding sz/sx entry (+-1);
  * ancilla-ancilla edge weight = -(sx_c . sz_c') mod d.

IN legs = ('data', q, 0), OUT legs = ('data', q, 2K-1).

`to_zx` renders the block as a pyzx graph: Z-spider per node, Hadamard edge
per CZ edge; X-measured nodes either as effects (plain spider, outcome +1
post-selected frame) or with an open boundary leg for web analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

NodeKey = Tuple[str, int, int]   # ('data', q, t) or ('ancilla', c, 2k)

# Signed base generator of the qudit perfect code: X^{-1} Z Z^{-1} X I.
# Entry per qubit position: (pauli, power) with power in {+1, -1, 0}.
_BASE_SIGNED = (("X", -1), ("Z", 1), ("Z", -1), ("X", 1), ("I", 0))


def signed_generators(order: str = "anc") -> List[Tuple[List[int], List[int]]]:
    """Return the 4 signed generators as (sx, sz) integer 5-vectors.

    order='anc'    : ordering used in the arXiv:2607.13784 ancillary data,
                     generator c = right-shift of the base by (3-c).
    order='xzzxi'  : plain cyclic order starting from X^{-1}ZZ^{-1}XI
                     (generator c = right-shift by c); unsigned support is
                     the textbook XZZXI cyclic family.
    """
    if order not in ("anc", "xzzxi"):
        raise ValueError(f"unknown generator order {order!r}")
    gens = []
    for c in range(4):
        shift = (3 - c) % 5 if order == "anc" else c % 5
        sx = [0] * 5
        sz = [0] * 5
        for pos, (pauli, power) in enumerate(_BASE_SIGNED):
            q = (pos + shift) % 5
            if pauli == "X":
                sx[q] = power
            elif pauli == "Z":
                sz[q] = power
        gens.append((sx, sz))
    return gens


@dataclass
class FoliatedBlock:
    """A K-round foliated [[5,1,3]] perfect-code block."""

    rounds: int
    generator_order: str
    generators: List[Tuple[List[int], List[int]]]      # signed (sx, sz) per c
    nodes: List[NodeKey]                               # index -> key (anc ordering)
    node_index: Dict[NodeKey, int]                     # key -> index
    edges_signed: Dict[FrozenSet[NodeKey], int]        # edge -> weight (+-1)
    adjacency: Dict[NodeKey, set]                      # unsigned neighbour sets
    roles: Dict[NodeKey, str]                          # 'in'|'out'|'data'|'ancilla'
    in_legs: List[NodeKey]                             # ('data', q, 0)
    out_legs: List[NodeKey]                            # ('data', q, 2K-1)
    measurements: Dict[NodeKey, str]                   # full foliation: all 'X'
    interior_measurements: List[NodeKey]               # measured when in/out open
    detectors: List[dict]                              # {'c','k','nodes','weights'}

    def signed_adjacency_matrix(self) -> List[List[int]]:
        """Dense signed adjacency in the anc node ordering (matches the
        adjacency_matrix.csv of the ancillary data for rounds=2)."""
        n = len(self.nodes)
        mat = [[0] * n for _ in range(n)]
        for edge, w in self.edges_signed.items():
            u, v = tuple(edge)
            i, j = self.node_index[u], self.node_index[v]
            mat[i][j] = w
            mat[j][i] = w
        return mat


def build_foliated_perfect_block(rounds: int, generator_order: str = "anc") -> FoliatedBlock:
    """Build the K-round foliated [[5,1,3]] block (K = rounds >= 1)."""
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    K = rounds
    gens = signed_generators(generator_order)

    # --- nodes, in the ancillary-data ordering -----------------------------
    nodes: List[NodeKey] = []
    for k in range(K):
        for q in range(5):
            for dt in (0, 1):
                nodes.append(("data", q, 2 * k + dt))
        for c in range(4):
            nodes.append(("ancilla", c, 2 * k))
    node_index = {key: i for i, key in enumerate(nodes)}

    # --- edges -------------------------------------------------------------
    edges_signed: Dict[FrozenSet[NodeKey], int] = {}

    def add_edge(u: NodeKey, v: NodeKey, w: int) -> None:
        if w % 2 == 0:
            return
        e = frozenset((u, v))
        if e in edges_signed:
            raise ValueError(f"duplicate edge {u} {v}")
        edges_signed[e] = w

    # data worldlines: CZ chains with alternating sign (-1)^(t+1)
    for q in range(5):
        for t in range(2 * K - 1):
            add_edge(("data", q, t), ("data", q, t + 1), (-1) ** (t + 1))

    # ancilla-data couplings per round
    for k in range(K):
        for c, (sx, sz) in enumerate(gens):
            anc = ("ancilla", c, 2 * k)
            for q in range(5):
                if sz[q]:
                    add_edge(anc, ("data", q, 2 * k), sz[q])
                if sx[q]:
                    add_edge(anc, ("data", q, 2 * k + 1), sx[q])

    # ancilla-ancilla dressing within a round: weight -(sx_c . sz_c')
    for k in range(K):
        for c in range(4):
            for c2 in range(c + 1, 4):
                sxc, _ = gens[c]
                _, szc2 = gens[c2]
                w = -sum(a * b for a, b in zip(sxc, szc2))
                if w % 2:
                    add_edge(("ancilla", c, 2 * k), ("ancilla", c2, 2 * k), w)

    adjacency: Dict[NodeKey, set] = {key: set() for key in nodes}
    for e in edges_signed:
        u, v = tuple(e)
        adjacency[u].add(v)
        adjacency[v].add(u)

    # --- roles / legs / measurement plan -----------------------------------
    in_legs = [("data", q, 0) for q in range(5)]
    out_legs = [("data", q, 2 * K - 1) for q in range(5)]
    roles: Dict[NodeKey, str] = {}
    for key in nodes:
        kind, _, t = key
        if kind == "ancilla":
            roles[key] = "ancilla"
        elif t == 0:
            roles[key] = "in"
        elif t == 2 * K - 1:
            roles[key] = "out"
        else:
            roles[key] = "data"
    measurements = {key: "X" for key in nodes}
    interior_measurements = [key for key in nodes if roles[key] in ("ancilla", "data")]

    # --- detectors D(c,k), k = 0..K-2 --------------------------------------
    detectors: List[dict] = []
    for k in range(K - 1):
        for c, (sx, sz) in enumerate(gens):
            det_nodes: List[NodeKey] = [("ancilla", c, 2 * k), ("ancilla", c, 2 * k + 2)]
            det_weights: List[int] = [1, -1]
            for q in range(5):          # csv ordering: ancillas first, then by qubit q
                if sz[q]:
                    det_nodes.append(("data", q, 2 * k + 1))
                    det_weights.append(sz[q])
                if sx[q]:
                    det_nodes.append(("data", q, 2 * k + 2))
                    det_weights.append(-sx[q])
            detectors.append({"c": c, "k": k, "nodes": det_nodes, "weights": det_weights})

    return FoliatedBlock(
        rounds=K,
        generator_order=generator_order,
        generators=gens,
        nodes=nodes,
        node_index=node_index,
        edges_signed=edges_signed,
        adjacency=adjacency,
        roles=roles,
        in_legs=in_legs,
        out_legs=out_legs,
        measurements=measurements,
        interior_measurements=interior_measurements,
        detectors=detectors,
    )


def to_zx(
    block: FoliatedBlock,
    open_in: bool = True,
    open_out: bool = True,
    measured_as: str = "effect",
):
    """Render the block as a pyzx graph.

    Z-spider (phase 0) per node, Hadamard edge per CZ edge.

    open_in / open_out : give the 5 in / out chain-end nodes a boundary leg
        (graph inputs / outputs).  When False those nodes are treated like
        the interior measured nodes.
    measured_as : 'effect' -> X-measurement with outcome +1 post-selected,
        i.e. <+| absorbed into the spider (no extra leg; correct for
        detecting-region computation and for post-selected contraction);
        'open' -> each measured node gets its own boundary leg (appended to
        the graph outputs; use for boundary-web / dependency analysis).

    Returns (graph, vmap) with vmap: NodeKey -> pyzx vertex id.
    """
    if measured_as not in ("effect", "open"):
        raise ValueError(f"unknown measured_as {measured_as!r}")
    import pyzx as zx
    from pyzx.utils import EdgeType, VertexType

    g = zx.Graph()
    vmap: Dict[NodeKey, int] = {}
    for key in block.nodes:
        kind, idx, t = key
        qcoord = idx if kind == "data" else 6 + idx      # drawing coordinates
        vmap[key] = g.add_vertex(VertexType.Z, qubit=qcoord, row=t)

    for e in block.edges_signed:
        u, v = tuple(e)
        g.add_edge((vmap[u], vmap[v]), EdgeType.HADAMARD)

    inputs: List[int] = []
    outputs: List[int] = []
    measured = list(block.interior_measurements)
    if open_in:
        for key in block.in_legs:
            b = g.add_vertex(VertexType.BOUNDARY, qubit=key[1], row=-1)
            g.add_edge((b, vmap[key]), EdgeType.SIMPLE)
            inputs.append(b)
    else:
        measured += block.in_legs
    if open_out:
        for key in block.out_legs:
            b = g.add_vertex(VertexType.BOUNDARY, qubit=key[1], row=2 * block.rounds)
            g.add_edge((vmap[key], b), EdgeType.SIMPLE)
            outputs.append(b)
    else:
        measured += block.out_legs

    if measured_as == "open":
        for key in measured:
            kind, idx, t = key
            qcoord = idx if kind == "data" else 6 + idx
            b = g.add_vertex(VertexType.BOUNDARY, qubit=qcoord, row=t + 0.5)
            g.add_edge((vmap[key], b), EdgeType.SIMPLE)
            outputs.append(b)
    # measured_as == 'effect': plain spider == <+| plugged in (outcome +1)

    g.set_inputs(tuple(inputs))
    g.set_outputs(tuple(outputs))
    return g, vmap


def web_fired_nodes(web, vmap: Dict[NodeKey, int]) -> FrozenSet[NodeKey]:
    """Set of block nodes 'fired' by a PauliWeb (X/Y component on the node's
    half-edges).  For a detecting region this is exactly the set of X
    measurements whose outcome product the detector checks."""
    rev = {v: key for key, v in vmap.items()}
    fired = set()
    for (u, _w), pauli in web.half_edges().items():
        if pauli in ("X", "Y") and u in rev:
            fired.add(rev[u])
    return frozenset(fired)
