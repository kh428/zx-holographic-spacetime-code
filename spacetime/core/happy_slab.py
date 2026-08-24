"""HaPPY-code structure from foliated [[5,1,3]] slabs (USER IDEA 1).

A *slab* is ONE round of foliation of the [[5,1,3]] perfect code:
``build_foliated_perfect_block(1)`` -- 14 nodes (5 data worldlines x 2
sub-layers + 4 ancillas), the pentagon unit.  Adjacent pentagons of the
HaPPY {5,4} tiling share ONE contracted planar leg; this module lifts that
static contraction to spacetime slabs and exposes the two candidate lifts:

  variant 'shared' : the shared leg's data worldline is ONE common chain
      that both tiles' ancillas couple to.  (A contracted index is one
      wire; sliced into spacetime, one wire = one worldline.)
  variant 'cup'    : cup-join (spider fusion) of the two tiles'
      corresponding in-legs and out-legs only, keeping two worldline
      copies.  For K=1 the two copies' chain CZs land in parallel between
      the same fused pair and cancel (CZ^2 = I / parallel H-edges between
      Z-spiders cancel mod 2): the shared worldline loses its
      time-propagation edge.

Everything is built on src.compose.Composite so the graph-state web
machinery (closed_webs, composite_to_zx) applies unchanged.

Main entry points
-----------------
build_slab_patch(tiles, contractions, variant, rounds=1) -> (Composite, info)
build_two_tile_slab(variant, rounds=1, leg_a=0, leg_b=0) -> (Composite, tiles, info)
build_three_tile_patch(variant, rounds=1) -> (Composite, tiles, info)
embed_composite(comp, tiles, rounds=1, ...) -> (pyzx graph, vmap)
pentagon_layout(...) -> per-tile (cx, cz, rot) placements
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from .compose import Composite, build_composite
from .foliated_block import build_foliated_perfect_block

GNode = Tuple[str, tuple]                      # (tile_label, node_key)
Tiles = Dict[str, Tuple[float, float, float]]  # label -> (cx, cz, rotation)
Contraction = Tuple[str, int, str, int]        # (tileA, legA, tileB, legB)

# default embedding constants (match nb_foliated_513 conventions)
R_DATA, R_ANC, T_SCALE = 1.8, 0.65, 1.5
PHASE_DATA, PHASE_ANC = math.pi / 2, math.pi / 4


def leg_direction(q: int, rot: float) -> float:
    """Angle of data leg q for a tile rotated by rot."""
    return 2 * math.pi * q / 5 + PHASE_DATA + rot


def pentagon_layout(contractions: Sequence[Contraction],
                    root: str, labels: Sequence[str]) -> Tiles:
    """Place tiles so every contracted leg pair meets midway between the two
    tile centers: neighbour center = root center + 2*R_DATA along the root's
    leg direction, neighbour rotated so its own contracted leg points back."""
    tiles: Tiles = {root: (0.0, 0.0, 0.0)}
    remaining = list(contractions)
    while remaining:
        progress = False
        for la, qa, lb, qb in list(remaining):
            if la in tiles and lb not in tiles:
                src_lbl, src_q, dst_lbl, dst_q = la, qa, lb, qb
            elif lb in tiles and la not in tiles:
                src_lbl, src_q, dst_lbl, dst_q = lb, qb, la, qa
            else:
                if la in tiles and lb in tiles:
                    remaining.remove((la, qa, lb, qb))
                    progress = True
                continue
            cx, cz, rot = tiles[src_lbl]
            ang = leg_direction(src_q, rot)
            ncx = cx + 2 * R_DATA * math.cos(ang)
            ncz = cz + 2 * R_DATA * math.sin(ang)
            # rotate the neighbour so its leg dst_q points at angle ang + pi
            nrot = (ang + math.pi) - (2 * math.pi * dst_q / 5 + PHASE_DATA)
            tiles[dst_lbl] = (ncx, ncz, nrot)
            remaining.remove((la, qa, lb, qb))
            progress = True
        if not progress:
            raise ValueError("contraction graph disconnected from root")
    for lbl in labels:
        tiles.setdefault(lbl, (0.0, 0.0, 0.0))
    return tiles


def build_slab_patch(labels: Sequence[str],
                     contractions: Sequence[Contraction],
                     variant: str,
                     rounds: int = 1) -> Tuple[Composite, dict]:
    """Compose one K=`rounds` slab per label, contracting each listed leg pair
    under `variant` ('shared' | 'cup').  Time-boundary legs of every surviving
    worldline are open; in the 'cup' variant the fused chain-end nodes have no
    dangling leg and become interior X-measured nodes (compose semantics).
    """
    if variant not in ("shared", "cup", "wire"):
        raise ValueError(f"unknown variant {variant!r}")
    blocks = {lbl: build_foliated_perfect_block(rounds) for lbl in labels}
    T = 2 * rounds

    used = {}                      # (label, leg) -> contraction partner
    for la, qa, lb, qb in contractions:
        for key in ((la, qa), (lb, qb)):
            if key in used:
                raise ValueError(f"leg {key} contracted twice")
            used[key] = True

    plain_joins: List[Tuple[GNode, GNode]] = []
    h_joins: List[Tuple[GNode, GNode]] = []
    for la, qa, lb, qb in contractions:
        if variant in ("shared", "wire"):
            # identify the two copies at every sub-layer...
            plain_joins += [((la, ("data", qa, t)), (lb, ("data", qb, t)))
                            for t in range(T)]
            if variant == "shared":
                # ...each chain edge now appears twice (parity 0 = cancelled
                # CZ pair); re-add it once so the common chain keeps its
                # single time-step wire: ONE physical worldline.
                h_joins += [((la, ("data", qa, t)), (la, ("data", qa, t + 1)))
                            for t in range(T - 1)]
            # 'wire': leave the chain CZs cancelled -- the fused column is a
            # stack of 2K static ZX contraction wires, one per sub-layer,
            # not a worldline (a virtual index has no time propagation).
        else:  # 'cup'
            plain_joins += [((la, ("data", qa, 0)), (lb, ("data", qb, 0))),
                            ((la, ("data", qa, T - 1)), (lb, ("data", qb, T - 1)))]

    # open time-boundary legs: one boundary per surviving worldline end.
    # 'shared': list only the first tile's copy (canonical rep) of a shared leg.
    # 'cup'   : fused ends have no dangling leg at all.
    skip_in_out = set()
    for la, qa, lb, qb in contractions:
        skip_in_out.add((lb, qb))                    # duplicate canonical rep
        if variant in ("cup", "wire"):
            skip_in_out.add((la, qa))                # fused, interior-measured
    open_inputs = [(lbl, ("data", q, 0)) for lbl in labels for q in range(5)
                   if (lbl, q) not in skip_in_out]
    open_outputs = [(lbl, ("data", q, T - 1)) for lbl in labels for q in range(5)
                    if (lbl, q) not in skip_in_out]

    comp = build_composite(blocks, plain_joins=plain_joins, h_joins=h_joins,
                           open_inputs=open_inputs, open_outputs=open_outputs)

    info = {
        "variant": variant,
        "rounds": rounds,
        "tiles": list(labels),
        "contractions": list(contractions),
        "n_nodes": len(comp.nodes),
        "n_edges": len(comp.edges),
        "n_open_worldlines": len(comp.open_inputs),   # chains with dangling ends
        "n_contracted_legs": len(contractions),
        "n_open": len(comp.open_inputs) + len(comp.open_outputs),
        "n_measured": len(comp.nodes) - len(set(comp.open_inputs) | set(comp.open_outputs)),
    }
    return comp, info


def build_two_tile_slab(variant: str, rounds: int = 1,
                        leg_a: int = 0, leg_b: int = 0):
    """Two adjacent HaPPY pentagons sharing one contracted planar leg,
    lifted to K=`rounds` slabs under `variant`.  Returns (comp, tiles, info)."""
    contractions = [("A", leg_a, "B", leg_b)]
    comp, info = build_slab_patch(["A", "B"], contractions, variant, rounds)
    tiles = pentagon_layout(contractions, "A", ["A", "B"])
    return comp, tiles, info


def build_three_tile_patch(variant: str, rounds: int = 1):
    """Central tile C + neighbours N1 (on C's leg 0) and N2 (on C's leg 1)."""
    contractions = [("C", 0, "N1", 0), ("C", 1, "N2", 0)]
    comp, info = build_slab_patch(["C", "N1", "N2"], contractions, variant, rounds)
    tiles = pentagon_layout(contractions, "C", ["C", "N1", "N2"])
    return comp, tiles, info


# ------------------------------------------------------------------ embedding
def raw_position(raw: GNode, tiles: Tiles):
    """(row, x, z) of a raw (un-fused) node in its tile's pentagonal prism."""
    lbl, (kind, idx, t) = raw
    cx, cz, rot = tiles[lbl]
    if kind == "data":
        ang, r = leg_direction(idx, rot), R_DATA
    else:
        ang, r = 2 * math.pi * idx / 4 + PHASE_ANC + rot, R_ANC
    return (t * T_SCALE, cx + r * math.cos(ang), cz + r * math.sin(ang))


def make_coords(comp: Composite, tiles: Tiles):
    """Canonical-node -> (row, qubit, z); fused nodes sit at the centroid of
    their members (a shared worldline lands midway between the two tiles)."""
    def coords(n):
        mem = comp.members.get(n, [n])
        ps = [raw_position(r, tiles) for r in mem]
        return tuple(sum(c) / len(ps) for c in zip(*ps))
    return coords


def embed_composite(comp: Composite, tiles: Tiles, rounds: int = 1):
    """composite_to_zx with the pentagonal-prism embedding; time-boundary
    boundary-vertices pushed out along the time axis (template convention).
    Returns (graph, vmap: canonical GNode -> vertex id)."""
    from src.compose import composite_to_zx
    g, vmap = composite_to_zx(comp, coords=make_coords(comp, tiles))
    for b in g.inputs():
        g.set_row(b, -T_SCALE)
    for b in g.outputs():
        g.set_row(b, 2 * rounds * T_SCALE)
    return g, vmap


# ------------------------------------------------------------------ web tools
def colour_aware_fired(web, graph, vmap: Dict[GNode, int]):
    """Canonical nodes whose X-measurement joins the web's parity check:
    the web label at the spider anticommutes with the spider's colour basis
    (Z-spider: X/Y fires; X-spider: Z/Y fires)."""
    from pyzx.utils import VertexType
    rev = {v: n for n, v in vmap.items()}
    fired = set()
    for (u, _v), p in web.half_edges().items():
        if u not in rev:
            continue
        t = graph.type(u)
        if (t == VertexType.Z and p in ("X", "Y")) or \
           (t == VertexType.X and p in ("Z", "Y")):
            fired.add(rev[u])
    return frozenset(fired)


def boundary_labels(web, graph):
    """Pauli label of the web on each dangling boundary leg: {boundary
    vertex -> 'X'|'Y'|'Z'} read off the half-edge at the boundary vertex."""
    from pyzx.utils import VertexType
    out = {}
    for (u, v), p in web.half_edges().items():
        if graph.type(u) == VertexType.BOUNDARY:
            out[u] = p
    return out


# ------------------------------------------------- signed stabilizer algebra
# Paulis as (x, z, phase) with operator = i^phase * X^x Z^z (row convention);
# a Hermitian +P Pauli string has phase = (#Y) mod 4, -P adds 2.
import numpy as np


def _row_mul(a, b):
    """(x1,z1,p1) * (x2,z2,p2) in the i^p X^x Z^z convention."""
    x1, z1, p1 = a
    x2, z2, p2 = b
    p = (p1 + p2 + 2 * int(np.sum(z1 & x2) % 2)) % 4
    return x1 ^ x2, z1 ^ z2, p


def pauli_row(n: int, ops: Dict[int, str], sign: int = +1):
    """Signed Pauli string on n qubits: ops maps qubit -> 'X'|'Y'|'Z'."""
    x = np.zeros(n, dtype=np.uint8)
    z = np.zeros(n, dtype=np.uint8)
    p = 0
    for q, o in ops.items():
        if o == "X":
            x[q] = 1
        elif o == "Z":
            z[q] = 1
        elif o == "Y":
            x[q] = z[q] = 1
            p += 1
        elif o != "I":
            raise ValueError(o)
    if sign == -1:
        p += 2
    return x, z, p % 4


def row_transpose(row):
    """P -> P^T (Y picks up a minus sign per Y factor)."""
    x, z, p = row
    return x.copy(), z.copy(), (p + 2 * int(np.sum(x & z) % 2)) % 4


def row_conj_H(row):
    """P -> H^n P H^n (X <-> Z per qubit; Y -> -Y)."""
    x, z, p = row
    return z.copy(), x.copy(), (p + 2 * int(np.sum(x & z) % 2)) % 4


def row_str(row) -> str:
    x, z, p = row
    s = "".join("Y" if xi and zi else "X" if xi else "Z" if zi else "I"
                for xi, zi in zip(x, z))
    pref = {0: "+", 1: "+i", 2: "-", 3: "-i"}[int((p - int(np.sum(x & z))) % 4)]
    return pref + s


def _gf2_solve(rows, target):
    """Coefficients c (uint8) with sum_i c_i rows_i = target over GF(2), or None."""
    if not rows:
        return None if target.any() else np.zeros(0, dtype=np.uint8)
    A = np.array(rows, dtype=np.uint8).T.copy()          # bits x gens
    b = target.astype(np.uint8).copy()
    ng = A.shape[1]
    piv = {}
    r = 0
    for c in range(ng):
        i = next((i for i in range(r, A.shape[0]) if A[i, c]), None)
        if i is None:
            continue
        A[[r, i]] = A[[i, r]]
        b[r], b[i] = b[i], b[r]
        for j in range(A.shape[0]):
            if j != r and A[j, c]:
                A[j] ^= A[r]
                b[j] ^= b[r]
        piv[c] = r
        r += 1
    sol = np.zeros(ng, dtype=np.uint8)
    for c, rr in piv.items():
        sol[c] = b[rr]
    # verify against the original system (also catches inconsistency)
    chk = np.zeros_like(target)
    for i, ci in enumerate(sol):
        if ci:
            chk ^= rows[i]
    return sol if not (chk ^ target).any() else None


def group_contains(gens, target_row):
    """Is the signed Pauli target in <gens>?  Returns (in_support, phase_diff):
    in_support = the unsigned string is a product of generators; phase_diff =
    target_phase - product_phase mod 4 (0 = exact signed membership)."""
    n = len(target_row[0])
    rows = [np.concatenate([x, z]) for x, z, _ in gens]
    sol = _gf2_solve(rows, np.concatenate([target_row[0], target_row[1]]))
    if sol is None:
        return False, None
    prod = (np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8), 0)
    for i, ci in enumerate(sol):
        if ci:
            prod = _row_mul(prod, gens[i])
    return True, int((target_row[2] - prod[2]) % 4)


def _measure_x_plus(gens, v):
    """Project the stabilizer group onto the +1 outcome of X_v, then drop
    qubit v (all rows keep full width; caller slices columns at the end).
    Raises if the diagram post-selects a zero-amplitude outcome."""
    anti = [i for i, (x, z, p) in enumerate(gens) if z[v]]
    n = len(gens[0][0])
    xv = pauli_row(n, {v: "X"})
    if anti:
        a = anti[0]
        for i in anti[1:]:
            gens[i] = _row_mul(gens[i], gens[a])
        gens[a] = xv
    else:
        ok, pd = group_contains(gens, xv)
        if not ok:
            raise ValueError(f"X_{v} outcome neither random nor determined")
        if pd != 0:
            raise ValueError(f"post-selection amplitude 0 at qubit {v}")
        rows = [np.concatenate([x, z]) for x, z, _ in gens]
        sol = _gf2_solve(rows, np.concatenate([xv[0], xv[1]]))
        a = next(i for i, ci in enumerate(sol) if ci)
        gens[a] = xv
    for i, g in enumerate(gens):
        if i != a and g[0][v]:
            gens[i] = _row_mul(g, gens[a])
    del gens[a]
    return gens


def graph_choi_group(comp: Composite):
    """Signed stabilizer group of the composite diagram's boundary state:
    graph state on all canonical nodes (X_v prod Z_neigh, all +), with every
    interior node X-measured and post-selected +1.  Returns (qubits, gens):
    qubits = comp.open_inputs + comp.open_outputs (canonical nodes), gens =
    list of (x, z, phase) rows over those qubits, phase in {0,2} = +/-."""
    order = list(comp.nodes)
    idx = {nd: i for i, nd in enumerate(order)}
    N = len(order)
    adj = np.zeros((N, N), dtype=np.uint8)
    for e in comp.edges:
        u, v = tuple(e)
        adj[idx[u], idx[v]] = adj[idx[v], idx[u]] = 1
    gens = []
    for i in range(N):
        x = np.zeros(N, dtype=np.uint8)
        x[i] = 1
        gens.append((x, adj[i].copy(), 0))
    open_nodes = list(comp.open_inputs) + list(comp.open_outputs)
    interior = [nd for nd in order if nd not in set(open_nodes)]
    for nd in interior:
        gens = _measure_x_plus(gens, idx[nd])
    cols = np.array([idx[nd] for nd in open_nodes])
    out = [(x[cols].copy(), z[cols].copy(), p) for x, z, p in gens]
    assert len(out) == len(open_nodes)
    return open_nodes, out


def flow_target(n_in: int, n_out: int, S_row, in_pos, out_pos, out_row=None):
    """Choi-state membership target for the flow S (in) -> out_row (out)
    [out_row defaults to S]: transpose(S) on the in-leg positions tensor
    out_row on the out-leg positions.  Positions index the Choi qubit list
    (inputs first)."""
    if out_row is None:
        out_row = S_row
    xin, zin, pin = row_transpose(S_row)
    xo, zo, po = out_row
    x = np.zeros(n_in + n_out, dtype=np.uint8)
    z = np.zeros(n_in + n_out, dtype=np.uint8)
    for j, q in enumerate(in_pos):
        x[q], z[q] = xin[j], zin[j]
    for j, q in enumerate(out_pos):
        x[q], z[q] = xo[j], zo[j]
    return x, z, (pin + po) % 4


def boundary_group_ranks(gens, n_in: int):
    """(rank of in-leg-only subgroup, rank of out-leg-only subgroup, total).
    total - r_in - r_out = number of independent genuine flow generators."""
    def subgroup_rank(keep_only_in: bool):
        rows = []
        for x, z, _ in gens:
            rows.append(np.concatenate([x, z]))
        M = np.array(rows, dtype=np.uint8)
        n = len(gens[0][0])
        outside = ([j for j in range(n) if j >= n_in] if keep_only_in
                   else [j for j in range(n) if j < n_in])
        mask = np.array(outside + [n + j for j in outside])
        # rank of {v in rowspace(M) : v zero on mask} = nrows - rank(M[:,mask])
        from src.compose import gf2_rank
        return len(gens) - gf2_rank(M[:, mask])
    return subgroup_rank(True), subgroup_rank(False), len(gens)


# ------------------------------------------- static HaPPY patch (reference)
_P1 = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], dtype=complex),
       "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0 + 0j, -1.0])}


def pauli_matrix(s: str) -> np.ndarray:
    m = np.array([[1.0 + 0j]])
    for c in s:
        m = np.kron(m, _P1[c])
    return m


def row_to_string(row) -> str:
    x, z, _ = row
    return "".join("Y" if xi and zi else "X" if xi else "Z" if zi else "I"
                   for xi, zi in zip(x, z))


def perfect_code_projector() -> Tuple[np.ndarray, List[str]]:
    """(P5, generator strings) of the static [[5,1,3]] code, anc ordering."""
    from src.foliated_block import signed_generators
    strs = []
    for sx, sz in signed_generators("anc"):
        strs.append("".join("Y" if x and z else "X" if x else "Z" if z else "I"
                            for x, z in zip(sx, sz)))
    P = np.eye(32, dtype=complex)
    for s in strs:
        P = P @ (np.eye(32) + pauli_matrix(s)) / 2
    return P, strs


def static_patch_reference(labels: Sequence[str],
                           contractions: Sequence[Contraction]):
    """The STATIC HaPPY patch these slabs should reproduce: contract one
    [[5,1,3]] code-space tensor per tile over the listed leg pairs.

    Returns (P, boundary, static_gens):
      P           : numeric projector onto the patch code space
                    (2^n_boundary square, rank 2^n_tiles),
      boundary    : [(label, leg)] uncontracted legs, row-major in P,
      static_gens : signed (x, z, phase) rows on the boundary -- a full set
                    of independent stabilizer generators of the static code,
                    signs fixed numerically against P (each verified
                    S P = +/- P exactly).
    """
    from src.compose import gf2_nullspace
    P5, gstrs = perfect_code_projector()
    u, s, _ = np.linalg.svd(P5)
    Vc = u[:, :2].reshape([2] * 5 + [2])          # legs q0..q4, logical

    # --- numeric contraction via einsum ---------------------------------
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    leg_letter: Dict[Tuple[str, int], str] = {}
    pos = 0
    for la, qa, lb, qb in contractions:
        leg_letter[(la, qa)] = leg_letter[(lb, qb)] = letters[pos]
        pos += 1
    boundary = [(lbl, q) for lbl in labels for q in range(5)
                if (lbl, q) not in leg_letter]
    for key in boundary:
        leg_letter[key] = letters[pos]
        pos += 1
    log_letter = {lbl: letters[pos + i] for i, lbl in enumerate(labels)}
    subs = ",".join("".join(leg_letter[(lbl, q)] for q in range(5)) + log_letter[lbl]
                    for lbl in labels)
    out = "".join(leg_letter[key] for key in boundary) + \
          "".join(log_letter[lbl] for lbl in labels)
    W = np.einsum(subs + "->" + out, *([Vc] * len(labels)))
    W = W.reshape(2 ** len(boundary), 2 ** len(labels))
    if np.linalg.matrix_rank(W, tol=1e-9) != 2 ** len(labels):
        raise ValueError("static contraction is rank-deficient")
    q, _ = np.linalg.qr(W)
    P = q[:, :2 ** len(labels)] @ q[:, :2 ** len(labels)].conj().T

    # --- stabilizer generators: GF(2) matching over the wires -----------
    grows = [pauli_row(5, {i: c for i, c in enumerate(gs) if c != "I"})
             for gs in gstrs]
    nvar = 4 * len(labels)
    cons = []
    for la, qa, lb, qb in contractions:
        for comp_ in (0, 1):                       # x then z component
            row = np.zeros(nvar, dtype=np.uint8)
            for ci, r in enumerate(grows):
                row[4 * labels.index(la) + ci] ^= r[comp_][qa]
                row[4 * labels.index(lb) + ci] ^= r[comp_][qb]
            cons.append(row)
    static_gens = []
    for cvec in gf2_nullspace(np.array(cons, dtype=np.uint8)):
        per_tile = {}
        for li, lbl in enumerate(labels):
            p = (np.zeros(5, dtype=np.uint8), np.zeros(5, dtype=np.uint8), 0)
            for ci in range(4):
                if cvec[4 * li + ci]:
                    p = _row_mul(p, grows[ci])
            per_tile[lbl] = p
        ops = {}
        for j, (lbl, q) in enumerate(boundary):
            x, z, _ = per_tile[lbl]
            o = "Y" if x[q] and z[q] else "X" if x[q] else "Z" if z[q] else "I"
            if o != "I":
                ops[j] = o
        Smat = pauli_matrix(row_to_string(pauli_row(len(boundary), ops)))
        lam = np.vdot(P, Smat @ P) / np.vdot(P, P)
        if abs(abs(lam) - 1) > 1e-8 or np.abs(Smat @ P - lam * P).max() > 1e-8:
            raise ValueError("matched operator does not stabilize the patch")
        static_gens.append(pauli_row(len(boundary), ops,
                                     sign=+1 if lam.real > 0 else -1))
    return P, boundary, static_gens


def row_matrix(row) -> np.ndarray:
    """Dense matrix of a signed Pauli row (must be Hermitian, i.e. +/-P)."""
    x, z, p = row
    ph = (p - int(np.sum(x & z))) % 4
    if ph not in (0, 2):
        raise ValueError("row is not +/- a Hermitian Pauli string")
    return (1 if ph == 0 else -1) * pauli_matrix(row_to_string(row))


def boundary_positions(comp: Composite, qubits: List[GNode],
                       boundary: Sequence[Tuple[str, int]], rounds: int = 1):
    """Choi-qubit positions (in_pos, out_pos) of the listed (tile, leg)
    worldline ends, for use with flow_target."""
    T = 2 * rounds
    in_pos = [qubits.index(comp.rep[(lbl, ("data", q, 0))]) for lbl, q in boundary]
    out_pos = [qubits.index(comp.rep[(lbl, ("data", q, T - 1))]) for lbl, q in boundary]
    return in_pos, out_pos


def symplectic_commute(r1, r2) -> bool:
    """True iff the two Pauli rows commute."""
    x1, z1, _ = r1
    x2, z2, _ = r2
    return int(np.sum(x1 & z2) + np.sum(z1 & x2)) % 2 == 0


def web_tiles(web, comp: Composite, vmap: Dict[GNode, int]) -> frozenset:
    """Tile labels a web's half-edge support touches (fused nodes count for
    every member tile) -- cross-tile webs witness inter-tile correlation."""
    rev = {v: n for n, v in vmap.items()}
    tiles = set()
    for (u, _v), _p in web.half_edges().items():
        if u in rev:
            for lbl, _key in comp.members.get(rev[u], [rev[u]]):
                tiles.add(lbl)
    return frozenset(tiles)
