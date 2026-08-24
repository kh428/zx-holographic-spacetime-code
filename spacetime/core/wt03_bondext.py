"""Radius-general BOND-EXTENDED HaPPY code: every leg-end is a qubit.

The contracted code of `wt02_pipeline.contracted_code(R)` has all of its
qubits on the boundary -- contraction identifies the two legs of each internal
bond and eliminates them.  Here the bonds stay physical: the code lives on all
5k leg-ends (k = number of tiles), the bond sector is held by per-bond Bell
checks, and the tile checks are LIFTED so that the whole set commutes.

    R = 0 -> [[5, 1]]      1 tile,   0 bonds
    R = 1 -> [[30, 6]]     6 tiles,  5 bonds
    R = 2 -> [[105, 21]]  21 tiles, 25 bonds

Why the tile checks must be lifted.  A bare [[5,1,3]] generator of one tile
acts on a bond leg with some Pauli P and on the partner leg (owned by the
neighbouring tile) with nothing; the bond's XX and ZZ checks then fail to
commute with it unless P's x- and z-parts match across the bond.  Collecting
that failure into a MISMATCH matrix Phi (one column pair per bond) makes the
fix linear: the Bell-compatible subgroup is exactly the left kernel of Phi,
and its elements -- products of bare tile generators spanning several tiles --
are the lifted checks.  This is the r=1/r=2 construction of
`ft_internal_edges/src/{bond_extended,radius2}.py` with the hard-coded sizes
removed; the leg frame is only fixed up to a rotation, which is harmless
because the [[5,1,3]] code is cyclic.

Everything is derived from the encoder V and machine-checked: the leg census
5k = 2*bonds + boundary, pairwise commutation of the extended check set, its
rank (giving [[5k, k]]), even Y-counts (Def. S-1 rule 3), and the fact that
the contracted code's logical operators remain logical here.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import folded_block as fold
from . import wt02_pipeline as W


# ------------------------------------------------------------ GF(2) helpers
def gf2_nullspace(M: np.ndarray) -> List[np.ndarray]:
    """Basis of {v : M v = 0} over GF(2)."""
    M = np.asarray(M, dtype=np.uint8).copy()
    rows, cols = M.shape
    piv: Dict[int, int] = {}
    r = 0
    for c in range(cols):
        nz = np.flatnonzero(M[r:, c])
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            M[[r, i]] = M[[i, r]]
        hit = np.flatnonzero(M[:, c])
        hit = hit[hit != r]
        if hit.size:
            M[hit] ^= M[r]
        piv[c] = r
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for c, rr in piv.items():
            v[c] = M[rr, f]
        basis.append(v)
    return basis


def left_nullspace(M: np.ndarray) -> List[np.ndarray]:
    return gf2_nullspace(np.asarray(M, dtype=np.uint8).T)


def y_count(row: np.ndarray) -> int:
    n = len(row) // 2
    return int((row[:n] & row[n:]).sum())


def row_weight(row: np.ndarray) -> int:
    n = len(row) // 2
    return int((row[:n] | row[n:]).sum())


def row_to_pauli(row: np.ndarray) -> str:
    n = len(row) // 2
    return "".join("Y" if row[i] and row[n + i] else "X" if row[i]
                   else "Z" if row[n + i] else "I" for i in range(n))


def sym_pair_rows(a: np.ndarray, b: np.ndarray) -> int:
    n = len(a) // 2
    return int((a[:n] @ b[n:] + a[n:] @ b[:n]) % 2)


# ------------------------------------------------------- 1. tiling geometry
def _pentagon_cycle(V, centre: int, legs: Sequence[int]) -> List[int]:
    """Angle-sorted cyclic order of a tile's 5 legs, asserted consistent with
    the pentagon's own edges."""
    cx, cy = V.qubit(centre), V.row(centre)
    order = sorted(legs, key=lambda l: math.atan2(V.row(l) - cy,
                                                  V.qubit(l) - cx))
    legset = set(legs)
    for i, l in enumerate(order):
        mates = set(V.neighbors(l)) & legset
        assert len(mates) == 2 and order[(i + 1) % 5] in mates, \
            "angle order inconsistent with the pentagon edges"
    return order


def derive_geometry(V) -> Dict:
    """Tiles, legs, bonds and boundary legs of the radius-R patch, read off
    the encoder.  Radius-general: nothing here knows R."""
    from pyzx.utils import EdgeType

    ins, outs = list(V.inputs()), list(V.outputs())
    k = len(ins)

    centre_of, legs_of = {}, {}
    for b in ins:
        nbs = list(V.neighbors(b))
        assert len(nbs) == 1, "a bulk leg should touch exactly one tile"
        centre_of[b] = nbs[0]
        legs_of[b] = [w for w in V.neighbors(nbs[0]) if w != b]
        assert len(legs_of[b]) == 5, "a tile should have five planar legs"
    tile_of_leg = {l: b for b in ins for l in legs_of[b]}

    outset = set(outs)
    bonds_raw: List[Tuple[int, int]] = []
    bnd_of_leg: Dict[int, int] = {}
    for b in ins:
        legset = set(legs_of[b])
        for l in legs_of[b]:
            for w in V.neighbors(l):
                if w == centre_of[b] or w in legset:
                    continue
                if w in outset:
                    assert V.edge_type(V.edge(l, w)) == EdgeType.SIMPLE
                    bnd_of_leg[l] = w
                elif w in tile_of_leg:
                    assert V.edge_type(V.edge(l, w)) == EdgeType.SIMPLE
                    if l < w:
                        bonds_raw.append((l, w))
                else:
                    raise AssertionError(f"unclassifiable leg neighbour {w}")
    assert 5 * k == 2 * len(bonds_raw) + len(bnd_of_leg), \
        f"leg census: 5*{k} != 2*{len(bonds_raw)} + {len(bnd_of_leg)}"

    # ---- central tile, adjacency, depth ---------------------------------
    opos = np.array([[float(V.qubit(next(iter(V.neighbors(o))))),
                      float(V.row(next(iter(V.neighbors(o)))))] for o in outs])
    centroid = opos.mean(axis=0)
    bulk_pos = {b: np.array([float(V.qubit(centre_of[b])),
                             float(V.row(centre_of[b]))]) for b in ins}
    dists = {b: float(np.linalg.norm(bulk_pos[b] - centroid)) for b in ins}
    central_in = min(ins, key=lambda b: dists[b])

    adj = defaultdict(set)
    for l, w in bonds_raw:
        adj[tile_of_leg[l]].add(tile_of_leg[w])
        adj[tile_of_leg[w]].add(tile_of_leg[l])
    depth = {central_in: 0}
    frontier = [central_in]
    while frontier:
        nxt = []
        for t in frontier:
            for u in adj[t]:
                if u not in depth:
                    depth[u] = depth[t] + 1
                    nxt.append(u)
        frontier = nxt
    assert len(depth) == k, "the tiling is not connected through its bonds"

    def partner(leg: int) -> Optional[int]:
        for l, w in bonds_raw:
            if l == leg:
                return w
            if w == leg:
                return l
        return None

    # ---- deterministic tile order and leg frames ------------------------
    # tiles by (depth, angle); leg 0 = the bond toward the shallowest
    # neighbour (ties by angle).  The frame is fixed only up to rotation,
    # which does not matter: the [[5,1,3]] code is cyclic.
    ang_of = {b: math.atan2(bulk_pos[b][1] - centroid[1],
                            bulk_pos[b][0] - centroid[0]) for b in ins}
    tiles = sorted(ins, key=lambda b: (depth[b], ang_of[b]))
    labels = [f"T{i}" for i in range(k)]
    label_of = dict(zip(tiles, labels))
    rank_of = {t: i for i, t in enumerate(tiles)}

    cyc = {b: _pentagon_cycle(V, centre_of[b], legs_of[b]) for b in ins}
    leg_order: Dict[int, List[int]] = {}
    for t in tiles:
        seq = cyc[t]
        bondlegs = [l for l in seq if partner(l) is not None]
        if bondlegs:
            leg0 = min(bondlegs,
                       key=lambda l: (rank_of[tile_of_leg[partner(l)]],))
        else:
            leg0 = seq[0]
        i = seq.index(leg0)
        leg_order[t] = seq[i:] + seq[:i]

    qubits = [(label_of[t], q) for t in tiles for q in range(5)]
    q_index = {key: i for i, key in enumerate(qubits)}
    leg_spider_of_q = [leg_order[t][q] for t in tiles for q in range(5)]
    q_of_leg = {l: i for i, l in enumerate(leg_spider_of_q)}

    bonds, bond_family = [], []
    for l, w in bonds_raw:
        qa, qb = q_of_leg[l], q_of_leg[w]
        da, db = depth[tile_of_leg[l]], depth[tile_of_leg[w]]
        if (da, db) > (db, da):
            qa, qb, da, db = qb, qa, db, da
        bonds.append((qa, qb))
        bond_family.append(f"d{da}-d{db}")
    order = np.argsort([b[0] for b in bonds])
    bonds = [bonds[i] for i in order]
    bond_family = [bond_family[i] for i in order]

    fam_of_bond_qubit = {}
    for (qa, qb), f in zip(bonds, bond_family):
        fam_of_bond_qubit[qa] = fam_of_bond_qubit[qb] = f
    bnd_qubits = [i for i in range(5 * k) if i not in fam_of_bond_qubit]
    out_index = {o: i for i, o in enumerate(outs)}
    perm = [out_index[bnd_of_leg[leg_spider_of_q[q]]] for q in bnd_qubits]
    assert sorted(perm) == list(range(len(outs))), "boundary map is not a bijection"

    leg_pos = {q: (float(V.qubit(leg_spider_of_q[q])),
                   float(V.row(leg_spider_of_q[q]))) for q in range(5 * k)}
    return dict(V=V, ins=ins, outs=outs, k=k, n=5 * k,
                central_in=central_in, central_in_index=ins.index(central_in),
                tiles=tiles, labels=labels, label_of=label_of,
                depth={label_of[t]: depth[t] for t in tiles},
                depth_hist=dict(Counter(depth.values())),
                QUBITS=qubits, Q_INDEX=q_index, BONDS=bonds,
                BOND_FAMILY=bond_family, BND_QUBITS=bnd_qubits,
                FAM_OF_BOND_QUBIT=fam_of_bond_qubit, perm=perm,
                leg_pos=leg_pos, leg_spider_of_q=leg_spider_of_q,
                centroid=(float(centroid[0]), float(centroid[1])),
                centre_pos={label_of[t]: (float(bulk_pos[t][0]),
                                          float(bulk_pos[t][1]))
                            for t in tiles},
                bond_families=dict(Counter(bond_family)))


# --------------------------------------------------- 2. the extended checks
def tile_generator_strings() -> List[str]:
    from .foliated_block import signed_generators
    out = []
    for gsx, gsz in signed_generators("anc"):
        out.append("".join(
            "Y" if (abs(x) % 2 and abs(z) % 2) else
            "X" if abs(x) % 2 else "Z" if abs(z) % 2 else "I"
            for x, z in zip(gsx, gsz)))
    return out


def embed_pauli(geo: Dict, lbl: str, s: str) -> np.ndarray:
    n = geo["n"]
    row = np.zeros(2 * n, dtype=np.uint8)
    for leg, p in enumerate(s):
        q = geo["Q_INDEX"][(lbl, leg)]
        if p in "XY":
            row[q] = 1
        if p in "ZY":
            row[n + q] = 1
    return row


def tile_check_matrix(geo: Dict) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    strs = tile_generator_strings()
    rows, names = [], []
    for lbl in geo["labels"]:
        for c, s in enumerate(strs):
            rows.append(embed_pauli(geo, lbl, s))
            names.append((lbl, c))
    return np.array(rows, dtype=np.uint8), names


def bell_rows(geo: Dict) -> Tuple[np.ndarray, List[str]]:
    """XX and ZZ on the two legs of each bond -- the cup, as a check pair."""
    n = geo["n"]
    rows, names = [], []
    for kk, (a, b) in enumerate(geo["BONDS"]):
        r1 = np.zeros(2 * n, dtype=np.uint8)
        r2 = np.zeros(2 * n, dtype=np.uint8)
        r1[a] = r1[b] = 1
        r2[n + a] = r2[n + b] = 1
        rows += [r1, r2]
        names += [f"bell{kk}.XX[{geo['BOND_FAMILY'][kk]}]",
                  f"bell{kk}.ZZ[{geo['BOND_FAMILY'][kk]}]"]
    if not rows:
        return np.zeros((0, 2 * n), dtype=np.uint8), []
    return np.array(rows, dtype=np.uint8), names


def mismatch_matrix(rows: np.ndarray, geo: Dict) -> np.ndarray:
    """Failure of each row to commute with the bond checks: two bits per bond
    (x-mismatch, z-mismatch).  A row is Bell-compatible iff its whole
    mismatch vector vanishes."""
    n, nb = geo["n"], len(geo["BONDS"])
    out = np.zeros((rows.shape[0], 2 * nb), dtype=np.uint8)
    for kk, (a, b) in enumerate(geo["BONDS"]):
        out[:, 2 * kk] = rows[:, a] ^ rows[:, b]
        out[:, 2 * kk + 1] = rows[:, n + a] ^ rows[:, n + b]
    return out


def _fix_y_parity(rows: np.ndarray) -> np.ndarray:
    """Def. S-1 rule 3 needs every generator to have an EVEN Y-count.  Y-count
    parity is not linear, so this is a deterministic repair pass: an odd row is
    replaced by its product with another basis row that makes it even (the
    span is unchanged).  Raises if no such pair exists."""
    out = [r.copy() for r in rows]
    for i in range(len(out)):
        if y_count(out[i]) % 2 == 0:
            continue
        for j in range(len(out)):
            if j == i:
                continue
            cand = out[i] ^ out[j]
            if cand.any() and y_count(cand) % 2 == 0:
                out[i] = cand
                break
        else:
            raise AssertionError(f"cannot make row {i} even-Y within the span")
    return np.array(out, dtype=np.uint8)


def lifted_tile_checks(geo: Dict) -> Dict:
    """The Bell-compatible subgroup of the tile-check group = left kernel of
    the mismatch matrix, realised as products of bare tile generators."""
    S_tile, tile_names = tile_check_matrix(geo)
    Phi = mismatch_matrix(S_tile, geo)
    ker = left_nullspace(Phi) if Phi.shape[1] else \
        [np.eye(S_tile.shape[0], dtype=np.uint8)[i]
         for i in range(S_tile.shape[0])]
    B = np.array([np.bitwise_xor.reduce(S_tile[v.astype(bool)], axis=0)
                  for v in ker], dtype=np.uint8) if ker else \
        np.zeros((0, 2 * geo["n"]), dtype=np.uint8)
    assert not mismatch_matrix(B, geo).any(), "lift is not Bell-compatible"
    B = _fix_y_parity(B)
    assert not mismatch_matrix(B, geo).any(), "y-parity repair broke the lift"
    return dict(S_tile=S_tile, tile_names=tile_names, Phi=Phi,
                phi_rank=int(fold.gf2_rank(Phi)) if Phi.shape[1] else 0,
                B=B, lift_dim=len(ker),
                lift_weights=dict(sorted(Counter(row_weight(r)
                                                 for r in B).items())))


def bond_extended_code(R: int) -> Dict:
    """The radius-R bond-extended code, in the same shape as
    `wt02_pipeline.contracted_code` so the rest of the pipeline is reusable."""
    con = W.contracted_code(R)
    geo = derive_geometry(con["V"])
    lift = lifted_tile_checks(geo)
    bells, bell_names = bell_rows(geo)
    n = geo["n"]

    S_ext = np.vstack([lift["B"], bells]) if len(bells) else lift["B"]
    names = [f"lift{i}" for i in range(len(lift["B"]))] + bell_names
    G = np.array([[sym_pair_rows(a, b) for b in S_ext] for a in S_ext],
                 dtype=np.uint8)
    assert not G.any(), "extended checks do not pairwise commute"
    rank = int(fold.gf2_rank(S_ext))
    k_log = n - rank
    assert k_log == geo["k"], \
        f"[[{n},{k_log}]] but the patch has {geo['k']} bulk legs"
    ys = [y_count(r) for r in S_ext]
    assert all(y % 2 == 0 for y in ys), "odd-Y generator breaks Def. S-1"

    # an independent generating set, for the foliation
    keep, span = [], W.Gf2Span()
    for i, r in enumerate(S_ext):
        if span.add(r.copy()):
            keep.append(i)
    rows = S_ext[keep]
    strings = [row_to_pauli(r) for r in rows]
    sx = rows[:, :n].copy()
    sz = rows[:, n:].copy()
    weights = dict(sorted(Counter(row_weight(r) for r in rows).items()))
    return dict(R=R, V=con["V"], geo=geo, contracted=con, strings=strings,
                n=n, k=k_log, m=len(rows), weights=weights, code_rows=rows,
                sx=sx, sz=sz, S_ext=S_ext, names_ext=names,
                kept=keep, kept_names=[names[i] for i in keep],
                lift=lift, bells=bells, bell_names=bell_names,
                n_bonds=len(geo["BONDS"]),
                n_boundary=len(geo["BND_QUBITS"]))


# ------------------------------------------------------- 3. logical sector
def logical_reps_ext(code: Dict) -> Dict:
    """The contracted code's logical representatives, carried onto the
    extended qubit set (boundary legs keep their support, bond legs get
    identity).  Checked here: they commute with every extended check, they
    are not in the check group, and they pair up symplectically as X_i/Z_i."""
    geo, con = code["geo"], code["contracted"]
    reps_c = W.logical_reps(con["V"], con["strings"])
    n_c, n_e = con["n"], code["n"]
    slot_of_out = {}                       # contracted qubit -> extended qubit
    for q, o in zip(geo["BND_QUBITS"], geo["perm"]):
        slot_of_out[o] = q
    assert len(slot_of_out) == n_c

    out = []
    for rep in reps_c["reps"]:
        x, z = np.asarray(rep[1]), np.asarray(rep[2])
        row = np.zeros(2 * n_e, dtype=np.uint8)
        for j in range(n_c):
            q = slot_of_out[j]
            row[q], row[n_e + q] = x[j], z[j]
        out.append((rep[0], row[:n_e].copy(), row[n_e:].copy()))

    span = W.Gf2Span()
    for r in code["code_rows"]:
        span.add(r.copy())
    for name, x, z in out:
        row = np.concatenate([x, z])
        assert all(sym_pair_rows(row, s) == 0 for s in code["S_ext"]), \
            f"logical {name} does not commute with the extended checks"
        assert not span.contains(row.copy()), f"logical {name} is a stabiliser"
    kk = len(out) // 2
    pair = np.array([[sym_pair_rows(np.concatenate([out[i][1], out[i][2]]),
                                    np.concatenate([out[j][1], out[j][2]]))
                      for j in range(2 * kk)] for i in range(2 * kk)],
                    dtype=np.uint8)
    want = np.zeros_like(pair)          # reps are interleaved X_b, Z_b
    for i in range(kk):
        want[2 * i, 2 * i + 1] = want[2 * i + 1, 2 * i] = 1
    assert np.array_equal(pair, want), \
        f"logical pairing is not standard X/Z:\n{pair}"
    return dict(reps=out, central=reps_c["central"], n=n_e, k=kk,
                central_margin=reps_c.get("central_margin"),
                pairing_ok=True)


# ------------------------------------------------------------ 4. embedding
def leg_positions(code: Dict) -> np.ndarray:
    """(n, 2) true Poincare-disk position of every extended qubit -- these
    fill the disk, which is the whole point of keeping the bonds physical."""
    geo = code["geo"]
    xy = np.array([geo["leg_pos"][q] for q in range(code["n"])], dtype=float)
    return xy - np.array(geo["centroid"], dtype=float)
