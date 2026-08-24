"""Radius-R HaPPY spacetime pipeline: contracted code -> foliation -> fold
-> decoding matrices -> heralded-erasure Monte Carlo.

This is the arbitrary-radius generalisation of the radius-1-only machinery in
`r1_fold_decoding.py` (which hard-codes n=20, k=6, m=14).  Every size is read
off the encoder instead: n = #boundary legs, k = #bulk legs, m = #stabiliser
generators returned by `webs_to_checks.code_stabiliser_basis`.

Pipeline (all steps machine-checked, nothing assumed):

  1. CONTRACTED CODE   `happy_spacetime.build_happy_encoder(R)` -> encoder V;
     `code_stabiliser_basis(V)` -> the [[n, k]] generator strings (the
     trivial-bulk subgroup of V's own stabiliser webs).
  2. FOLIATION         Def. S-1 of arXiv:2607.13784 via
     `foliated_general.build_general_foliated_block(sx, sz, K)`.  Exactness of
     the all-X measurement plan requires every generator to have an EVEN
     Y-count (rule 3 of Def. S-1); this is asserted, not assumed.
  3. FOLD              two copies joined in-to-in and out-to-out by plain
     (Bell-cup) seam wires, `r1_fold_decoding.build_fold_graph`.
  4. CLASSIFY          the closed-web space of the fold splits as
        per-copy detectors  (+)  seam webs  (+)  logical loops
     with the logical loops obtained as the ALIGNED images of the single
     copy's OPEN webs that survive the quotient by detectors + seams.  Names
     come from symplectic pairing against logical representatives extracted
     from the encoder V itself.
  5. DECODE            (H_det, O) over the fixed-half-edge convention.
  6. ERASURE MC        heralded erasure, uniform Pauli on erased wires,
     BP+OSD-0 with the caller's decoder spec.

Sign conventions are GF(2) throughout (webs carry no sign); the simulations
only need unsigned anticommutation.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyzx as zx
from pyzx.utils import VertexType

from . import folded_block as fold
from . import foliated_general as fg
from .r1_fold_decoding import (aligned_web, build_fold_graph, pauli_xz,
                               sym_pair, xz_to_str)

# --------------------------------------------------------------- GF(2) helpers


class Gf2Span:
    """Incrementally-maintained RREF basis over GF(2), rows stored as packed
    python ints (bit j = column j; pivot = LOWEST set bit = same leading-
    column convention as the previous uint8 implementation)."""

    def __init__(self) -> None:
        self.rows: List[int] = []
        self.piv: List[int] = []

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _to_int(v) -> int:
        if isinstance(v, int):
            return v
        v = np.asarray(v, dtype=np.uint8)
        return int.from_bytes(
            np.packbits(v, bitorder="little").tobytes(), "little")

    def reduce(self, v) -> int:
        x = self._to_int(v)
        for r, p in zip(self.rows, self.piv):
            if (x >> p) & 1:
                x ^= r
        return x

    def contains(self, v) -> bool:
        return self.reduce(v) == 0

    def add(self, v) -> bool:
        """Insert v; True iff it enlarged the span."""
        x = self.reduce(v)
        if x == 0:
            return False
        p = (x & -x).bit_length() - 1
        for i in range(len(self.rows)):
            if (self.rows[i] >> p) & 1:
                self.rows[i] ^= x
        self.rows.append(x)
        self.piv.append(p)
        return True


def gf2_solve(rows: Sequence[np.ndarray], target: np.ndarray) -> Optional[np.ndarray]:
    """c with sum_i c_i rows_i = target over GF(2), or None (verified)."""
    rows = [np.asarray(r, dtype=np.uint8) for r in rows]
    if not rows:
        return None if np.any(target) else np.zeros(0, dtype=np.uint8)
    A = np.array(rows, dtype=np.uint8).T.copy()
    b = np.asarray(target, dtype=np.uint8).copy()
    ng = A.shape[1]
    piv: Dict[int, int] = {}
    r = 0
    for c in range(ng):
        col = A[r:, c]
        nz = np.flatnonzero(col)
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
            b[r], b[i] = b[i], b[r]
        hit = np.flatnonzero(A[:, c])
        hit = hit[hit != r]
        if hit.size:
            A[hit] ^= A[r]
            b[hit] ^= b[r]
        piv[c] = r
        r += 1
        if r == A.shape[0]:
            break
    sol = np.zeros(ng, dtype=np.uint8)
    for c, rr in piv.items():
        sol[c] = b[rr]
    chk = np.zeros_like(np.asarray(target, dtype=np.uint8))
    for i, ci in enumerate(sol):
        if ci:
            chk = chk ^ rows[i]
    return sol if not np.any(chk ^ np.asarray(target, dtype=np.uint8)) else None


def rowspace_equal(A: np.ndarray, B: np.ndarray) -> bool:
    A = np.asarray(A, dtype=np.uint8)
    B = np.asarray(B, dtype=np.uint8)
    rA, rB = fold.gf2_rank(A), fold.gf2_rank(B)
    return rA == rB == fold.gf2_rank(np.vstack([A, B]))


# ------------------------------------------------- closed webs, done directly
def closed_web_basis_direct(g) -> List:
    """Basis of the closed-web space of a CLOSED all-Z, phase-0 diagram whose
    edges are SIMPLE or HADAMARD -- solved as one small GF(2) system instead of
    pyzx's red-green firing machinery.

    Parameterisation.  A web assigns Pauli X^a Z^b to each half-edge.  At a
    phase-0 Z spider the web must be in <X^(x)deg, Z_i Z_j>, so the X part is
    all-or-nothing: one FIRE bit f_u per spider, giving X-exponent f_u on every
    half-edge at u; and the Z exponents at u must sum to 0 mod 2.
      * HADAMARD edge {u,v}: the H-conjugation rule forces the Z exponent of
        the half-edge at u to be f_v (and at v to be f_u) -- no new freedom.
      * SIMPLE edge e = {u,v}: both halves must carry the SAME Pauli, which
        forces f_u = f_v and leaves one free Z bit h_e shared by both halves.
    So the closed webs are exactly the GF(2) solutions of
        (a) for every spider u:  sum_{H-edges u~v} f_v + sum_{simple e ni u} h_e = 0
        (b) for every simple edge {u,v}:  f_u + f_v = 0
    in the |V| + |simple| variables (f, h).  For the radius-2 fold that is a
    686 x 686 system, versus the ~13000 x 13000 dense system pyzx's
    Hadamard-expansion route produces (which is why this exists).

    Verified against `folded_block.closed_web_basis` (i.e. against pyzx) at
    R = 0 and R = 1 in the notebook: same dimension AND same span.
    """
    from pyzx.pauliweb import PauliWeb
    from pyzx.utils import EdgeType, VertexType

    assert g.num_inputs() + g.num_outputs() == 0, "diagram is not closed"
    verts = sorted(g.vertices())
    for v in verts:
        assert g.type(v) == VertexType.Z and g.phase(v) == 0, \
            "closed_web_basis_direct assumes phase-0 Z spiders only"
    vidx = {v: i for i, v in enumerate(verts)}
    simple, had = [], []
    for e in g.edges():
        s, t = g.edge_st(e)
        (simple if g.edge_type(e) == EdgeType.SIMPLE else had).append((s, t))
    nv, ns = len(verts), len(simple)

    A = np.zeros((nv + ns, nv + ns), dtype=np.uint8)
    for s, t in had:                                  # (a) H-edge neighbours
        A[vidx[s], vidx[t]] ^= 1
        A[vidx[t], vidx[s]] ^= 1
    for j, (s, t) in enumerate(simple):               # (a) simple-edge Z bits
        A[vidx[s], nv + j] ^= 1
        A[vidx[t], nv + j] ^= 1
        A[nv + j, vidx[s]] ^= 1                       # (b) fire bits agree
        A[nv + j, vidx[t]] ^= 1

    def pauli(a: int, b: int) -> str:
        return ("I", "Z", "X", "Y")[(2 * int(a)) + int(b)]

    for sol in fold.gf2_nullspace(A):
        w = PauliWeb(g)
        for s, t in had:
            fs, ft = sol[vidx[s]], sol[vidx[t]]
            if pauli(fs, ft) != "I":
                w.add_half_edge((s, t), pauli(fs, ft))
            if pauli(ft, fs) != "I":
                w.add_half_edge((t, s), pauli(ft, fs))
        for j, (s, t) in enumerate(simple):
            p = pauli(sol[vidx[s]], sol[nv + j])
            if p != "I":
                w.add_half_edge((s, t), p)
                w.add_half_edge((t, s), p)
        yield w
    return


def closed_web_vecs_direct(g, cols) -> np.ndarray:
    """All closed-web VECTORS of a closed all-Z phase-0 diagram, emitted
    directly from the (f, h) parameterisation -- no PauliWeb dicts, closure
    guaranteed by construction (same system as closed_web_basis_direct)."""
    from pyzx.utils import EdgeType, VertexType

    assert g.num_inputs() + g.num_outputs() == 0
    verts = sorted(g.vertices())
    vidx = {v: i for i, v in enumerate(verts)}
    simple, had = [], []
    for e in g.edges():
        st = g.edge_st(e)
        (simple if g.edge_type(e) == EdgeType.SIMPLE else had).append(st)
    nv, ns = len(verts), len(simple)
    A = np.zeros((nv + ns, nv + ns), dtype=np.uint8)
    for st, tt in had:
        A[vidx[st], vidx[tt]] ^= 1
        A[vidx[tt], vidx[st]] ^= 1
    for j, (st, tt) in enumerate(simple):
        A[vidx[st], nv + j] ^= 1
        A[vidx[tt], nv + j] ^= 1
        A[nv + j, vidx[st]] ^= 1
        A[nv + j, vidx[tt]] ^= 1
    # packed nullspace: the fold system is ~(nv+ns)^2 -- 25k^2 at e45 n=2,
    # far beyond the 686^2 this routine was written for (4th balloon)
    from .gf2compat import _pack, _rref_inplace, _unpack
    Wp, rws, cls = _pack(A)
    del A
    pivs = _rref_inplace(Wp, rws, cls, full_reduce=True)
    Rr = np.array(_unpack(Wp[:len(pivs)], cls), dtype=np.uint8)
    del Wp
    free = sorted(set(range(cls)) - set(pivs))
    if not free:
        return
    S = np.zeros((len(free), cls), dtype=np.uint8)
    for k, fc in enumerate(free):
        S[k, fc] = 1
        S[k, pivs] = Rr[:, fc]
    del Rr
    hb1 = np.array([cols[(st, tt)] for st, tt in had], dtype=np.int64)
    hb2 = np.array([cols[(tt, st)] for st, tt in had], dtype=np.int64)
    his = np.array([vidx[st] for st, tt in had], dtype=np.int64)
    hit = np.array([vidx[tt] for st, tt in had], dtype=np.int64)
    sb1 = np.array([cols[(st, tt)] for st, tt in simple], dtype=np.int64)
    sb2 = np.array([cols[(tt, st)] for st, tt in simple], dtype=np.int64)
    sis = np.array([vidx[st] for st, tt in simple], dtype=np.int64)
    shj = np.arange(nv, nv + ns, dtype=np.int64)
    for k in range(S.shape[0]):
        row = np.zeros(2 * len(cols), dtype=np.uint8)
        sk = S[k]
        row[hb1] = sk[his]; row[hb1 + 1] = sk[hit]
        row[hb2] = sk[hit]; row[hb2 + 1] = sk[his]
        row[sb1] = sk[sis]; row[sb1 + 1] = sk[shj]
        row[sb2] = sk[sis]; row[sb2 + 1] = sk[shj]
        yield row


# ------------------------------------------------------- 1. contracted code
def contracted_code(R: int) -> Dict:
    """The radius-R HaPPY encoder and the [[n, k]] code it encodes."""
    import compat
    compat.install_pyzx_rules_shim()
    from happy_spacetime import build_happy_encoder
    from webs_to_checks import code_stabiliser_basis

    V = build_happy_encoder(R)
    if isinstance(V, tuple):
        V = V[0]
    basis, strings = code_stabiliser_basis(V)
    n, m = len(strings[0]), len(strings)
    k = V.num_inputs()
    assert V.num_outputs() == n
    assert n - m == k, f"[[{n},{n - m}]] but encoder has {k} bulk legs"
    weights: Dict[int, int] = {}
    for s in strings:
        w = sum(1 for ch in s if ch != "I")
        weights[w] = weights.get(w, 0) + 1
    code_rows = np.array([np.concatenate(pauli_xz(s)) for s in strings],
                         dtype=np.uint8)
    assert fold.gf2_rank(code_rows) == m, "generators not independent"
    sx, sz = fg.pauli_strings_to_sxsz(strings)
    sym = (sx @ sz.T + sz @ sx.T) % 2
    assert not sym.any(), "generators do not pairwise commute"
    return dict(R=R, V=V, strings=strings, n=n, k=k, m=m,
                weights=dict(sorted(weights.items())), code_rows=code_rows,
                sx=sx, sz=sz)


def boundary_angles(V) -> List[float]:
    """Polar angle of each boundary leg about the centroid of all of them —
    the encoder's own (Poincare-disk) layout, used for the 3D embeddings."""
    pos = np.array([[float(V.row(next(iter(V.neighbors(b))))),
                     float(V.qubit(next(iter(V.neighbors(b)))))]
                    for b in V.outputs()])
    c = pos.mean(axis=0)
    return [math.atan2(p[1] - c[1], p[0] - c[0]) for p in pos]


# ------------------------------------------------------- 2. foliated block
def foliate(code: Dict, K: int = 2) -> Dict:
    """Def. S-1 foliation of the contracted code; asserts the all-X plan."""
    blk = fg.build_general_foliated_block(code["sx"], code["sz"], K)
    ycounts = [int((code["sx"][c] & code["sz"][c]).sum()) for c in range(code["m"])]
    odd = [c for c, y in enumerate(ycounts) if y % 2]
    all_x = all(b == "X" for b in blk.anc_basis)
    assert (not odd) == all_x
    return dict(block=blk, K=K, ycounts=ycounts, odd_y_generators=odd,
                all_X_plan_exact=all_x)


def single_open_zx(blk) -> Tuple:
    """The single open block as a ZX diagram + its open webs (pyzx).

    `compute_pauli_webs` shares the one expensive firing-assignment nullspace
    between the stabiliser and detecting-region passes; calling the two
    single-purpose wrappers would compute it twice.
    """
    from src.lazy_webs import LazyRegions

    block = blk['block'] if isinstance(blk, dict) else blk
    g1, vmap1 = fg.to_zx(block)
    cols1 = fold.half_edge_cols(g1)
    lz = LazyRegions(g1, fold, cols1)
    stabs1 = lz.stab_webs()
    return g1, vmap1, lz, stabs1


# ------------------------------------------ logical reps from the encoder
def logical_reps(V, stab_strings: Sequence[str]) -> Dict:
    """2k logical representatives of the [[n, k]] code, one X/Z pair per bulk
    input of V, plus the central-bulk-qubit identification.

    Arbitrary-k generalisation of `r1_fold_decoding.logical_reps_from_v1`.
    Every stabilising web of V is an (unsigned) P_bulk (x) P_boundary of the
    encoder's Choi state; solving over GF(2) for combos whose bulk part is
    EXACTLY X_b (resp. Z_b) leaves a boundary Pauli that represents logical
    Xbar_b (resp. Zbar_b) modulo code stabilisers -- which pair trivially, so
    the naming is well defined.
    """
    from pyzx.web.compute import compute_stabilisers

    ins, outs = list(V.inputs()), list(V.outputs())
    k, n = len(ins), len(outs)
    webs = compute_stabilisers(V)

    def leg(w, b):
        nb = next(iter(V.neighbors(b)))
        return w.half_edges().get((nb, b), "I")

    bulk_rows, bnd_rows = [], []
    for w in webs:
        bx, bz = pauli_xz("".join(leg(w, b) for b in ins))
        px, pz = pauli_xz("".join(leg(w, b) for b in outs))
        bulk_rows.append(np.concatenate([bx, bz]))
        bnd_rows.append(np.concatenate([px, pz]))

    reps: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for b in range(k):
        for kind, col in (("X", b), ("Z", k + b)):
            target = np.zeros(2 * k, dtype=np.uint8)
            target[col] = 1
            sol = gf2_solve(bulk_rows, target)
            assert sol is not None, f"no web combo realises {kind}_bulk{b}"
            acc = np.zeros(2 * n, dtype=np.uint8)
            for i, ci in enumerate(sol):
                if ci:
                    acc = acc ^ bnd_rows[i]
            reps.append((f"{kind}bar(bulk{b})", acc[:n].copy(), acc[n:].copy()))

    stab_xz = [pauli_xz(s) for s in stab_strings]
    for name, rx, rz in reps:                       # centralize the code group
        for sx_, sz_ in stab_xz:
            assert sym_pair(rx, rz, sx_, sz_) == 0, f"{name} fails to centralize"
    for i, (ni, xi, zi) in enumerate(reps):         # standard symplectic form
        for j, (nj, xj, zj) in enumerate(reps):
            want = 1 if (i // 2 == j // 2 and i != j) else 0
            assert sym_pair(xi, zi, xj, zj) == want, f"pairing {ni},{nj}"

    bulk_pos = np.array([[float(V.row(next(iter(V.neighbors(b))))),
                          float(V.qubit(next(iter(V.neighbors(b)))))]
                         for b in ins])
    out_pos = np.array([[float(V.row(next(iter(V.neighbors(b))))),
                         float(V.qubit(next(iter(V.neighbors(b)))))]
                        for b in outs])
    dists = np.linalg.norm(bulk_pos - out_pos.mean(axis=0), axis=1)
    i_c = int(np.argmin(dists))
    order = np.sort(dists)
    if len(order) == 1:
        central_margin = float("inf")               # the only bulk qubit
    else:
        central_margin = float(order[1] / order[0]) if order[0] > 0 else float("inf")
    return dict(reps=reps, central=i_c, central_margin=central_margin,
                central_distances=[round(float(d), 4) for d in dists],
                n_webs=len(webs), k=k, n=n)


def class_of(x, z, reps) -> np.ndarray:
    return np.array([sym_pair(x, z, rx, rz) for _, rx, rz in reps], dtype=np.uint8)


def loop_target(kind: str, b: int, k: int) -> np.ndarray:
    """A loop measuring Xbar_b anticommutes with Zbar_b alone (and v.v.)."""
    t = np.zeros(2 * k, dtype=np.uint8)
    t[2 * b + 1 if kind == "X" else 2 * b] = 1
    return t


# ---------------------------------------------------------- 3+4. fold + classify
def fold_and_classify(blk, g1, vmap1, regions1, stabs1, reps_in, central: int,
                      code_rows_in: np.ndarray, weight_passes: int = 2,
                      verbose: bool = False) -> Dict:
    """Fold two copies, decompose the closed-web space, name the loops.

    Returns everything the decoding step needs.  Arbitrary-radius version of
    `r1_fold_decoding.classify_fold`: k, n, m are read off the inputs.
    """
    t0 = time.perf_counter()
    K = blk.rounds
    n, m = blk.n, blk.m
    k = len(reps_in) // 2
    nodes, edges = blk.nodes, blk.edges
    in_legs, out_legs = blk.in_legs, blk.out_legs
    detectors = (None if blk.detectors is None else
                 [(f"D(c={d['c']},k={d['k']})", d["nodes"]) for d in blk.detectors])

    def log(msg):
        if verbose:
            print(f"      [{time.perf_counter() - t0:6.1f}s] {msg}", flush=True)

    gF, vmapF, seams = build_fold_graph(nodes, edges, in_legs, out_legs)
    n_wires_expected = 2 * len(edges) + len(in_legs) + len(out_legs)
    wires = fold.sorted_wires(gF)
    assert len(wires) == gF.num_edges() == n_wires_expected
    assert len(set(wires)) == len(wires), "duplicate wire"
    log(f"fold graph: {gF.num_vertices()} spiders, {gF.num_edges()} wires")

    colsF = fold.half_edge_cols(gF)
    cols1 = fold.half_edge_cols(g1)
    if hasattr(regions1, "vecs"):
        stab_vecs = np.array([fold.web_vec(w, cols1) for w in stabs1],
                             dtype=np.uint8)
        M1 = (np.vstack([stab_vecs, regions1.vecs]) if len(regions1)
              else stab_vecs)
        open_webs = _LazyOpenWebs(stabs1, regions1)
    else:
        open_webs = list(stabs1) + list(regions1)
        M1 = np.array([fold.web_vec(w, cols1) for w in open_webs],
                      dtype=np.uint8)
    in_b, out_b = list(g1.inputs()), list(g1.outputs())

    span_all = Gf2Span()
    MallF = []                                     # packed rows (x8 smaller)
    _mw = None
    for v in closed_web_vecs_direct(gF, colsF):    # generator: one row alive
        _mw = v.shape[0]
        MallF.append(np.packbits(v))
        span_all.add(v)
    d_all = len(span_all)
    log(f"closed-web space of the fold: dim {d_all} (from {len(MallF)} webs)")

    # --- per-copy detectors ------------------------------------------------
    det_webs, det_names = [], []
    span_det = Gf2Span()
    if detectors is None:
        # scheduled blocks: per-copy detectors = the open block's detecting
        # regions (pyzx firing nullspace — valid webs by construction),
        # embedded verbatim on each copy of the fold graph
        inv1 = {v: nd for nd, v in vmap1.items()}
        Rvecs = (regions1.vecs if hasattr(regions1, "vecs")
                 else np.array([fold.web_vec(w, cols1) for w in regions1],
                               dtype=np.uint8))
        for ci in (0, 1):
            # column permutation cols1 -> colsF for this copy
            perm = np.full(2 * len(cols1), -1, dtype=np.int64)
            for e1, b1 in cols1.items():
                u, vtx = e1
                if u not in inv1 or vtx not in inv1:
                    continue          # boundary half-edge: regions zero there
                eF = (vmapF[(ci, inv1[u])], vmapF[(ci, inv1[vtx])])
                bF = colsF.get(eF)
                if bF is None:
                    continue
                perm[b1] = bF
                perm[b1 + 1] = bF + 1
            ncolF = 2 * len(colsF)
            ok = perm >= 0
            for widx in range(len(Rvecs)):
                assert not Rvecs[widx][~ok].any(), "region touches boundary"
                v = np.zeros(ncolF, dtype=np.uint8)
                v[perm[ok]] = Rvecs[widx][ok]
                assert span_all.contains(v), f"copy{ci}:R{widx} outside space"
                assert span_det.add(v), f"copy{ci}:R{widx} dependent"
                row2 = np.empty(ncolF // 2, dtype=np.uint8)
                row2[0::2] = v[1::4]          # X-flip vs Z-label (validated)
                row2[1::2] = v[0::4]
                det_webs.append(("vecp", np.packbits(row2), row2.shape[0]))
                det_names.append(f"copy{ci}:R{widx}")
                del v
        r_det = len(span_det)
        assert r_det == 2 * len(regions1)
        log(f"per-copy detectors (embedded regions): rank {r_det}")
    else:
        for ci in (0, 1):
            for name, det_nodes in detectors:
                w = fold.fire_web(gF, [vmapF[(ci, nd)] for nd in det_nodes])
                assert fold.web_is_closed(gF, w), f"copy{ci}:{name} not closed"
                v = fold.web_vec(w, colsF)
                assert span_all.contains(v), f"copy{ci}:{name} outside closed-web space"
                assert span_det.add(v), f"copy{ci}:{name} dependent"
                det_webs.append(w)
                det_names.append(f"copy{ci}:{name}")
        r_det = len(span_det)
        assert r_det == 2 * len(detectors)   # scheduled: Σ_c(E_c-1) per copy
        log(f"per-copy detectors: rank {r_det} = 2 * len(detectors)")

    # --- seam webs: one-boundary open combos, mirror-completed -------------
    Mreg1 = (regions1.vecs if hasattr(regions1, "vecs")
             else (np.array([fold.web_vec(w, cols1) for w in regions1],
                            dtype=np.uint8)
                   if len(regions1)
                   else np.zeros((0, M1.shape[1]), dtype=np.uint8)))
    seam_webs, seam_sides = [], []
    per_seam = {"in": 0, "out": 0}
    for sname, avoid_bnd in (("in", out_b), ("out", in_b)):
        avoid = fold.boundary_wire_cols(g1, cols1, avoid_bnd)
        base = Gf2Span()
        for r in Mreg1:
            base.add(r)
        raw = []
        for lam in fold.gf2_left_nullspace(M1[:, avoid]):
            w1 = fold.combine_webs(g1, open_webs, lam)
            vec = fold.web_vec(w1, cols1)
            if not base.add(vec):
                continue
            raw.append(aligned_web(gF, vmapF, g1, vmap1, w1))
        if weight_passes:
            raw = fold.greedy_weight_reduce(raw, det_webs, weight_passes)
        for w in raw:
            assert fold.web_is_closed(gF, w), f"{sname}-seam web not closed"
        seam_webs += raw
        seam_sides += [sname] * len(raw)
        per_seam[sname] = len(raw)
        log(f"{sname}-seam webs: {len(raw)}")

    # seam-restricted Paulis vs the truncated round-0 / final checks
    seam_wire_of: Dict[tuple, tuple] = {}
    pos = {"in": 0, "out": 0}
    for u, v, side in seams:
        seam_wire_of[(side, pos[side])] = (min(u, v), max(u, v))
        pos[side] += 1
    seam_pauli_rows: Dict[str, list] = {"in": [], "out": []}
    seam_pauli_strs = []
    for w, side in zip(seam_webs, seam_sides):
        es = w.half_edges()
        s = "".join(es.get(seam_wire_of[(side, j)], "I") for j in range(n))
        seam_pauli_strs.append(f"seam-{side}:{s}")
        x, z = pauli_xz(s)
        seam_pauli_rows[side].append(np.concatenate([x, z]))
    code_rows_out = np.concatenate([code_rows_in[:, n:], code_rows_in[:, :n]],
                                   axis=1)              # H-conjugate: x <-> z
    seam_group_matches = {
        side: bool(len(seam_pauli_rows[side]) and rowspace_equal(
            np.array(seam_pauli_rows[side], dtype=np.uint8),
            code_rows_in if side == "in" else code_rows_out))
        for side in ("in", "out")}

    span_det_all = Gf2Span()
    for r in span_det.rows:
        span_det_all.add(r)
    for w in seam_webs:
        assert span_det_all.add(fold.web_vec(w, colsF)), "seam web not independent"
    r_det_all = len(span_det_all)
    assert r_det_all == r_det + len(seam_webs)

    # --- logical loops: quotient of the aligned open webs ------------------
    reps_open = []
    span_q = Gf2Span()
    for r in span_det_all.rows:
        span_q.add(r)
    for w1 in open_webs:
        fw = aligned_web(gF, vmapF, g1, vmap1, w1)
        if span_q.add(fold.web_vec(fw, colsF)):
            reps_open.append(w1)
    assert len(reps_open) == 2 * k, f"logical quotient dim {len(reps_open)} != {2 * k}"
    log(f"logical quotient: dim {len(reps_open)} = 2k")

    cls_rows = [class_of(*pauli_xz(fold.boundary_pauli_string(g1, w1, in_b)),
                         reps_in) for w1 in reps_open]
    assert fold.gf2_rank(np.array(cls_rows, dtype=np.uint8)) == 2 * k, \
        "quotient classes not independent -- naming would be ill-defined"

    log_webs, log_names, log_docs = [], [], []
    for b in range(k):
        for kind in ("X", "Z"):
            target = loop_target(kind, b, k)
            sol = gf2_solve(cls_rows, target)
            assert sol is not None
            w1 = fold.combine_webs(g1, reps_open, sol)
            x, z = pauli_xz(fold.boundary_pauli_string(g1, w1, in_b))
            assert (class_of(x, z, reps_in) == target).all()
            fw = aligned_web(gF, vmapF, g1, vmap1, w1)
            if weight_passes:
                fw = fold.greedy_weight_reduce([fw], det_webs + seam_webs,
                                               weight_passes)[0]
            assert fold.web_is_closed(gF, fw)
            assert not span_det_all.contains(fold.web_vec(fw, colsF)), \
                "a 'logical' loop is really a detector"
            log_webs.append(fw)
            log_names.append(f"loop-{kind}bar(bulk{b})")
            log_docs.append(dict(name=f"loop-{kind}bar(bulk{b})",
                                 in_pauli=xz_to_str(x, z),
                                 out_pauli=fold.boundary_pauli_string(g1, w1, out_b),
                                 n_half_edges=len(fw.half_edges())))
    log(f"named {len(log_webs)} logical loops")

    span_full = Gf2Span()
    for r in span_det_all.rows:
        span_full.add(r)
    for w in log_webs:
        assert span_full.add(fold.web_vec(w, colsF)), "loop dependent on det+seam"
    decomposition = dict(per_copy_detectors=int(r_det),
                         seam_in=per_seam["in"], seam_out=per_seam["out"],
                         logical=2 * k,
                         total=int(r_det + len(seam_webs) + 2 * k),
                         d_all=int(d_all))
    assert len(span_full) == decomposition["total"] == d_all, decomposition
    for pv in MallF:
        v = np.unpackbits(pv)[:_mw]
        assert span_full.contains(v), "closed-web basis not fully classified"

    return dict(g=gF, vmapF=vmapF, seams=seams, wires=wires,
                det_webs=det_webs, seam_webs=seam_webs, log_webs=log_webs,
                det_names=det_names, seam_names=seam_pauli_strs,
                log_names=log_names, log_docs=log_docs,
                decomposition=decomposition, per_seam=per_seam,
                seam_group_matches=seam_group_matches, central=central,
                weight_passes=weight_passes,
                wall_s=round(time.perf_counter() - t0, 1))


# ------------------------------------------------------- 5. decoding matrices
class _LazyOpenWebs:
    """stabs (small list) + LazyRegions, indexable as one sequence."""
    def __init__(self, stabs, lz):
        self.stabs, self.lz = list(stabs), lz
    def __len__(self):
        return len(self.stabs) + len(self.lz)
    def __getitem__(self, i):
        return (self.stabs[i] if i < len(self.stabs)
                else self.lz[i - len(self.stabs)])
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def web_rows(webs, wires) -> np.ndarray:
    """Fixed-half-edge fault/label anticommutation matrix.

    Columns: 2 per wire in `wires` order -- col 2i = (wire i, X flip),
    col 2i+1 = (wire i, Z flip).  A fault component on wire (u, v) is tested
    against the web label on the half-edge (u, v) with u < v; a Hadamard edge
    conjugates fault and label EQUALLY when the fault is slid across, so the
    anticommutation value does not depend on which half is used -- fixing the
    lower-id half is therefore a convention, not a choice of physics.
    """
    idx = {w: i for i, w in enumerate(wires)}
    H = np.zeros((len(webs), 2 * len(wires)), dtype=np.uint8)
    for r, w in enumerate(webs):
        if isinstance(w, tuple) and w[0] == "vecp":
            H[r] = np.unpackbits(w[1])[:w[2]]
            continue
        for (u, v), p in w.half_edges().items():
            if u >= v:
                continue
            i = idx[(u, v)]
            if p in ("Z", "Y"):
                H[r, 2 * i] = 1
            if p in ("X", "Y"):
                H[r, 2 * i + 1] = 1
    return H


def wire_categories(res, blk) -> List[str]:
    """One category label per wire: seam-in / seam-out / worldline / anc-data
    / anc-anc, tagged with the copy index for the intra-copy wires."""
    rev = {v: kk for kk, v in res["vmapF"].items()}
    seam_of = {(min(u, v), max(u, v)): side for u, v, side in res["seams"]}
    cats = []
    for u, v in res["wires"]:
        side = seam_of.get((u, v))
        if side is not None:
            cats.append(f"seam-{side}")
            continue
        cu, ku = rev[u]
        cv, kv = rev[v]
        assert cu == cv
        kinds = {ku[0], kv[0]}
        if kinds == {"data"}:
            base = "worldline"
        elif kinds == {"ancilla"}:
            base = "anc-anc"
        else:
            base = "anc-data"
        cats.append(f"{base}(copy{cu})")
    return cats


def decoding(res, blk) -> Dict:
    """(wires, H_det, O_all, O_central) + the silent-component audit."""
    wires = res["wires"]
    H_det = web_rows(res["det_webs"] + res["seam_webs"], wires)
    O_all = web_rows(res["log_webs"], wires)
    c = res["central"]
    idx_c = [res["log_names"].index(f"loop-Xbar(bulk{c})"),
             res["log_names"].index(f"loop-Zbar(bulk{c})")]
    O_central = O_all[idx_c].copy()

    covH = H_det.any(axis=0)
    cats = wire_categories(res, blk)
    silent: Dict[str, int] = {}
    for j in np.flatnonzero((~covH) & O_all.any(axis=0)):
        silent[cats[int(j) // 2]] = silent.get(cats[int(j) // 2], 0) + 1
    return dict(wires=wires, H_det=H_det, O_all=O_all, O_central=O_central,
                central_rows=idx_c, categories=cats,
                n_silent_all=int(sum(silent.values())), silent_by_family=silent,
                n_silent_central=int(((~covH) & O_central.any(axis=0)).sum()),
                H_rank=int(fold.gf2_rank(H_det)),
                H_nnz=int(H_det.sum()), O_nnz=int(O_all.sum()))


# ------------------------------------------------------------ 6. erasure MC
PRIOR_ERASED = 0.5
PRIOR_INTACT = 1e-9

DECODER_SPEC = dict(name="ldpc.BpOsdDecoder", bp_method="product_sum",
                    max_iter=30, osd_method="OSD_0", osd_order=0,
                    schedule="parallel",
                    priors="per-shot: 0.5 on erased components, 1e-9 else")


def make_decoder(H: np.ndarray, channel: Optional[np.ndarray] = None):
    from ldpc import BpOsdDecoder

    if channel is None:
        channel = np.full(H.shape[1], PRIOR_INTACT)
    return BpOsdDecoder(np.ascontiguousarray(H, dtype=np.uint8),
                        error_channel=list(channel),
                        max_iter=DECODER_SPEC["max_iter"],
                        bp_method=DECODER_SPEC["bp_method"],
                        schedule=DECODER_SPEC["schedule"],
                        osd_method=DECODER_SPEC["osd_method"],
                        osd_order=DECODER_SPEC["osd_order"])


def sample_shot(rng: np.random.Generator, n_wires: int, p_e: float):
    """Heralded erasure: iid wire erasure, uniform Pauli on erased wires."""
    erased_idx = np.flatnonzero(rng.random(n_wires) < p_e)
    e = np.zeros(2 * n_wires, dtype=np.uint8)
    if erased_idx.size:
        bits = rng.integers(0, 2, size=(erased_idx.size, 2), dtype=np.uint8)
        e[2 * erased_idx] = bits[:, 0]
        e[2 * erased_idx + 1] = bits[:, 1]
    return erased_idx, e


def run_point(H: np.ndarray, observables: Dict[str, np.ndarray], p_e: float,
              shots: int, seed: int, check_syndrome: bool = True,
              time_budget_s: Optional[float] = None) -> Dict:
    """Monte-Carlo one (structure, p_e) point; deterministic in `seed`.

    Every observable grouping in `observables` is scored on the SAME shots
    (identical erasure patterns and corrections), so the groupings are
    directly comparable.  If `time_budget_s` is set the run stops early on the
    first shot past the budget and reports how many shots were actually used.
    """
    H = np.ascontiguousarray(H, dtype=np.uint8)
    obs = {k: np.ascontiguousarray(v, dtype=np.uint8) for k, v in observables.items()}
    n_wires = H.shape[1] // 2
    rng = np.random.default_rng(seed)
    dec = make_decoder(H)
    fails = {k: 0 for k in obs}
    ch = np.full(2 * n_wires, PRIOR_INTACT)
    t0 = time.perf_counter()
    done = 0
    for _ in range(shots):
        erased_idx, e = sample_shot(rng, n_wires, p_e)
        s = (H @ e) % 2
        ch[:] = PRIOR_INTACT
        ch[2 * erased_idx] = PRIOR_ERASED
        ch[2 * erased_idx + 1] = PRIOR_ERASED
        dec.update_channel_probs(ch)
        e_hat = dec.decode(s).astype(np.uint8)
        if check_syndrome:
            assert (((H @ e_hat) % 2) == s).all(), "inconsistent correction"
        r = e_hat ^ e
        for kk, O in obs.items():
            if (((O @ r) % 2) != 0).any():
                fails[kk] += 1
        done += 1
        if time_budget_s is not None and time.perf_counter() - t0 > time_budget_s:
            break
    wall = time.perf_counter() - t0
    return dict(p_e=float(p_e), shots=int(done), requested_shots=int(shots),
                fails={k: int(v) for k, v in fails.items()},
                ler={k: v / done for k, v in fails.items()},
                seed=int(seed), wall_s=round(wall, 3))


def wilson(fails: int, shots: int, z: float = 1.959963984540054):
    """Wilson 95% score interval for a binomial proportion."""
    if shots == 0:
        return 0.0, 1.0
    p = fails / shots
    z2 = z * z
    denom = 1 + z2 / shots
    centre = (p + z2 / (2 * shots)) / denom
    half = z * math.sqrt(p * (1 - p) / shots + z2 / (4 * shots * shots)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------- embedding
def boundary_positions(V) -> np.ndarray:
    """The encoder's OWN Poincare-disk coordinates for each boundary leg, in
    `V.outputs()` order (the qubit order the stabiliser strings use).

    `gen_zx_HaPPY` lays the ZX encoder out at the hypertiling {5,4} centre
    coordinates, so these are the true hyperbolic positions -- not angles.
    `boundary_angles` throws the radial part away and puts every leg on one
    circle; use this instead whenever the picture is meant to be the tiling."""
    pos = np.array([[float(V.row(nb)), float(V.qubit(nb))]
                    for b in V.outputs()
                    for nb in (next(iter(V.neighbors(b))),)], dtype=float)
    return pos - pos.mean(axis=0)


def bulk_positions(V) -> np.ndarray:
    """Same, for the bulk (logical) legs, in `V.inputs()` order."""
    out = np.array([[float(V.row(nb)), float(V.qubit(nb))]
                    for b in V.inputs()
                    for nb in (next(iter(V.neighbors(b))),)], dtype=float)
    bnd = np.array([[float(V.row(nb)), float(V.qubit(nb))]
                    for b in V.outputs()
                    for nb in (next(iter(V.neighbors(b))),)], dtype=float)
    return out - bnd.mean(axis=0)


def check_positions(blk, data_xy: np.ndarray) -> Dict[int, np.ndarray]:
    """Each check ancilla at the centroid of the ACTUAL positions of its
    support.  A tile-local check therefore sits on its tile; a deep,
    high-weight check whose support wraps the whole boundary sits near the
    centre.  Depth is measured, not inferred from a circular mean."""
    xy = np.asarray(data_xy, dtype=float)
    out: Dict[int, np.ndarray] = {}
    for c in range(blk.m):
        supp = [q for q in range(blk.n) if blk.sx[c, q] or blk.sz[c, q]]
        out[c] = xy[supp].mean(axis=0) if supp else np.zeros(2)
    return out


def embed_fold_hyperbolic(gF, vmapF, blk, data_xy: np.ndarray,
                          r_data: float = 2.6, t_scale: float = 1.5,
                          dx: float = 0.0, spread: float = 0.0) -> Dict:
    """Embed the fold on the encoder's true {5,4} geometry.

    Data worldlines stand at their own Poincare-disk coordinates, check
    ancillas at the centroid of their support (`check_positions`), time along
    the row axis, the two folded copies offset along the qubit axis.  The only
    liberty taken is a single uniform rescale so the disk has outer radius
    `r_data` at every R -- a similarity, so the tiling is undistorted.

    Coincident checks (distinct checks may share a support centroid) are
    separated deterministically on a small ring, by sorted index; the final
    positions are asserted distinct."""
    xy = np.asarray(data_xy, dtype=float)
    xy = xy - xy.mean(axis=0)
    scale = r_data / max(float(np.hypot(*xy.T).max()), 1e-12)
    xy = xy * scale
    anc = {c: p * scale for c, p in check_positions(blk, data_xy).items()}
    if spread <= 0:
        spread = 0.075 * r_data
    groups: Dict[Tuple[float, float], List[int]] = {}
    for c, p in anc.items():
        groups.setdefault((round(float(p[0]), 6), round(float(p[1]), 6)),
                          []).append(c)
    for (px, py), members in groups.items():
        if len(members) == 1:
            continue
        for j, c in enumerate(sorted(members)):
            th = 2 * math.pi * j / len(members)
            anc[c] = np.array([px + spread * math.cos(th),
                               py + spread * math.sin(th)])
    if dx == 0.0:
        dx = 2.6 * r_data
    seen: Dict[Tuple[float, float, float], int] = {}
    for key, v in vmapF.items():
        ci, (kind, idx, t) = key
        p = xy[idx] if kind == "data" else anc[idx]
        y, z, row = float(p[0]) + ci * dx, float(p[1]), t * t_scale
        gF.set_row(v, row)
        gF.set_qubit(v, y)
        gF.set_vdata(v, "z", z)
        pos = (round(y, 4), round(z, 4), round(row, 4))
        assert pos not in seen, f"spiders {seen[pos]} and {v} coincide at {pos}"
        seen[pos] = v
    depth = {c: float(np.hypot(*anc[c])) / r_data for c in anc}
    return dict(scale=scale, dx=dx, anc=anc, data_xy=xy,
                n_collision_groups=sum(1 for g in groups.values() if len(g) > 1),
                depth=depth)


def embed_fold(gF, vmapF, blk, data_angles: Sequence[float],
               r_data: float = 2.6, r_min: float = 0.35, margin: float = 0.7,
               t_scale: float = 1.5, dx: float = 0.0) -> None:
    """Two holographic prisms, one per copy, offset by `dx` (default: enough
    to clear both prisms).  Data worldlines sit on the boundary circle at the
    encoder's own leg angles; ancillas sit at the circular mean of their
    support with a radius set by the mean resultant length, so deep-bulk
    checks sink to the centre (`foliated_general.holographic_anc_polar`).
    Checks whose supports share a circular mean *and* a mean resultant length
    land on the same polar slot; they are spread deterministically (by sorted
    ancilla index -- no random jitter) so that no two spiders are drawn on top
    of each other, and the result is asserted."""
    anc = dict(fg.holographic_anc_polar(blk, list(data_angles), r_data=r_data,
                                        r_min=r_min, margin=margin))
    slots: Dict[Tuple[float, float], List[int]] = {}
    for i, (ang, r) in anc.items():
        slots.setdefault((round(ang, 6), round(r, 6)), []).append(i)
    for (ang, r), members in slots.items():
        if len(members) == 1:
            continue
        for j, i in enumerate(sorted(members)):        # deterministic spread
            off = j - (len(members) - 1) / 2
            anc[i] = (ang + 0.20 * off, max(r_min * 0.6, r + 0.16 * off))
    if dx == 0.0:
        dx = 2.6 * r_data
    seen: Dict[Tuple[float, float, float], int] = {}
    for key, v in vmapF.items():
        ci, (kind, idx, t) = key
        ang, r = (data_angles[idx], r_data) if kind == "data" else anc[idx]
        x, z, row = r * math.cos(ang) + ci * dx, r * math.sin(ang), t * t_scale
        gF.set_row(v, row)
        gF.set_qubit(v, x)
        gF.set_vdata(v, "z", z)
        pos = (round(x, 4), round(z, 4), round(row, 4))
        assert pos not in seen, f"spiders {seen[pos]} and {v} coincide at {pos}"
        seen[pos] = v


# ------------------------------------------------- 7. id_simp (standing rule)
def identity_chains(gF, wires) -> Tuple[List[List[int]], List[int]]:
    """Group wires into maximal chains joined by degree-2 phase-0 spiders.

    A degree-2, phase-0 spider is an identity: it carries no physical content
    but contributes an extra wire = an extra fault location in the edge-error
    model.  `zx.simplify.id_simp` deletes it and fuses its two wires; here we
    do the matching bookkeeping on the fault-location list, so the decoding
    matrices describe the *simplified* diagram.  Returns (chains, ids) where
    `chains` is a list of wire-index lists and `ids` the removed spiders.
    """
    idx = {w: i for i, w in enumerate(wires)}
    ids = [v for v in gF.vertices()
           if gF.type(v) in (VertexType.Z, VertexType.X)
           and gF.phase(v) == 0 and gF.vertex_degree(v) == 2
           and len(set(gF.neighbors(v))) == 2]
    parent = list(range(len(wires)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for v in ids:                       # fuse the two wires at each identity
        (a, b) = [idx[(min(v, n), max(v, n))] for n in gF.neighbors(v)]
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: Dict[int, List[int]] = {}
    for i in range(len(wires)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), ids


def id_simp_decoding(res, dec, verbose: bool = True) -> Dict:
    """Contract identity chains in the decoding data (standing rule: id_simp).

    Every Pauli web is constant along an identity chain up to the X<->Z swap
    induced by Hadamard edges, so a chain's wires carry *identical* fault
    columns after that swap: they are one fault location, not several.  The
    equality is asserted, not assumed -- it is a machine check of
    identity-insertion invariance of the webs on this diagram.
    """
    gF, wires = res["g"], dec["wires"]
    chains, ids = identity_chains(gF, wires)
    H, Oa, Oc = dec["H_det"], dec["O_all"], dec["O_central"]
    cats = dec["categories"]
    keep, newcats, swaps = [], [], 0
    for ch in chains:
        rep = min(ch)                            # deterministic: no random ids
        cx, cz = H[:, 2 * rep], H[:, 2 * rep + 1]
        for w in ch:                             # verify the web invariance
            wx, wz = H[:, 2 * w], H[:, 2 * w + 1]
            same = np.array_equal(wx, cx) and np.array_equal(wz, cz)
            swap = np.array_equal(wx, cz) and np.array_equal(wz, cx)
            assert same or swap, f"web not constant along identity chain {ch}"
            swaps += int(swap and not same)
        seam = [cats[w] for w in ch if cats[w].startswith("seam")]
        newcats.append(seam[0] if seam else cats[rep])
        keep.append(rep)
    cols = np.array([[2 * r, 2 * r + 1] for r in keep]).ravel()
    out = dict(dec)
    out.update(wires=[wires[r] for r in keep], categories=newcats,
               H_det=H[:, cols].copy(), O_all=Oa[:, cols].copy(),
               O_central=Oc[:, cols].copy(),
               n_identities=len(ids), n_wires_before=len(wires),
               n_wires_after=len(keep), n_hadamard_swaps=swaps)
    covH = out["H_det"].any(axis=0)
    silent: Dict[str, int] = {}
    for j in np.flatnonzero((~covH) & out["O_all"].any(axis=0)):
        silent[newcats[int(j) // 2]] = silent.get(newcats[int(j) // 2], 0) + 1
    out.update(n_silent_all=int(sum(silent.values())), silent_by_family=silent,
               n_silent_central=int(((~covH)
                                     & out["O_central"].any(axis=0)).sum()),
               H_rank=int(fold.gf2_rank(out["H_det"])))
    if verbose:
        print(f"  id_simp: {len(ids)} identity spiders -> wires "
              f"{len(wires)} -> {len(keep)}  ({100*(1-len(keep)/len(wires)):.1f}% "
              f"fewer fault locations); H rank {dec['H_rank']} -> {out['H_rank']}")
    return out


def id_simp_check(res) -> Dict:
    """Cross-check the bookkeeping against pyzx's own id_simp on a clone."""
    g2 = res["g"].clone()
    n0, e0 = g2.num_vertices(), g2.num_edges()
    zx.simplify.id_simp(g2)
    return dict(before=(n0, e0), after=(g2.num_vertices(), g2.num_edges()))
