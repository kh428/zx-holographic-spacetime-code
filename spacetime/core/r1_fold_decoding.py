"""Decoding matrices for the radius-1 blockwise-vs-global erasure comparison.

nb25 proved the blockwise radius-1 HaPPY spacetime (six foliated [[5,1,3]]
tiles, cup joins on the 5 center-ring bonds) and nb24's global foliation
(Def. S-1 of arXiv:2607.13784 on the extracted 14-generator [[20,6]] group)
realize the SAME static code.  As spacetime DIAGRAMS they have different wire
sets — the blockwise picture exposes bond-loop wires as erasure locations,
the global picture has weight-12 delocalized check fans — so their
pure-erasure performance can differ.  This module builds and machine-checks
the (H_det, O) decoding matrices of the three structures the comparison
needs, all at K = 2 rounds, both folds folded on BOTH time sides so
time-boundary artifacts do not pollute the comparison:

  GLOBAL-FOLD      two copies of the general foliated [[20,6]] block
                   (`foliated_general` on `code_stabiliser_basis(V1)`),
                   joined in-in and out-out by plain/Bell seam wires;
  BLOCKWISE-FOLD   two copies of the nb25 6-tile cup patch, joined in-in
                   and out-out on the 20 boundary worldlines (plain seams);
  BLOCKWISE-OPEN   the nb25 patch as-is (time boundaries open) — the
                   boundary-artifact control.

Error model (user-pinned, = nb22/nb23): fault locations are ALL physical
wires of each diagram — including the interior closed bond-loop wires of the
blockwise picture and the seam wires of the folds; each wire carries
independent X- and Z-flip components (heralded uniform-Pauli erasure);
columns 2i / 2i+1 = (wire i, X flip) / (wire i, Z flip); a fault component
is tested against the web label on the FIXED half-edge at the lower vertex
id (side-invariant, see folded_block.decoding_matrices).

Observables: TWO groupings are exported per structure —
  O_all      the 12 logical loops (X/Z of all 6 bulk qubits);
  O_central  the CENTRAL bulk qubit's X/Z pair only.
The central bulk qubit is identified from the encoder V1 itself: the bulk
input whose spider sits nearest the centroid of the 20 boundary spiders
(the Poincare-disk center; unambiguous, see manifest distances), and the
loops are NAMED by symplectic pairing of their in-boundary Pauli against
logical representatives extracted from V1's own stabiliser webs (the Choi /
flow machinery: solve for web combos whose bulk part is exactly X_b or Z_b;
the boundary part is then a representative of that bulk qubit's logical, up
to [[20,6]] stabilizers — which pair trivially, so the naming is
well-defined).  Signs are NOT tracked (GF(2) throughout); the sims only
need unsigned anticommutation.

Everything structural is asserted, not assumed.  Honest negatives are
reported in the manifest rather than patched over.
"""

from __future__ import annotations

import json
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import folded_block as fold
from .compose import composite_to_zx
from .happy_slab import _gf2_solve, build_slab_patch

PRIOR_NOTE = ("heralded uniform-Pauli erasure on ALL wires; priors 0.5 "
              "erased / 1e-9 else; BP+OSD-0 minimum_sum max_iter=10")


# ------------------------------------------------------------------ GF(2)
def pauli_xz(s: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([1 if c in "XY" else 0 for c in s], dtype=np.uint8)
    z = np.array([1 if c in "ZY" else 0 for c in s], dtype=np.uint8)
    return x, z


def xz_to_str(x: np.ndarray, z: np.ndarray) -> str:
    return "".join("Y" if xi and zi else "X" if xi else "Z" if zi else "I"
                   for xi, zi in zip(x, z))


def sym_pair(x1, z1, x2, z2) -> int:
    """Symplectic pairing: 0 = commute, 1 = anticommute."""
    return int((np.sum(x1 & z2) + np.sum(z1 & x2)) % 2)


def rowspace_equal(A: np.ndarray, B: np.ndarray) -> bool:
    from webs_to_checks import gf2_rank
    rA, rB = gf2_rank(A), gf2_rank(B)
    return rA == rB == gf2_rank(np.vstack([A, B]))


# ------------------------------------------- logical reps from the encoder
def logical_reps_from_v1(V1, stab_strings: Sequence[str]) -> Dict:
    """12 logical representatives of the [[20,6]] code, one X/Z pair per
    bulk input of V1, plus the central-bulk-qubit identification.

    Method (the flow/Choi machinery): every stabiliser web of V1 gives an
    (unsigned) stabilizer  P_bulk (x) P_boundary  of the encoder's Choi
    state.  Solving over GF(2) for web combos with P_bulk EXACTLY X_b
    (resp. Z_b, identity on the other five bulk legs) yields a boundary
    Pauli that is a logical X (resp. Z) representative of bulk qubit b.
    Different solutions differ by webs with trivial bulk part, i.e. by
    [[20,6]] stabilizers — the class pairing below is blind to those.

    Machine checks: every rep commutes with all 14 stabilizer generators;
    the 12 reps pair in the standard symplectic pattern
    <Xbar_a, Zbar_b> = delta_ab, <Xbar,Xbar> = <Zbar,Zbar> = 0.
    """
    from pyzx.web.compute import compute_stabilisers

    ins, outs = list(V1.inputs()), list(V1.outputs())
    assert len(ins) == 6 and len(outs) == 20
    webs = compute_stabilisers(V1)

    def leg(w, b):
        nb = next(iter(V1.neighbors(b)))
        return w.half_edges().get((nb, b), "I")

    bulk_rows, bnd_rows = [], []
    for w in webs:
        bx, bz = pauli_xz("".join(leg(w, b) for b in ins))
        px, pz = pauli_xz("".join(leg(w, b) for b in outs))
        bulk_rows.append(np.concatenate([bx, bz]))
        bnd_rows.append(np.concatenate([px, pz]))

    reps: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for b in range(6):
        for kind, col in (("X", b), ("Z", 6 + b)):
            target = np.zeros(12, dtype=np.uint8)
            target[col] = 1
            sol = _gf2_solve(bulk_rows, target)
            assert sol is not None, f"no web combo realises {kind}_bulk{b}"
            acc = np.zeros(40, dtype=np.uint8)
            for i, ci in enumerate(sol):
                if ci:
                    acc ^= bnd_rows[i]
            reps.append((f"{kind}bar(bulk{b})", acc[:20].copy(), acc[20:].copy()))

    stab_xz = [pauli_xz(s) for s in stab_strings]
    for name, rx, rz in reps:                      # centralize the code group
        for sx, sz in stab_xz:
            assert sym_pair(rx, rz, sx, sz) == 0, f"{name} fails to centralize"
    for i, (ni, xi, zi) in enumerate(reps):        # standard symplectic pairing
        for j, (nj, xj, zj) in enumerate(reps):
            want = 1 if (i // 2 == j // 2 and i != j) else 0
            assert sym_pair(xi, zi, xj, zj) == want, f"pairing {ni},{nj}"

    # central bulk input: spider nearest the boundary-spider centroid
    bulk_pos = np.array([[float(V1.row(next(iter(V1.neighbors(b))))),
                          float(V1.qubit(next(iter(V1.neighbors(b)))))]
                         for b in ins])
    out_pos = np.array([[float(V1.row(next(iter(V1.neighbors(b))))),
                         float(V1.qubit(next(iter(V1.neighbors(b)))))]
                        for b in outs])
    dists = np.linalg.norm(bulk_pos - out_pos.mean(axis=0), axis=1)
    i_c = int(np.argmin(dists))
    order = np.sort(dists)
    assert order[1] > 3 * order[0], "central bulk spider not clearly separated"
    return dict(reps=reps, central=i_c,
                central_distances=[round(float(d), 4) for d in dists],
                n_webs=len(webs))


def class_of(x: np.ndarray, z: np.ndarray, reps) -> np.ndarray:
    """12-bit logical class of a boundary Pauli: c[j] = <P, rep_j>."""
    return np.array([sym_pair(x, z, rx, rz) for _, rx, rz in reps],
                    dtype=np.uint8)


def loop_target(kind: str, b: int) -> np.ndarray:
    """Class vector of a loop measuring Xbar_b / Zbar_b: it anticommutes
    with the CONJUGATE rep only (Xbar_b pairs 1 with Zbar_b alone)."""
    t = np.zeros(12, dtype=np.uint8)
    t[2 * b + 1 if kind == "X" else 2 * b] = 1
    return t


# ------------------------------------------------------------ fold graphs
def build_fold_graph(nodes, edges, in_legs, out_legs):
    """Two copies of an all-Z / all-H-edge block, joined leg-to-leg by
    explicit SIMPLE (plain / Bell cup) seam wires.  Returns
    (gF, vmapF: (copy, node) -> vid, seams: [(vid0, vid1, side)])."""
    import pyzx as zx
    from pyzx.utils import EdgeType, VertexType

    gF = zx.Graph()
    vmapF: Dict[tuple, int] = {}
    for ci in (0, 1):
        for nd in nodes:
            vmapF[(ci, nd)] = gF.add_vertex(VertexType.Z, qubit=0, row=0)
    for ci in (0, 1):
        for e in edges:
            u, v = tuple(e)
            gF.add_edge((vmapF[(ci, u)], vmapF[(ci, v)]), EdgeType.HADAMARD)
    seams = []
    for side, legs in (("in", in_legs), ("out", out_legs)):
        for nd in legs:
            u, v = vmapF[(0, nd)], vmapF[(1, nd)]
            gF.add_edge((u, v), EdgeType.SIMPLE)
            seams.append((u, v, side))
    gF.set_inputs(())
    gF.set_outputs(())
    return gF, vmapF, seams


def aligned_web(gF, vmapF, g1, vmap1, web):
    """The open web placed identically in both copies; on each seam wire the
    label continues to the partner copy (folded_block.aligned_fold_web,
    generalised to arbitrary node keys)."""
    from pyzx.pauliweb import PauliWeb

    rev = {v: k for k, v in vmap1.items()}
    bnd = set(g1.inputs()) | set(g1.outputs())
    w = PauliWeb(gF)
    for (u, v), p in web.half_edges().items():
        if u in bnd:
            continue
        ku = rev[u]
        for ci in (0, 1):
            fu = vmapF[(ci, ku)]
            fv = vmapF[(1 - ci, ku)] if v in bnd else vmapF[(ci, rev[v])]
            w.add_half_edge((fu, fv), p)
    return w


# --------------------------------------------------------- the classifier
def classify_fold(tag: str, nodes, edges, in_legs, out_legs, g1, vmap1,
                  regions1, stabs1, detectors, reps_in, central: int,
                  code_rows_in: np.ndarray, weight_passes: int = 2) -> Dict:
    """Fold two copies of the given block, decompose the closed-web space
    (per-copy detectors / seam webs / logical loops), machine-check every
    claim, and export the decoding matrices.

    detectors    : [(name, [node, ...])] canonical per-copy detectors;
    reps_in      : the 12 logical reps ORDERED LIKE THE IN-LEGS;
    code_rows_in : 14 x 40 [x|z] stabilizer generators in in-leg order —
                   used to test seam webs against the truncated checks.
    """
    t0 = time.perf_counter()
    gF, vmapF, seams = build_fold_graph(nodes, edges, in_legs, out_legs)
    n_wires_expected = 2 * len(edges) + len(in_legs) + len(out_legs)
    wires = fold.sorted_wires(gF)
    assert len(wires) == gF.num_edges() == n_wires_expected
    assert len(set(wires)) == len(wires), "duplicate wire (half-edge double count)"

    colsF = fold.half_edge_cols(gF)
    cols1 = fold.half_edge_cols(g1)
    open_webs = list(stabs1) + list(regions1)
    M1 = np.array([fold.web_vec(w, cols1) for w in open_webs], dtype=np.uint8)
    in_b, out_b = list(g1.inputs()), list(g1.outputs())

    regsF = fold.closed_web_basis(gF)
    MallF = np.array([fold.web_vec(w, colsF) for w in regsF], dtype=np.uint8)
    d_all = fold.gf2_rank(MallF)

    # --- per-copy detectors ---------------------------------------------
    det_webs, det_names = [], []
    for ci in (0, 1):
        for name, det_nodes in detectors:
            w = fold.fire_web(gF, [vmapF[(ci, nd)] for nd in det_nodes])
            assert fold.web_is_closed(gF, w), f"copy{ci}:{name} not closed"
            assert fold.gf2_in_span(MallF, fold.web_vec(w, colsF))
            det_webs.append(w)
            det_names.append(f"copy{ci}:{name}")
    Mdet = np.array([fold.web_vec(w, colsF) for w in det_webs], dtype=np.uint8)
    r_det = fold.gf2_rank(Mdet)
    assert r_det == 2 * len(detectors), f"per-copy detector rank {r_det}"

    # --- seam webs: one-boundary open combos, mirror-completed ----------
    Mreg1 = (np.array([fold.web_vec(w, cols1) for w in regions1], dtype=np.uint8)
             if len(regions1) else np.zeros((0, M1.shape[1]), dtype=np.uint8))
    seam_webs, seam_sides = [], []
    per_seam = {"in": 0, "out": 0}
    for sname, avoid_bnd in (("in", out_b), ("out", in_b)):
        avoid = fold.boundary_wire_cols(g1, cols1, avoid_bnd)
        base = Mreg1.copy()
        raw = []
        for lam in fold.gf2_left_nullspace(M1[:, avoid]):
            w1 = fold.combine_webs(g1, open_webs, lam)
            vec = fold.web_vec(w1, cols1)
            if fold.gf2_in_span(base, vec):
                continue
            base = np.vstack([base, vec])
            raw.append(aligned_web(gF, vmapF, g1, vmap1, w1))
        raw = fold.greedy_weight_reduce(raw, det_webs, weight_passes)
        for w in raw:
            assert fold.web_is_closed(gF, w), f"{sname}-seam web not closed"
        seam_webs += raw
        seam_sides += [sname] * len(raw)
        per_seam[sname] = len(raw)

    # seam-restricted Paulis vs the truncated round-0 / final checks:
    # the label each seam web leaves on the seam wires, one Pauli per leg
    # (leg order = in_legs / out_legs), tested for row-space equality with
    # the code stabilizer group (out side: H-conjugated, x<->z, since the
    # out legs sit an odd number of sub-layer hops from the in legs).
    seam_wire_of = {}                       # (side, leg position) -> (u, v)
    pos = {"in": 0, "out": 0}
    for u, v, side in seams:
        seam_wire_of[(side, pos[side])] = (min(u, v), max(u, v))
        pos[side] += 1
    seam_pauli_rows = {"in": [], "out": []}
    seam_pauli_strs = []
    for w, side in zip(seam_webs, seam_sides):
        es = w.half_edges()
        s = "".join(es.get(seam_wire_of[(side, j)], "I")
                    for j in range(len(in_legs)))
        seam_pauli_strs.append(f"seam-{side}:{s}")
        x, z = pauli_xz(s)
        seam_pauli_rows[side].append(np.concatenate([x, z]))
    code_rows_out = np.concatenate([code_rows_in[:, 20:], code_rows_in[:, :20]],
                                   axis=1)          # H-conjugate: x <-> z
    seam_group_matches = {
        side: bool(len(seam_pauli_rows[side]) and rowspace_equal(
            np.array(seam_pauli_rows[side], dtype=np.uint8),
            code_rows_in if side == "in" else code_rows_out))
        for side in ("in", "out")}

    Mdet_all = np.vstack([Mdet] + [fold.web_vec(w, colsF) for w in seam_webs])
    r_det_all = fold.gf2_rank(Mdet_all)
    assert r_det_all == r_det + len(seam_webs), "seam webs not independent"

    # --- logical loops: quotient of aligned open webs by det+seam -------
    reps_open = []
    base = Mdet_all.copy()
    for w1 in open_webs:
        fw = aligned_web(gF, vmapF, g1, vmap1, w1)
        vec = fold.web_vec(fw, colsF)
        if not fold.gf2_in_span(base, vec):
            base = np.vstack([base, vec])
            reps_open.append(w1)
    assert len(reps_open) == 12, f"logical quotient dim {len(reps_open)} != 12"

    cls_rows = []
    for w1 in reps_open:
        x, z = pauli_xz(fold.boundary_pauli_string(g1, w1, in_b))
        cls_rows.append(class_of(x, z, reps_in))
    assert fold.gf2_rank(np.array(cls_rows, dtype=np.uint8)) == 12, \
        "quotient classes not independent — naming would be ill-defined"

    log_webs, log_names, log_docs = [], [], []
    for b in range(6):
        for kind in ("X", "Z"):
            target = loop_target(kind, b)
            sol = _gf2_solve(cls_rows, target)
            assert sol is not None
            w1 = fold.combine_webs(g1, reps_open, sol)
            x, z = pauli_xz(fold.boundary_pauli_string(g1, w1, in_b))
            assert (class_of(x, z, reps_in) == target).all()
            fw = aligned_web(gF, vmapF, g1, vmap1, w1)
            fw = fold.greedy_weight_reduce([fw], det_webs + seam_webs,
                                           weight_passes)[0]
            assert fold.web_is_closed(gF, fw)
            assert not fold.gf2_in_span(Mdet_all, fold.web_vec(fw, colsF))
            log_webs.append(fw)
            log_names.append(f"loop-{kind}bar(bulk{b})")
            log_docs.append(dict(
                name=f"loop-{kind}bar(bulk{b})",
                in_pauli=xz_to_str(x, z),
                out_pauli=fold.boundary_pauli_string(g1, w1, out_b),
                n_half_edges=len(fw.half_edges())))

    Mfull = np.vstack([Mdet_all] + [fold.web_vec(w, colsF) for w in log_webs])
    decomposition = dict(per_copy_detectors=int(r_det),
                         seam_in=per_seam["in"], seam_out=per_seam["out"],
                         logical=12,
                         total=int(r_det + len(seam_webs) + 12), d_all=int(d_all))
    assert fold.gf2_rank(Mfull) == decomposition["total"] == d_all, decomposition
    assert all(fold.gf2_in_span(Mfull, r) for r in MallF), \
        "closed-web basis not fully classified"

    # --- decoding matrices ----------------------------------------------
    wires, H_det, O_all = fold.decoding_matrices(gF, det_webs + seam_webs,
                                                 log_webs)
    idx_c = [log_names.index(f"loop-Xbar(bulk{central})"),
             log_names.index(f"loop-Zbar(bulk{central})")]
    O_central = O_all[idx_c].copy()
    covH = H_det.any(axis=0)
    n_silent_all = int((~covH & O_all.any(axis=0)).sum())
    n_silent_central = int((~covH & O_central.any(axis=0)).sum())
    assert n_silent_all == 0, f"{tag}: fold has silent logical components"

    return dict(tag=tag, g=gF, vmapF=vmapF, seams=seams,
                wires=wires, H_det=H_det, O_all=O_all, O_central=O_central,
                central_rows=idx_c,
                det_webs=det_webs, seam_webs=seam_webs, log_webs=log_webs,
                det_names=det_names + seam_pauli_strs, log_names=log_names,
                log_docs=log_docs, decomposition=decomposition,
                per_seam=per_seam, seam_group_matches=seam_group_matches,
                seam_pauli_strs=seam_pauli_strs,
                n_silent_all=n_silent_all, n_silent_central=n_silent_central,
                wall_s=round(time.perf_counter() - t0, 1))


# ----------------------------------------------------------- the builders
def _global_pieces(K: int = 2):
    """The nb24 general foliated [[20,6]] block + its open webs."""
    import compat
    compat.install_pyzx_rules_shim()
    from happy_spacetime import build_happy_encoder
    from pyzx.web.compute import compute_detecting_regions, compute_stabilisers
    from webs_to_checks import code_stabiliser_basis

    import src.foliated_general as fg

    V1 = build_happy_encoder(1)
    if isinstance(V1, tuple):
        V1 = V1[0]
    _, strings = code_stabiliser_basis(V1)
    assert len(strings) == 14 and len(strings[0]) == 20
    sx, sz = fg.pauli_strings_to_sxsz(strings)
    b1 = fg.build_general_foliated_block(sx, sz, K)
    assert all(a == "X" for a in b1.anc_basis), \
        "odd-Y generator would make to_zx inexact"
    g1, vmap1 = fg.to_zx(b1)
    regions1 = list(compute_detecting_regions(g1))
    stabs1 = list(compute_stabilisers(g1))
    assert len(regions1) == 14 * (K - 1) and len(stabs1) == 40
    detectors = [(f"D(c={d['c']},k={d['k']})", d["nodes"])
                 for d in b1.detectors]
    return V1, strings, b1, g1, vmap1, regions1, stabs1, detectors


def _blockwise_pieces(V1, K: int = 2):
    """The nb25 6-tile cup patch + its open webs + the geometric leg
    correspondence patch-boundary-slot -> V1 output (nb25 Step 1)."""
    from pyzx.web.compute import compute_detecting_regions, compute_stabilisers

    LABELS = ["C", "N0", "N1", "N2", "N3", "N4"]
    CONTR = [("C", k, f"N{k}", 0) for k in range(5)]
    comp, info = build_slab_patch(LABELS, CONTR, "cup", rounds=K)
    assert (info["n_nodes"], info["n_edges"]) == (158, 354)
    assert info["n_open_worldlines"] == 20

    # tiling geometry off V1 (identical to nb25 cell 4)
    bulk_pos = np.array([[float(V1.row(next(iter(V1.neighbors(b))))),
                          float(V1.qubit(next(iter(V1.neighbors(b)))))]
                         for b in V1.inputs()])
    out_pos = np.array([[float(V1.row(next(iter(V1.neighbors(b))))),
                         float(V1.qubit(next(iter(V1.neighbors(b)))))]
                        for b in V1.outputs()])
    i_c = int(np.argmin(np.linalg.norm(bulk_pos - out_pos.mean(axis=0), axis=1)))
    c0 = bulk_pos[i_c]
    ring_idx = [i for i in range(6) if i != i_c]
    phis = {i: math.atan2(bulk_pos[i][1] - c0[1], bulk_pos[i][0] - c0[0])
            for i in ring_idx}
    ring_order = sorted(ring_idx, key=lambda i: phis[i])
    assign = {}
    for j in range(20):
        dd = [float(np.linalg.norm(out_pos[j] - bulk_pos[i])) for i in ring_idx]
        i_t = ring_idx[int(np.argmin(dd))]
        ang = math.atan2(out_pos[j][1] - bulk_pos[i_t][1],
                         out_pos[j][0] - bulk_pos[i_t][0])
        inward = math.atan2(c0[1] - bulk_pos[i_t][1], c0[0] - bulk_pos[i_t][0])
        assign[j] = (i_t, (ang - inward) % (2 * math.pi))
    v1_leg = {}
    for k, i_t in enumerate(ring_order):
        legs = sorted((d, j) for j, (it, d) in assign.items() if it == i_t)
        assert len(legs) == 4
        for qi, (_d, j) in enumerate(legs):
            v1_leg[(k, qi + 1)] = j

    contracted = {(la, qa) for la, qa, _, _ in CONTR} | \
                 {(lb, qb) for _, _, lb, qb in CONTR}
    boundary = [(lbl, q) for lbl in LABELS for q in range(5)
                if (lbl, q) not in contracted]
    assert [comp.rep[(lbl, ("data", q, 0))] for lbl, q in boundary] == \
        list(comp.open_inputs), "boundary slot order != open_inputs order"
    perm = [v1_leg[(int(lbl[1:]), q)] for lbl, q in boundary]
    assert sorted(perm) == list(range(20))

    g1, vmap1 = composite_to_zx(comp)
    regions1 = list(compute_detecting_regions(g1))
    stabs1 = list(compute_stabilisers(g1))
    assert len(regions1) == 24 and len(stabs1) == 40

    detectors = []
    for lbl in LABELS:
        for d in comp.blocks[lbl].detectors:
            detectors.append((f"{lbl}:D(c={d['c']},k={d['k']})",
                              [comp.rep[(lbl, nd)] for nd in d["nodes"]]))
    assert len(detectors) == 24
    return comp, g1, vmap1, regions1, stabs1, detectors, boundary, perm, \
        contracted


def permute_reps(reps, perm: Sequence[int]):
    """Reps (V1-output qubit order) re-indexed to a boundary-slot order:
    slot i of the new order is V1 output perm[i]."""
    p = np.array(perm)
    return [(name, x[p].copy(), z[p].copy()) for name, x, z in reps]


# ------------------------------------------------------- wire annotations
def wire_rows_fold(result, rev_desc) -> List[Tuple[str, str, str, str]]:
    """(desc_u, desc_v, edge_type, category) per wire of a fold result."""
    from pyzx.utils import EdgeType

    g = result["g"]
    seam_of = {(min(u, v), max(u, v)): side for u, v, side in result["seams"]}
    out = []
    for u, v in result["wires"]:
        et = "S" if g.edge_type(g.edge(u, v)) == EdgeType.SIMPLE else "H"
        cat = seam_of.get((u, v))
        if cat is None:
            cat = rev_desc["category"](u, v)
        else:
            cat = f"seam-{cat}"
        out.append((rev_desc["name"](u), rev_desc["name"](v), et, cat))
    return out


def build_open_control(comp, g1, vmap1, regions1, stabs1, reps_block,
                       central: int, weight_passes: int = 2) -> Dict:
    """The nb25 patch as-is: detectors = its 24 closed detecting regions
    (all it has); observables = 12 flow webs picked from the 40 stabiliser
    webs by logical class of their in-boundary Pauli, weight-reduced modulo
    the REGIONS ONLY (adding a region never changes what an observable
    measures; adding another stabiliser web would — the representative is
    pinned and documented, caveat per nb23)."""
    t0 = time.perf_counter()
    in_b, out_b = list(g1.inputs()), list(g1.outputs())
    cls_rows = []
    for w in stabs1:
        x, z = pauli_xz(fold.boundary_pauli_string(g1, w, in_b))
        cls_rows.append(class_of(x, z, reps_block))
    assert fold.gf2_rank(np.array(cls_rows, dtype=np.uint8)) == 12

    log_webs, log_names, log_docs = [], [], []
    for b in range(6):
        for kind in ("X", "Z"):
            target = loop_target(kind, b)
            sol = _gf2_solve(cls_rows, target)
            assert sol is not None
            w = fold.combine_webs(g1, stabs1, sol)
            w = fold.greedy_weight_reduce([w], list(regions1), weight_passes)[0]
            x, z = pauli_xz(fold.boundary_pauli_string(g1, w, in_b))
            assert (class_of(x, z, reps_block) == target).all()
            log_webs.append(w)
            log_names.append(f"flow-{kind}bar(bulk{b})")
            log_docs.append(dict(
                name=f"flow-{kind}bar(bulk{b})",
                in_pauli=xz_to_str(x, z),
                out_pauli=fold.boundary_pauli_string(g1, w, out_b),
                n_half_edges=len(w.half_edges())))

    wires, H_det, O_all = fold.decoding_matrices(g1, list(regions1), log_webs)
    assert len(wires) == g1.num_edges() == 354 + 40
    assert len(set(wires)) == len(wires)
    idx_c = [log_names.index(f"flow-Xbar(bulk{central})"),
             log_names.index(f"flow-Zbar(bulk{central})")]
    O_central = O_all[idx_c].copy()
    covH = H_det.any(axis=0)
    silent_cols_all = np.flatnonzero(~covH & O_all.any(axis=0))
    silent_cols_c = np.flatnonzero(~covH & O_central.any(axis=0))
    bnd = set(in_b) | set(out_b)
    leg_wire_idx = [i for i, (u, v) in enumerate(wires)
                    if u in bnd or v in bnd]
    silent_wires = sorted({int(c) // 2 for c in silent_cols_all})
    return dict(tag="blockwise_open", g=g1, wires=wires,
                H_det=H_det, O_all=O_all, O_central=O_central,
                central_rows=idx_c,
                det_names=[f"region{i}" for i in range(len(regions1))],
                log_names=log_names, log_docs=log_docs,
                n_silent_all=int(silent_cols_all.size),
                n_silent_central=int(silent_cols_c.size),
                silent_wires=silent_wires,
                n_silent_wires_on_legs=sum(w in leg_wire_idx
                                           for w in silent_wires),
                leg_wire_idx=leg_wire_idx,
                wall_s=round(time.perf_counter() - t0, 1))


# ------------------------------------------------- erasure-support profile
def _int_rank(cols: List[int]) -> int:
    """Rank over GF(2) of bitmask-encoded columns."""
    basis: List[int] = []
    r = 0
    for v in cols:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
            basis.sort(reverse=True)
            r += 1
    return r


def erasure_support_counts(H: np.ndarray, O: np.ndarray, cats,
                           pairs: bool = True, cross_check: int = 200) -> Dict:
    """Which single wires / wire pairs, when erased, support a fault that
    commutes with every detector yet flips an observable — the FUNDAMENTAL
    erasure failure (even ML decoding then fails with prob >= 1/2).  These
    counts are the leading low-p_e behavior of the LER curve:
    LER ~ (n_single) p/2 + O(p^2), or O(p^2) with the pair count when
    n_single = 0.

    Columns are packed as ints (bitmask over stacked H;O rows): the erased
    set supports a logical iff rank([H;O]|_cols) > rank(H|_cols).  The
    packing is cross-checked against erasure_sim.erased_set_supports_logical
    on `cross_check` random sets."""
    from collections import Counter

    nH = H.shape[0]
    n_wires = H.shape[1] // 2
    HO = np.vstack([H, O]).astype(np.uint8)
    packed = []                       # per column: (H bits, H+O bits)
    for c in range(HO.shape[1]):
        bits = int("".join("1" if b else "0" for b in HO[:, c]), 2) \
            if HO[:, c].any() else 0
        packed.append(bits)
    # bit i of a packed value corresponds to row (nrows-1-i): H rows are the
    # HIGH bits, O rows the LOW bits, so >> (n_O_rows) keeps the H part.
    def supports(wires_sel) -> bool:
        cols = []
        for w in wires_sel:
            cols += [packed[2 * w], packed[2 * w + 1]]
        rH = _int_rank([c >> (HO.shape[0] - nH) for c in cols])
        rHO = _int_rank(cols)
        return rHO > rH

    rng = np.random.default_rng(0)
    from .erasure_sim import erased_set_supports_logical
    for _ in range(cross_check):
        k = int(rng.integers(1, 4))
        sel = sorted(rng.choice(n_wires, size=k, replace=False).tolist())
        assert supports(sel) == erased_set_supports_logical(H, O, sel), \
            f"packed-rank method disagrees on {sel}"

    singles = [i for i in range(n_wires) if supports([i])]
    out = dict(n_single=len(singles),
               singles_by_category=dict(Counter(str(cats[i]) for i in singles)))
    if pairs:
        pair_cats: Counter = Counter()
        n_pairs = 0
        single_set = set(singles)
        for i in range(n_wires):
            for j in range(i + 1, n_wires):
                if i in single_set or j in single_set:
                    continue          # count only IRREDUCIBLE pairs
                if supports([i, j]):
                    n_pairs += 1
                    pair_cats[tuple(sorted((str(cats[i]), str(cats[j]))))] += 1
        out["n_irreducible_pairs"] = n_pairs
        out["pairs_by_category"] = {" + ".join(k): int(v)
                                    for k, v in sorted(pair_cats.items())}
    return out


# ------------------------------------------------------------------ export
def _save_npz(path, result, wire_rows) -> None:
    np.savez_compressed(
        path,
        H_det=result["H_det"].astype(np.uint8),
        O_all=result["O_all"].astype(np.uint8),
        O_central=result["O_central"].astype(np.uint8),
        central_rows=np.array(result["central_rows"], dtype=np.int64),
        wires_u=np.array([u for u, _ in result["wires"]], dtype=np.int64),
        wires_v=np.array([v for _, v in result["wires"]], dtype=np.int64),
        wire_desc_u=np.array([r[0] for r in wire_rows]),
        wire_desc_v=np.array([r[1] for r in wire_rows]),
        wire_type=np.array([r[2] for r in wire_rows]),
        wire_category=np.array([r[3] for r in wire_rows]),
        det_names=np.array(result["det_names"]),
        log_names=np.array(result["log_names"]))


def _smoke_sim(result, p_e: float = 0.05, shots: int = 200,
               seed: int = 12345) -> Dict:
    """Tiny pluggability check of (H, O) against the pinned nb23 simulator —
    NOT the deliverable curve (that is the next task's Monte Carlo)."""
    from .erasure_sim import run_point

    out = {}
    for oname in ("O_all", "O_central"):
        pt = run_point(result["H_det"], result[oname], p_e, shots, seed)
        out[oname] = dict(p_e=p_e, shots=shots, fails=pt["fails"],
                          ler=pt["ler"])
    return out


def build_all(results_dir, weight_passes: int = 2,
              smoke: bool = True) -> Dict:
    """Build, machine-check, and save all three structures + manifest."""
    import pathlib
    results_dir = pathlib.Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    manifest: Dict = dict(
        built="r1_fold_decoding.build_all",
        rounds=2,
        error_model=PRIOR_NOTE,
        note=("single r=1 instance: NO threshold claim is possible (that "
              "needs radius scaling); deliverable is LER vs p_e at matched "
              "noise, folded on both time sides in the fold structures"))

    # ---------------- shared ingredients --------------------------------
    V1, strings, b1, g1g, vmap1g, regs1g, stabs1g, dets_g = _global_pieces(2)
    rep_info = logical_reps_from_v1(V1, strings)
    reps_v1 = rep_info["reps"]
    central = rep_info["central"]
    code_rows_v1 = np.array(
        [np.concatenate(pauli_xz(s)) for s in strings], dtype=np.uint8)
    manifest["central_qubit"] = dict(
        index_in_v1_inputs=central,
        method=("bulk input whose spider is nearest the centroid of the 20 "
                "boundary spiders (Poincare-disk center); loop naming via "
                "symplectic pairing against logical reps extracted from "
                "V1's stabiliser webs (bulk part solved to be exactly "
                "X_b / Z_b)"),
        distances_to_boundary_centroid=rep_info["central_distances"],
        n_v1_stabiliser_webs=rep_info["n_webs"])

    # in-boundary Paulis of every open web must centralize the code group
    # (validates the machinery + orderings against the diagrams themselves)
    for w in stabs1g:
        x, z = pauli_xz(fold.boundary_pauli_string(g1g, w, list(g1g.inputs())))
        for s in strings:
            sx, sz = pauli_xz(s)
            assert sym_pair(x, z, sx, sz) == 0

    # ---------------- (1) GLOBAL-FOLD -----------------------------------
    res_g = classify_fold(
        "global_fold", b1.nodes, b1.edges, b1.in_legs, b1.out_legs,
        g1g, vmap1g, regs1g, stabs1g, dets_g, reps_v1, central,
        code_rows_v1, weight_passes)
    rev_g = {v: k for k, v in res_g["vmapF"].items()}

    def gname(vid):
        ci, (kind, idx, t) = rev_g[vid]
        return f"c{ci}.{'d' if kind == 'data' else 'a'}{idx}.t{t}"

    def gcat(u, v):
        (_, (ku, iu, _tu)), (_, (kv, iv, _tv)) = rev_g[u], rev_g[v]
        if ku == kv == "data":
            return "chain"
        return "anc-anc" if ku == kv else "anc-data"

    rows_g = wire_rows_fold(res_g, {"name": gname, "category": gcat})
    assert len(res_g["wires"]) == 2 * 412 + 40

    # ---------------- (2) BLOCKWISE-FOLD --------------------------------
    comp, g1b, vmap1b, regs1b, stabs1b, dets_b, boundary, perm, contracted \
        = _blockwise_pieces(V1, 2)
    reps_block = permute_reps(reps_v1, perm)
    code_rows_block = code_rows_v1[:, np.array(perm + [20 + j for j in perm])]
    # the patch's open webs must centralize the PERMUTED code group — this
    # machine-checks the geometric leg correspondence end-to-end
    for w in stabs1b:
        x, z = pauli_xz(fold.boundary_pauli_string(g1b, w, list(g1b.inputs())))
        for r in code_rows_block:
            assert sym_pair(x, z, r[:20], r[20:]) == 0

    manifest["boundary_orders"] = dict(
        global_fold="V1 output order (code_stabiliser_basis qubit order)",
        blockwise="patch boundary slots (N0,q1..q4),(N1,q1..q4),...,(N4,q1..q4)",
        blockwise_slot_to_v1_output=perm,
        note=("in_pauli/out_pauli strings and seam_paulis are written in "
              "each structure's own boundary order; slot i of the blockwise "
              "order is V1 output perm[i] (geometric leg correspondence, "
              "machine-checked by the centralizer test)"))

    res_b = classify_fold(
        "blockwise_fold", comp.nodes, comp.edges,
        list(comp.open_inputs), list(comp.open_outputs),
        g1b, vmap1b, regs1b, stabs1b, dets_b, reps_block, central,
        code_rows_block, weight_passes)
    rev_b = {v: k for k, v in res_b["vmapF"].items()}
    bond_nodes = {comp.rep[(lbl, ("data", q, t))]
                  for (lbl, q) in contracted for t in range(4)}

    def bname(vid):
        ci, nd = rev_b[vid]
        lbl, (kind, idx, t) = nd
        star = "*" if len(comp.members.get(nd, [nd])) > 1 else ""
        return f"c{ci}.{lbl}.{'d' if kind == 'data' else 'a'}{idx}.t{t}{star}"

    def bcat(u, v):
        (_, ndu), (_, ndv) = rev_b[u], rev_b[v]
        ku, kv = ndu[1][0], ndv[1][0]
        if ku == kv == "data":
            if ndu in bond_nodes and ndv in bond_nodes:
                return "bond-loop"
            return "chain"
        return "anc-anc" if ku == kv else "anc-data"

    rows_b = wire_rows_fold(res_b, {"name": bname, "category": bcat})
    assert len(res_b["wires"]) == 2 * 354 + 40
    n_bond = sum(r[3] == "bond-loop" for r in rows_b)
    assert n_bond == 60, f"bond-loop wires {n_bond} != 2 copies x 5 bonds x 6"

    # ---------------- (3) BLOCKWISE-OPEN control ------------------------
    res_o = build_open_control(comp, g1b, vmap1b, regs1b, stabs1b,
                               reps_block, central, weight_passes)
    rev1b = {v: k for k, v in vmap1b.items()}
    bnd1b = set(g1b.inputs()) | set(g1b.outputs())

    def oname(vid):
        if vid in bnd1b:
            return f"leg{vid}"
        lbl, (kind, idx, t) = rev1b[vid]
        star = "*" if len(comp.members.get(rev1b[vid], [rev1b[vid]])) > 1 else ""
        return f"{lbl}.{'d' if kind == 'data' else 'a'}{idx}.t{t}{star}"

    def ocat(u, v):
        if u in bnd1b or v in bnd1b:
            return "leg"
        ndu, ndv = rev1b[u], rev1b[v]
        ku, kv = ndu[1][0], ndv[1][0]
        if ku == kv == "data":
            if ndu in bond_nodes and ndv in bond_nodes:
                return "bond-loop"
            return "chain"
        return "anc-anc" if ku == kv else "anc-data"

    from pyzx.utils import EdgeType as _ET
    rows_o = [(oname(u), oname(v),
               "S" if res_o["g"].edge_type(res_o["g"].edge(u, v)) == _ET.SIMPLE
               else "H", ocat(u, v)) for u, v in res_o["wires"]]
    assert sum(r[3] == "bond-loop" for r in rows_o) == 30

    # ---------------- save ----------------------------------------------
    for res, rows, fname in ((res_g, rows_g, "r1_global_fold_k2.npz"),
                             (res_b, rows_b, "r1_blockwise_fold_k2.npz"),
                             (res_o, rows_o, "r1_blockwise_open_k2.npz")):
        _save_npz(results_dir / fname, res, rows)
        from collections import Counter
        cats = Counter(r[3] for r in rows)
        ent = dict(
            npz=fname,
            n_wires=len(res["wires"]),
            n_fault_components=int(res["H_det"].shape[1]),
            H_det_shape=list(res["H_det"].shape),
            O_all_shape=list(res["O_all"].shape),
            O_central_rows=res["central_rows"],
            wire_categories={k: int(v) for k, v in sorted(cats.items())},
            n_silent_vs_O_all=res["n_silent_all"],
            n_silent_vs_O_central=res["n_silent_central"],
            observables=res["log_docs"],
            wall_s=res["wall_s"])
        if "decomposition" in res:
            ent["dim_decomposition"] = res["decomposition"]
            ent["seam_webs_per_seam"] = res["per_seam"]
            ent["seam_group_equals_code_group"] = res["seam_group_matches"]
            ent["seam_paulis"] = res["seam_pauli_strs"]
        else:
            ent["n_detectors"] = int(res["H_det"].shape[0])
            ent["silent_wires"] = res["silent_wires"]
            ent["n_silent_wires_on_open_legs"] = res["n_silent_wires_on_legs"]
            ent["caveat"] = ("observable representatives are pinned but "
                            "representative-DEPENDENT (nb23 caveat): adding "
                            "a stabiliser web would change what they "
                            "measure at the time boundary")
        if smoke:
            ent["smoke_sim"] = _smoke_sim(res)
        cats = [r[3] for r in rows]
        ent["erasure_support"] = dict(
            note=("wires/pairs whose HERALDED erasure supports a fault that "
                  "no detector sees but an observable feels — even ML fails "
                  "with prob >= 1/2 on such shots; pairs counted only if "
                  "neither member is already a supporting single"),
            O_all=erasure_support_counts(res["H_det"], res["O_all"], cats),
            O_central=erasure_support_counts(res["H_det"], res["O_central"],
                                             cats))
        manifest[res["tag"]] = ent

    manifest["wall_s_total"] = round(time.perf_counter() - t0, 1)
    from .erasure_sim import versions
    manifest["versions"] = versions()
    with open(results_dir / "r1_fold_decoding_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    manifest = build_all(root / "results")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ()}, indent=1, default=str))


if __name__ == "__main__":
    main()
