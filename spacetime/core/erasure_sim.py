"""Pure-erasure BP+OSD-0 Monte Carlo on web-derived check matrices.

Error model (user-pinned, identical to the nb22 export):
  * fault locations = the EDGES (wires) of the ZX diagram, one per physical
    wire, seam wires included;
  * pure erasure is HERALDED: each wire is erased iid with probability p_e;
    an erased wire carries a uniform Pauli from {I, X, Y, Z}, i.e. its X- and
    Z-components are flipped independently with probability 1/2 each;
  * columns: 2 per wire — col 2i = (wire i, X flip), col 2i+1 = (wire i, Z
    flip);
  * decoder: ldpc BpOsdDecoder, OSD method 'OSD_0' (order 0), BP max_iter=10
    (minimum_sum, ms_scaling_factor 1.0, parallel schedule), with PER-SHOT
    channel priors: 0.5 on both components of every erased wire, 1e-9
    elsewhere;
  * a shot FAILS iff O @ (e_hat XOR e) != 0 (mod 2) for the observable
    matrix O.

Fixed-side H-edge convention (from src/folded_block.decoding_matrices): a
fault component on wire (u, v) is tested against the web label on the
half-edge at the LOWER vertex id.  Sliding a fault through a Hadamard
conjugates fault and label equally, so anticommutation is side-invariant.

The module also rebuilds the fold decoding matrices for arbitrary K
(`build_fold_decoding`, the nb22 pipeline generalised) and the OPEN-block
boundary-artifact control (`build_open_control`).
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import folded_block as fold
from .foliated_block import build_foliated_perfect_block, to_zx

PRIOR_ERASED = 0.5
PRIOR_INTACT = 1e-9

DECODER_SPEC = dict(name="ldpc.BpOsdDecoder", bp_method="minimum_sum",
                    ms_scaling_factor=1.0, schedule="parallel", max_iter=10,
                    osd_method="OSD_0", osd_order=0,
                    priors="per-shot: 0.5 on erased components, 1e-9 else")


# ------------------------------------------------------------------ sampling
def sample_shot(rng: np.random.Generator, n_wires: int, p_e: float):
    """One heralded-erasure shot.

    Returns (erased_idx, e) where erased_idx lists the erased wires and e is
    the length-2*n_wires GF(2) fault vector (uniform Pauli per erased wire =
    independent 1/2 flips of the X- and Z-components)."""
    erased_idx = np.flatnonzero(rng.random(n_wires) < p_e)
    e = np.zeros(2 * n_wires, dtype=np.uint8)
    if erased_idx.size:
        bits = rng.integers(0, 2, size=(erased_idx.size, 2), dtype=np.uint8)
        e[2 * erased_idx] = bits[:, 0]
        e[2 * erased_idx + 1] = bits[:, 1]
    return erased_idx, e


def channel_for(erased_idx: np.ndarray, n_wires: int) -> np.ndarray:
    """Per-shot BP priors: 0.5 on both components of erased wires, 1e-9 else."""
    ch = np.full(2 * n_wires, PRIOR_INTACT)
    ch[2 * erased_idx] = PRIOR_ERASED
    ch[2 * erased_idx + 1] = PRIOR_ERASED
    return ch


def make_decoder(H: np.ndarray, channel: Optional[np.ndarray] = None):
    """A BpOsdDecoder per the pinned spec (channel defaults to all-intact)."""
    from ldpc import BpOsdDecoder

    if channel is None:
        channel = np.full(H.shape[1], PRIOR_INTACT)
    return BpOsdDecoder(H, error_channel=list(channel),
                        max_iter=DECODER_SPEC["max_iter"],
                        bp_method=DECODER_SPEC["bp_method"],
                        ms_scaling_factor=DECODER_SPEC["ms_scaling_factor"],
                        schedule=DECODER_SPEC["schedule"],
                        osd_method=DECODER_SPEC["osd_method"],
                        osd_order=DECODER_SPEC["osd_order"])


# ---------------------------------------------------------------- simulation
def run_point(H: np.ndarray, O: np.ndarray, p_e: float, shots: int,
              seed: int, rebuild_per_shot: bool = False,
              check_syndrome: bool = True) -> Dict:
    """Monte-Carlo one (structure, p_e) point.  Deterministic in `seed`.

    rebuild_per_shot=False uses decoder.update_channel_probs (machine-checked
    against per-shot reconstruction in the notebook); True rebuilds the
    decoder every shot."""
    H = np.ascontiguousarray(H, dtype=np.uint8)
    O = np.ascontiguousarray(O, dtype=np.uint8)
    n_wires = H.shape[1] // 2
    rng = np.random.default_rng(seed)
    dec = None if rebuild_per_shot else make_decoder(H)
    fails = 0
    t0 = time.perf_counter()
    for _ in range(shots):
        erased_idx, e = sample_shot(rng, n_wires, p_e)
        s = (H @ e) % 2
        ch = channel_for(erased_idx, n_wires)
        if rebuild_per_shot:
            dec = make_decoder(H, ch)
        else:
            dec.update_channel_probs(ch)
        e_hat = dec.decode(s).astype(np.uint8)
        if check_syndrome:
            assert (((H @ e_hat) % 2) == s).all(), "decoder returned inconsistent correction"
        if (((O @ (e_hat ^ e)) % 2) != 0).any():
            fails += 1
    wall = time.perf_counter() - t0
    return dict(p_e=float(p_e), shots=int(shots), fails=int(fails),
                ler=fails / shots, seed=int(seed), wall_s=round(wall, 3))


def erased_set_supports_logical(H: np.ndarray, O: np.ndarray,
                                erased_idx: Sequence[int]) -> bool:
    """True iff some fault supported on the erased wires commutes with every
    detector but flips an observable — the FUNDAMENTAL erasure failure: the
    syndrome cannot distinguish logical cosets, so no
    decoder fails with probability >= 1/2 on such a shot."""
    cols = sorted([2 * i for i in erased_idx] + [2 * i + 1 for i in erased_idx])
    if not cols:
        return False
    A = np.vstack([H, O])[:, cols]
    for v in fold.gf2_nullspace(A[: H.shape[0]]):
        if ((A[H.shape[0]:] @ v) % 2).any():
            return True
    return False


def erasure_distance_leq(H: np.ndarray, O: np.ndarray, wmax: int = 2) -> Optional[int]:
    """Smallest w <= wmax s.t. some w-wire erasure supports a logical
    (None if all size-<=wmax sets are safe)."""
    from itertools import combinations

    n_wires = H.shape[1] // 2
    for w in range(1, wmax + 1):
        for c in combinations(range(n_wires), w):
            if erased_set_supports_logical(H, O, c):
                return w
    return None


def wilson_interval(fails: int, shots: int, z: float = 1.959963984540054):
    """Wilson 95% score interval for a binomial proportion."""
    if shots == 0:
        return 0.0, 1.0
    p = fails / shots
    z2 = z * z
    denom = 1 + z2 / shots
    centre = (p + z2 / (2 * shots)) / denom
    half = z * np.sqrt(p * (1 - p) / shots + z2 / (4 * shots * shots)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)


# ---------------------------------------------- fold matrices for arbitrary K
def _open_block_webs(rounds: int):
    """Open K-round block + its detecting regions and stabiliser webs."""
    from pyzx.web.compute import compute_detecting_regions, compute_stabilisers

    block = build_foliated_perfect_block(rounds)
    g1, vmap1 = to_zx(block, open_in=True, open_out=True, measured_as="effect")
    regions = compute_detecting_regions(g1)
    stabs = compute_stabilisers(g1)
    return block, g1, vmap1, list(regions), list(stabs)


def _cls_of(pauli_string: str) -> Tuple[int, int]:
    """(anticommutes with Z...Z, anticommutes with X...X) of a Pauli string —
    the (Xbar, Zbar) logical class of a [[5,1,3]] boundary Pauli."""
    ax = sum(c in "XY" for c in pauli_string) % 2
    az = sum(c in "ZY" for c in pauli_string) % 2
    return ax, az


def build_fold_decoding(rounds: int = 2, join: str = "plain",
                        weight_reduce_passes: int = 4) -> Dict:
    """The nb22 pipeline for arbitrary K: fold two K-round blocks in-in and
    out-out (plain/Bell seams), classify the closed-web space, and export the
    decoding matrices.  Every structural claim is asserted, not assumed."""
    K = rounds
    block, g1, vmap1, regions1, stabs1 = _open_block_webs(K)
    open_webs = stabs1 + regions1
    cols1 = fold.half_edge_cols(g1)
    M1 = np.array([fold.web_vec(w, cols1) for w in open_webs], dtype=np.uint8)
    in_b, out_b = list(g1.inputs()), list(g1.outputs())

    fb = fold.build_folded(rounds=K, join=join)
    gF, vmapF = fold.to_zx_folded(fb)
    fold.embed_fold_prisms(gF, vmapF)
    colsF = fold.half_edge_cols(gF)

    regsF = fold.closed_web_basis(gF)
    MallF = np.array([fold.web_vec(w, colsF) for w in regsF], dtype=np.uint8)
    d_all = fold.gf2_rank(MallF)

    # --- per-copy detectors --------------------------------------------------
    det_named = fold.canonical_detector_webs(fb, gF, vmapF)
    det_names = [n for n, _ in det_named]
    det_webs = [w for _, w in det_named]
    for w in det_webs:
        assert fold.web_is_closed(gF, w)
        assert fold.gf2_in_span(MallF, fold.web_vec(w, colsF))
    Mdet = np.array([fold.web_vec(w, colsF) for w in det_webs], dtype=np.uint8)
    assert fold.gf2_rank(Mdet) == 8 * (K - 1)

    # --- seam webs: open combos supported on ONE boundary, mirror-completed --
    Mreg1 = (np.array([fold.web_vec(w, cols1) for w in regions1], dtype=np.uint8)
             if regions1 else np.zeros((0, M1.shape[1]), dtype=np.uint8))
    raw_seam, raw_kind = [], []
    for sname, avoid_bnd in (("in", out_b), ("out", in_b)):
        avoid = fold.boundary_wire_cols(g1, cols1, avoid_bnd)
        base = Mreg1.copy()
        for lam in fold.gf2_left_nullspace(M1[:, avoid]):
            w1 = fold.combine_webs(g1, open_webs, lam)
            vec = fold.web_vec(w1, cols1)
            if fold.gf2_in_span(base, vec):
                continue
            base = np.vstack([base, vec])
            raw_seam.append(fold.aligned_fold_web(fb, gF, vmapF, g1, vmap1, w1))
            raw_kind.append(sname)
    assert raw_kind == ["in"] * 4 + ["out"] * 4, f"unexpected seam pattern {raw_kind}"
    seam_webs = (fold.greedy_weight_reduce(raw_seam[:4], det_webs, weight_reduce_passes)
                 + fold.greedy_weight_reduce(raw_seam[4:], det_webs, weight_reduce_passes))

    def seam_label_string(w, sname):
        es = w.half_edges()
        return "".join(es.get((vmapF[u], vmapF[v]), "I")
                       for u, v, s in fb.seam_edges if s == sname)

    seam_names = []
    for w, sname in zip(seam_webs, raw_kind):
        assert fold.web_is_closed(gF, w)
        seam_names.append(f"seam-{sname}:{seam_label_string(w, sname)}")
    Mdet_all = np.vstack([Mdet] + [fold.web_vec(w, colsF) for w in seam_webs])
    assert fold.gf2_rank(Mdet_all) == 8 * (K - 1) + 8

    # --- logical loops: quotient of the aligned open webs by the detectors ---
    reps = []
    base = Mdet_all.copy()
    for w1 in open_webs:
        fw = fold.aligned_fold_web(fb, gF, vmapF, g1, vmap1, w1)
        vec = fold.web_vec(fw, colsF)
        if not fold.gf2_in_span(base, vec):
            base = np.vstack([base, vec])
            reps.append(w1)
    assert len(reps) == 2, f"logical quotient dimension {len(reps)} != 2"

    log_webs, log_names = [], []
    for target, lab in (((1, 0), "loop-Xbar"), ((0, 1), "loop-Zbar")):
        pick = None
        for coeffs in ((1, 0), (0, 1), (1, 1)):
            w1 = fold.combine_webs(g1, reps, coeffs)
            if _cls_of(fold.boundary_pauli_string(g1, w1, in_b)) == target:
                pick = w1
                break
        assert pick is not None, f"no combo of the quotient reps has class {target}"
        fw = fold.aligned_fold_web(fb, gF, vmapF, g1, vmap1, pick)
        fw = fold.greedy_weight_reduce([fw], det_webs + seam_webs,
                                       weight_reduce_passes)[0]
        assert fold.web_is_closed(gF, fw)
        assert not fold.gf2_in_span(Mdet_all, fold.web_vec(fw, colsF))
        log_webs.append(fw)
        log_names.append(lab)

    Mfull = np.vstack([Mdet_all] + [fold.web_vec(w, colsF) for w in log_webs])
    expected = 8 * (K - 1) + 8 + 2
    assert fold.gf2_rank(Mfull) == expected == d_all, (
        f"closed-web space dim {d_all} != {expected} = 8(K-1)+8+2")
    assert all(fold.gf2_in_span(Mfull, r) for r in MallF)

    # --- decoding matrices ---------------------------------------------------
    wires, H_det, O = fold.decoding_matrices(gF, det_webs + seam_webs, log_webs)
    covH, covO = H_det.any(axis=0), O.any(axis=0)
    n_silent = int((~covH & covO).sum())
    assert n_silent == 0, "fold must have no silent logical-flipping component"

    return dict(structure=f"fold_k{K}", rounds=K, join=join,
                fb=fb, g=gF, vmap=vmapF, g_open=g1, vmap_open=vmap1,
                det_webs=det_webs, seam_webs=seam_webs, log_webs=log_webs,
                det_names=det_names + seam_names, log_names=log_names,
                wires=wires, H_det=H_det, O=O, d_all=int(d_all),
                n_silent=n_silent,
                wire_table=fold.wire_table(fb, gF, vmapF))


# ------------------------------------------------------------- open control
def build_open_control(rounds: int = 2, weight_reduce_passes: int = 4) -> Dict:
    """The OPEN K-round block as a decoding problem — the boundary-artifact
    control.  Detectors = its 4(K-1) closed detecting regions ONLY (that is
    all the closed webs an open block has).  Observables = one logical-flow
    pair picked from the stabiliser webs by the logical class of their
    IN-boundary Pauli (Xbar-flow: anticommutes with Z...Z; Zbar-flow:
    anticommutes with X...X), weight-reduced modulo the regions only (adding
    a region never changes what the observable measures; adding another
    stabiliser web WOULD, so the representative choice is documented, not
    optimised away)."""
    K = rounds
    block, g1, vmap1, regions, stabs = _open_block_webs(K)
    assert len(regions) == 4 * (K - 1)
    in_b, out_b = list(g1.inputs()), list(g1.outputs())

    in_strs = [fold.boundary_pauli_string(g1, w, in_b) for w in stabs]
    classes = [_cls_of(s) for s in in_strs]

    log_webs, log_names, log_docs = [], [], []
    for target, lab in (((1, 0), "flow-Xbar"), ((0, 1), "flow-Zbar")):
        best = None
        for lam in range(1, 1 << len(stabs)):        # exhaustive: 2^10 combos
            cls = (0, 0)
            for j in range(len(stabs)):
                if (lam >> j) & 1:
                    cls = (cls[0] ^ classes[j][0], cls[1] ^ classes[j][1])
            if cls != target:
                continue
            pop = bin(lam).count("1")
            if best is None or pop < best[0]:
                best = (pop, lam)
        assert best is not None, f"no stabiliser-web combo has class {target}"
        coeffs = [(best[1] >> j) & 1 for j in range(len(stabs))]
        w = fold.combine_webs(g1, stabs, coeffs)
        w = fold.greedy_weight_reduce([w], regions, weight_reduce_passes)[0]
        assert _cls_of(fold.boundary_pauli_string(g1, w, in_b)) == target
        log_webs.append(w)
        log_names.append(lab)
        log_docs.append(dict(label=lab,
                             in_pauli=fold.boundary_pauli_string(g1, w, in_b),
                             out_pauli=fold.boundary_pauli_string(g1, w, out_b),
                             n_half_edges=len(w.half_edges())))

    wires, H_det, O = fold.decoding_matrices(g1, regions, log_webs)

    # boundary-artifact bookkeeping: leg wires + silent components
    bnd = set(in_b) | set(out_b)
    leg_wire_idx = [i for i, (u, v) in enumerate(wires) if u in bnd or v in bnd]
    covH, covO = H_det.any(axis=0), O.any(axis=0)
    silent_cols = np.flatnonzero(~covH & covO)          # flip O, seen by nothing
    silent_wires = sorted({int(c) // 2 for c in silent_cols})

    return dict(structure=f"open_k{K}", rounds=K,
                g=g1, vmap=vmap1, det_webs=list(regions),
                log_webs=log_webs, log_names=log_names, log_docs=log_docs,
                wires=wires, H_det=H_det, O=O,
                leg_wire_idx=leg_wire_idx,
                n_silent=int(silent_cols.size),
                silent_cols=[int(c) for c in silent_cols],
                silent_wires=silent_wires)


# ------------------------------------------------------------------- output
def versions() -> Dict[str, str]:
    import sys

    import ldpc
    import pyzx

    return dict(python=sys.version.split()[0], numpy=np.__version__,
                pyzx=pyzx.__version__,
                ldpc=getattr(ldpc, "__version__", "2.4.1"))


def write_point_json(path, structure: str, point: Dict) -> None:
    """One JSON per (structure, p_e) point, schema per the task spec."""
    rec = dict(structure=structure, p_e=point["p_e"], shots=point["shots"],
               fails=point["fails"], ler=point["ler"], seed=point["seed"],
               decoder=DECODER_SPEC, versions=versions(),
               wall_s=point["wall_s"])
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
