"""Build the fused ("tower") spacetime diagrams of the {4,5} networks and
their decoding matrices, from scratch.

The full network (all bulk legs contracted with encoder/un-encoder pairs) is
fused into a single spacetime ZX-diagram; the seam code and tower detector /
logical matrices are extracted with packed GF(2) linear algebra. Requires
LEGO_HQEC (https://github.com/QML-Group/HQEC) and hypertiling.

Usage: python scripts/build_old_towers.py [output_dir]
"""
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pyzx as zx
from pyzx.utils import VertexType, EdgeType
from pyzx import fuse as _fuse

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path(__file__).resolve().parent.parent / 'data' / 'matrices'

# ---------------- packed GF2 ----------------
def packed_rref(M):
    R = np.packbits(np.ascontiguousarray(M % 2), axis=1)
    n = M.shape[1]; piv = []; r = 0
    for c in range(n):
        byte, bit = divmod(c, 8); mask = 1 << (7 - bit)
        idx = np.flatnonzero((R[r:, byte] & mask) != 0)
        if idx.size == 0: continue
        p = idx[0] + r
        if p != r: R[[r, p]] = R[[p, r]]
        hit = np.flatnonzero((R[:, byte] & mask) != 0); hit = hit[hit != r]
        R[hit] ^= R[r]
        piv.append(c); r += 1
        if r == R.shape[0]: break
    return R, piv, r


def packed_nullspace(M):
    R, piv, r = packed_rref(M)
    n = M.shape[1]; pset = set(piv)
    free = [c for c in range(n) if c not in pset]
    basis = np.zeros((len(free), n), np.uint8)
    for bi, f in enumerate(free):
        basis[bi, f] = 1
        byte, bit = divmod(f, 8); mask = 1 << (7 - bit)
        for ri in np.flatnonzero((R[:r, byte] & mask) != 0):
            basis[bi, piv[ri]] = 1
    return basis


def unpacked_rref(M):
    R, piv, r = packed_rref(M)
    return np.unpackbits(R[:r], axis=1)[:, :M.shape[1]], piv


def reduce_mod(V, Srref_piv):
    Srref, piv = Srref_piv
    V = V.copy() % 2
    for ri, c in enumerate(piv):
        hit = np.flatnonzero(V[:, c])
        V[hit] ^= Srref[ri]
    return V


# ---------------- build (main process only) ----------------
def build_old_matrices():
    import types

    def _unspider(g, m):
        v, nbrs = m[0], list(m[1])
        w = g.add_vertex(ty=g.type(v), qubit=g.qubit(v), row=g.row(v))
        for nb in nbrs:
            et = g.edge_type(g.edge(v, nb)); g.remove_edge(g.edge(v, nb)); g.add_edge((w, nb), et)
        g.add_edge((v, w), EdgeType.SIMPLE)
        return w
    zx.rules = types.SimpleNamespace(unspider=_unspider)
    zx.draw = lambda *a, **k: None
    from spacetime.gen_tiled_45 import gen_tiled_codes
    from pyzx import fuse as _fuse
    from spacetime.core import wt02_pipeline as W

    def build_A(g_):
        edges = sorted({tuple(sorted(e)) for e in g_.edge_set()})
        eidx = {e: i for i, e in enumerate(edges)}
        E_ = len(edges)

        def at_side(e, v):
            ref = e[0]
            ha = g_.edge_type(g_.edge(*e)) == EdgeType.HADAMARD
            k = eidx[e]
            return (2 * k, 2 * k + 1) if (v == ref or not ha) else (2 * k + 1, 2 * k)
        rows = []
        for v in g_.vertices():
            if g_.type(v) != VertexType.Z: continue
            sides = [at_side(tuple(sorted((v, u))), v) for u in g_.neighbors(v)]
            for (xa, _), (xb, _) in zip(sides, sides[1:]):
                r = np.zeros(2 * E_, np.uint8); r[xa] ^= 1; r[xb] ^= 1; rows.append(r)
            r = np.zeros(2 * E_, np.uint8)
            for _, zc in sides: r[zc] ^= 1
            rows.append(r)
        return np.array(rows, np.uint8), at_side, E_

    def stub_cols(g_, blist, at_):
        cols = []
        for b in blist:
            e = tuple(sorted((b, next(iter(g_.neighbors(b))))))
            cols += list(at_(e, b))
        return cols

    def ang(g_, vl):
        return sorted(vl, key=lambda v: math.atan2(g_.qubit(v), g_.row(v)))

    def seam_code(gnet):
        A_, at_, E_ = build_A(gnet)
        K_ = packed_nullspace(A_)
        cB = stub_cols(gnet, list(gnet.inputs()), at_)
        cR = stub_cols(gnet, ang(gnet, list(gnet.outputs())), at_)
        C = packed_nullspace(np.ascontiguousarray(K_[:, cB].T))
        Wt = (C @ K_) % 2 if C.size else np.zeros((0, K_.shape[1]), np.uint8)
        S_bits = (Wt[:, cR] % 2).astype(np.uint8)
        print(f'  seam code: [[{len(list(gnet.outputs()))},{len(list(gnet.inputs()))}]] '
              f'-> {S_bits.shape[0]} stabilisers', flush=True)
        return S_bits

    def build_tower(gn):
        g0 = gn.copy(); g1 = gn.copy()
        g1.set_inputs([]); g1.set_outputs([])
        g1.set_inputs(g0.outputs()); g1.set_outputs(g0.inputs())
        g1.set_outputs(ang(g1, g1.outputs())); g0.set_inputs(ang(g0, g0.inputs()))
        gWu = g1 + g0
        gA, gB = gWu.copy(), gWu.copy()
        gA.set_outputs(ang(gA, gA.outputs())); gB.set_inputs(ang(gB, gB.inputs()))
        gTt = gA + gB
        vs = set(gTt.vertices())
        for ee in list(gTt.edge_set()):
            a, b = ee
            if a not in vs or b not in vs: continue
            if not gTt.connected(a, b): continue
            if gTt.edge_type(gTt.edge(a, b)) != EdgeType.SIMPLE: continue
            if VertexType.BOUNDARY in (gTt.type(a), gTt.type(b)): continue
            _fuse(gTt, a, b)
            for v_ in (a, b):
                if v_ not in gTt.graph:
                    vs.discard(v_)
        return gTt

    def tower_HO(gT_, S_bits):
        A_, at_, E_ = build_A(gT_)
        K_ = packed_nullspace(A_)
        cb = stub_cols(gT_, ang(gT_, list(gT_.inputs())), at_)
        ct = stub_cols(gT_, ang(gT_, list(gT_.outputs())), at_)
        Sr = unpacked_rref(S_bits)
        Rb = reduce_mod(K_[:, cb], Sr); Rt = reduce_mod(K_[:, ct], Sr)
        Cc = packed_nullspace(np.ascontiguousarray(np.hstack([Rb, Rt]).T))
        Hw = (Cc @ K_) % 2 if Cc.size else np.zeros((0, K_.shape[1]), np.uint8)
        span = W.Gf2Span(); rH = sum(span.add(r.copy().astype(np.uint8)) for r in Hw)
        Ow = [v for v in K_ if span.add(v.copy().astype(np.uint8))]
        Ow = np.array(Ow, np.uint8)
        print(f'  tower: {E_} edges | web dim {K_.shape[0]} | H {Hw.shape[0]} (rank {rH}) '
              f'| O {Ow.shape[0]} logical classes', flush=True)

        def to_rows(Wb):
            R = np.zeros((Wb.shape[0], 2 * E_), np.uint8)
            R[:, 0::2] = Wb[:, 1::2]; R[:, 1::2] = Wb[:, 0::2]
            return R
        return to_rows(Hw), to_rows(Ow), E_

    for n_, r_ in ((1, 4), (2, 5)):
        out = OUT / f'oldcon_n{n_}_mats.npz'
        if out.exists():
            print(f'old n={n_}: cached ({out.name})', flush=True); continue
        t0 = time.perf_counter()
        print(f'--- building old construction n={n_} ---', flush=True)
        gn, _ = gen_tiled_codes(4, 5, r_)
        gn = gn.copy()
        print(f'  network: {gn.num_vertices()} v | {len(gn.inputs())} bulk | '
              f'{len(gn.outputs())} rim', flush=True)
        S_bits = seam_code(gn)
        gTt = build_tower(gn)
        H_, O_, E_ = tower_HO(gTt, S_bits)
        np.savez_compressed(out, H=H_, O=O_, n_wires=E_)
        print(f'  saved {out.name}  [{time.perf_counter()-t0:.1f}s]', flush=True)




if __name__ == '__main__':
    build_old_matrices()
