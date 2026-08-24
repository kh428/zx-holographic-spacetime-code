"""Vector-first detecting regions (option-3 memory surgery).

Replicates pyzx.web._compute's region pass but keeps only the packed FIRING
VECTORS (small) plus enough state to regenerate any single region's PauliWeb
dict on demand — at most ONE dict alive at a time. Also precomputes each
region's web-vector over the open graph's half-edge columns, so consumers
that only need vectors never materialize a dict at all.
"""
from __future__ import annotations

import numpy as np

from pyzx.linalg import Mat2
from .gf2compat import packed_rref_with_pivots, packed_mul
from pyzx.web.red_green import to_red_green_form
from pyzx.web.firing_assignments import (
    determine_ordering, create_firing_verification,
    convert_firing_assignment_to_web_prototype)


class LazyRegions:
    """Sequence of detecting-region PauliWebs, regenerated one at a time."""

    def __init__(self, graph, fold_mod, cols1):
        g = graph.clone()
        self._orig = graph
        self._extra = to_red_green_form(g)
        self._g = g
        self._ord = determine_ordering(g)
        # fully-packed firing pass: build the firing matrix DIRECTLY in
        # numpy uint8 (create_firing_verification's Mat2.zeros is python
        # list-of-lists: ~8 B/entry -> ~15 GB at e45 n=2 r16 — the original
        # OOM killer). Byte-identical semantics to firing_assignments.py.
        from pyzx.utils import VertexType as _VT
        o = self._ord
        nzb = len(o.z_boundaries)
        n_nb = nzb + len(o.internal_spiders)
        npi2 = len(o.pi_2_spiders)
        rows_, cols_ = n_nb, n_nb + nzb
        A = np.zeros((rows_, cols_), dtype=np.uint8)
        if nzb:
            A[:nzb, :nzb] = np.eye(nzb, dtype=np.uint8)
        for e in g.edges():
            s_, t_ = g.edge_st(e)
            if (g.type(s_) != _VT.BOUNDARY and g.type(t_) != _VT.BOUNDARY):
                A[o.ord(s_), nzb + o.ord(t_)] ^= 1
                A[o.ord(t_), nzb + o.ord(s_)] ^= 1
        for i in range(npi2):
            A[rows_ - npi2 + i, cols_ - npi2 + i] ^= 1
        from .gf2compat import _pack, _rref_inplace, _unpack
        W_, rows, cols = _pack(A)
        del A                                    # dense copy not needed past here
        piv_cols = _rref_inplace(W_, rows, cols, full_reduce=True)
        R = np.array(_unpack(W_[:len(piv_cols)], cols), dtype=np.uint8)
        free = [c for c in range(cols) if c not in set(piv_cols)]
        # nullspace basis: one row per free var
        sol = np.zeros((len(free), cols), dtype=np.uint8)
        for k, fc in enumerate(free):
            sol[k, fc] = 1
            for r_i, pc in enumerate(piv_cols):
                sol[k, pc] = R[r_i, fc]
        del W_, R
        limit = min(len(self._ord.z_boundaries) * 2, sol.shape[1])
        bsel = sol[:, :limit].T.copy() if sol.size else             np.zeros((limit, 0), dtype=np.uint8)
        _, piv = packed_rref_with_pivots(bsel.tolist() if bsel.size else [])
        self.stab_sols = [sol[i].tolist() for i in piv]
        bnull = Mat2(bsel.tolist()).nullspace() if bsel.size else []
        self.region_sols = (packed_mul(bnull, sol.tolist())
                            if len(bnull) else [])
        del sol
        # precompute packed web-vectors over cols1 (one dict alive at a time)
        vecs = []
        for v in self.region_sols:
            w = self._make(v)
            vecs.append(fold_mod.web_vec(w, cols1))
            del w
        self.vecs = (np.array(vecs, dtype=np.uint8) if vecs
                     else np.zeros((0, 2 * len(cols1)), dtype=np.uint8))

    def _make(self, v):
        w = convert_firing_assignment_to_web_prototype(self._g, self._ord, v)
        self._extra.remove_from(self._g, w)
        w.g = self._orig
        return w

    def stab_webs(self):
        out = []
        for v in self.stab_sols:
            out.append(self._make(v))
        return out

    def __len__(self):
        return len(self.region_sols)

    def __iter__(self):
        for v in self.region_sols:
            yield self._make(v)

    def __getitem__(self, i):
        return self._make(self.region_sols[i])
