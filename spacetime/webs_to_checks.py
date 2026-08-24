"""Stabiliser-basis extraction from an encoder ZX-diagram via Pauli webs.

Uses the Pauli-web machinery of pyzx (``pyzx.web``) to compute the stabilising
webs of an encoder V, then recovers the trivial-bulk subgroup -- the code
stabilisers -- by GF(2) linear algebra.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from pyzx.pauliweb import PauliWeb
from pyzx.utils import EdgeType, VertexType
from pyzx.web import compute_stabilisers as web_compute_stabilisers


def gf2_rref(M: np.ndarray) -> np.ndarray:
    """Reduced row echelon form over GF(2); zero rows dropped.  The RREF is a
    canonical form of the row space, so two matrices have equal row spaces
    iff their gf2_rref's are identical arrays."""
    A = (np.array(M, dtype=np.uint8, copy=True) % 2)
    if A.size == 0:
        return A.reshape(0, M.shape[1] if M.ndim == 2 else 0)
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        hits = np.nonzero(A[:, c])[0]
        for i in hits:
            if i != r:
                A[i, :] ^= A[r, :]
        r += 1
        if r == rows:
            break
    return A[np.any(A, axis=1)]



def gf2_nullspace(M: np.ndarray) -> np.ndarray:
    """Basis (rows) of {v : M v = 0 mod 2}."""
    M = np.array(M, dtype=np.uint8) % 2
    rows, cols = M.shape
    R = gf2_rref(M)
    pivots = [int(np.nonzero(r)[0][0]) for r in R]
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for i, p in enumerate(pivots):
            if R[i, f]:
                v[p] = 1
        basis.append(v)
    return (np.array(basis, dtype=np.uint8) if basis
            else np.zeros((0, cols), dtype=np.uint8))



def wire_label(g, web: PauliWeb, edges: Sequence[tuple]) -> str:
    """The single Pauli label a web puts on a registered wire.

    A wire is a simple path of SIMPLE edges through degree-2 phase-0 Z
    identity spiders, so a web that is closed along it carries ONE label on
    every half-edge of every segment.  This function aggregates the segments
    (interface identity spiders contribute two graphical edges but ONE wire)
    and asserts the consistency the theory promises:
      * every segment is a SIMPLE edge (a Hadamard segment would make "the"
        label basis-dependent along the wire — none occur in these stacks);
      * the two half-edge labels of each segment agree;
      * all segments of the wire agree.
    Returns 'I', 'X', 'Y' or 'Z'.
    """
    labels = set()
    for e in edges:
        a, b = g.edge_st(e)
        assert g.edge_type(e) == EdgeType.SIMPLE, \
            f"wire segment {e} is not a SIMPLE edge; label would be ambiguous"
        l_ab, l_ba = web[(a, b)], web[(b, a)]
        assert l_ab == l_ba, f"half-edge labels differ on {e}: {l_ab} vs {l_ba}"
        labels.add(l_ab)
    assert len(labels) == 1, \
        f"web label not constant along wire (segments {list(edges)}: {labels})"
    return labels.pop()



def _pauli_to_xz(p: str) -> np.ndarray:
    """Pauli string -> GF(2) symplectic vector (x-part | z-part)."""
    x = [1 if c in "XY" else 0 for c in p]
    z = [1 if c in "ZY" else 0 for c in p]
    return np.array(x + z, dtype=np.uint8)



def _xz_to_pauli(v: np.ndarray) -> str:
    n = len(v) // 2
    return "".join("IXZY"[int(v[i]) + 2 * int(v[n + i])] for i in range(n))



def code_stabiliser_basis(V) -> Tuple[np.ndarray, List[str]]:
    """Stabiliser generators of the code V encodes, extracted from V itself.

    Stabilising webs of the ENCODER are pairs (P_bulk, P_boundary) with
    P_boundary V P_bulk^T = +-V; those with trivial bulk part are exactly the
    code stabilisers (V injective).  pyzx returns *a* GF(2) basis whose
    elements generally all carry nontrivial bulk support, so the trivial-bulk
    subgroup is recovered linearly: with B / S the bulk / boundary symplectic
    parts of the basis webs, the rows of ``ker(B^T) . S`` span the stabiliser
    group.  Independent of any hand-written generator list; signs are not
    tracked (GF(2)).  Returns (RREF symplectic basis, its Pauli strings)."""
    outs, ins = list(V.outputs()), list(V.inputs())

    def _leg_label(w, b):
        return wire_label(V, w, (V.edge(b, next(iter(V.neighbors(b)))),))

    stabs = web_compute_stabilisers(V)
    B = np.array([_pauli_to_xz("".join(_leg_label(w, b) for b in ins))
                  for w in stabs], dtype=np.uint8)
    S = np.array([_pauli_to_xz("".join(_leg_label(w, b) for b in outs))
                  for w in stabs], dtype=np.uint8)
    K = gf2_nullspace(B.T)                  # combinations with trivial bulk part
    basis = gf2_rref((K @ S) % 2) if K.size else np.zeros((0, 2 * len(outs)), np.uint8)
    return basis, [_xz_to_pauli(v) for v in basis]


