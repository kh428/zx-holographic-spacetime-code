"""Stabiliser + logical extraction from a ZX holographic code diagram.

Uses the Pauli-webs branch of pyzx (`pyzx.webs.compute_stabilisers`) to
enumerate Pauli webs of the ZX diagram, then converts to a binary
symplectic parity-check matrix using the paper's interleaved layout
`(x_0, z_0, x_1, z_1, ...)`.

Top-level API:

    extract_code(g, gauge='plus', keep_bulk_idx=0)
        → dict(S, L, H_raw, bulk_vs, boundary_vs, vmap, stab_webs,
               logical_webs)

Where:
  S             (m, 2*N_bdry)   boundary stabiliser rows (interleaved sympl)
  L             (2, 2*N_bdry)   logical X̄, Z̄ of the kept bulk qubit
  H_raw         (n_webs, 2*N)   raw parity check over all vertices (bulk+bdry)
  bulk_vs       list of ZX graph input vertex ids (bulk/logical legs)
  boundary_vs   list of ZX graph output vertex ids (boundary/physical legs)
  vmap          vertex → index map into the 2*N columns of H_raw
  stab_webs     list of PauliWeb objects for each row of S
  logical_webs  list of PauliWeb objects for each row of L
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional
import numpy as np


# ---------- GF(2) linear algebra ------------------------------------------

def gf2_rref_with_ops(A):
    A = A.copy()
    m, n = A.shape
    ops = []
    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, m):
            if A[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
            ops.append(("swap", r, pivot))
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
                ops.append(("xor", i, r))
        r += 1
        if r == m:
            break
    return A, ops


def gf2_rref(A):
    A = A.copy()
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, m):
            if A[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return A, pivots


def symplectic_form(n_qubits):
    Λ = np.zeros((2 * n_qubits, 2 * n_qubits), dtype=int)
    for i in range(n_qubits):
        Λ[2 * i, 2 * i + 1] = 1
        Λ[2 * i + 1, 2 * i] = 1
    return Λ


def symplectic_nullspace(H, n_qubits):
    Λ = symplectic_form(n_qubits)
    M = (H @ Λ) % 2
    M_rref, pivots = gf2_rref(M)
    n = M.shape[1]
    free_cols = [j for j in range(n) if j not in pivots]
    basis = []
    for f in free_cols:
        v = np.zeros(n, dtype=int)
        v[f] = 1
        for i, p in enumerate(pivots):
            if M_rref[i, f]:
                v[p] = 1
        basis.append(v)
    return np.array(basis) if basis else np.zeros((0, n), dtype=int)


# ---------- Pauli web algebra ---------------------------------------------

def pauli_multiply(p1, p2):
    if p1 == 'I':
        return p2
    if p2 == 'I':
        return p1
    if p1 == p2:
        return 'I'
    pair = {p1, p2}
    if pair == {'X', 'Z'}:
        return 'Y'
    if pair == {'X', 'Y'}:
        return 'Z'
    if pair == {'Z', 'Y'}:
        return 'X'
    return 'I'


def combine_pauli_webs(web1, web2):
    combined = deepcopy(web1)
    for edge, p2 in web2.es.items():
        if edge in combined.es:
            p1 = combined.es[edge]
            new_p = pauli_multiply(p1, p2)
            if new_p == 'I':
                del combined.es[edge]
            else:
                combined.es[edge] = new_p
        else:
            combined.es[edge] = p2
    return combined


# ---------- extraction ----------------------------------------------------

def build_parity_check_from_webs(g, webs):
    """Given a ZX graph and a list of Pauli webs, build the binary symplectic
    parity-check matrix over ALL input/output vertices (bulk + boundary)."""
    bulk_vertices = list(g.inputs())
    boundary_vertices = list(g.outputs())
    vertex_order = bulk_vertices + boundary_vertices
    vertex_to_idx = {v: i for i, v in enumerate(vertex_order)}
    n_vertices = len(vertex_order)
    n_cols = 2 * n_vertices
    H = np.zeros((len(webs), n_cols), dtype=int)

    def es_to_row(es):
        row = np.zeros(n_cols, dtype=int)
        for (v1, v2), p in es.items():
            for v in (v1, v2):
                if v in vertex_to_idx:
                    i = vertex_to_idx[v]
                    if p in ('X', 'Y'):
                        row[2 * i] = 1
                    if p in ('Z', 'Y'):
                        row[2 * i + 1] = 1
        return row

    for r, web in enumerate(webs):
        H[r] = es_to_row(web.es)
    return H, bulk_vertices, boundary_vertices, vertex_to_idx


def extract_boundary_stabilisers_and_logicals(H, bulk_vertices, n_vertices,
                                              webs, vertex_to_idx):
    """Given the full-graph parity matrix H, split into boundary-supported
    stabilisers S and bulk-coupled logical operators L. Uses RREF on the
    bulk-column subblock and tracks the same row-ops on the webs list."""
    bulk_cols = ([2 * vertex_to_idx[v] for v in bulk_vertices]
                 + [2 * vertex_to_idx[v] + 1 for v in bulk_vertices])
    boundary_cols = [i for i in range(2 * n_vertices) if i not in bulk_cols]

    H_bulk = H[:, bulk_cols].copy()
    _, ops = gf2_rref_with_ops(H_bulk)

    H_full = H.copy()
    for op in ops:
        if op[0] == "swap":
            _, i, j = op
            H_full[[i, j]] = H_full[[j, i]]
        else:  # xor
            _, i, j = op
            H_full[i] ^= H_full[j]

    web_rows = [deepcopy(web) for web in webs]
    for op in ops:
        if op[0] == "swap":
            _, i, j = op
            web_rows[i], web_rows[j] = web_rows[j], web_rows[i]
        else:
            _, i, j = op
            web_rows[i] = combine_pauli_webs(web_rows[i], web_rows[j])

    # rows with no bulk support → boundary stabilisers
    boundary_stab_mask = np.all(H_full[:, bulk_cols] == 0, axis=1)
    S_boundary = H_full[boundary_stab_mask][:, boundary_cols]
    stab_webs = [web_rows[i] for i, keep in enumerate(boundary_stab_mask) if keep]

    # rows with bulk support → logical operators (projected to boundary cols)
    bulk_stab_mask = np.any(H_full[:, bulk_cols] == 1, axis=1)
    L_boundary = H_full[bulk_stab_mask][:, boundary_cols]
    logical_webs = [web_rows[i] for i, keep in enumerate(bulk_stab_mask) if keep]

    return S_boundary, stab_webs, L_boundary, logical_webs


def extract_code(g, gauge: Optional[str] = "plus", keep_bulk_idx: int = 0):
    """End-to-end pipeline: take a ZX graph built by `build_zx_holo`,
    (optionally) apply a gauge projection on all bulk legs except the one
    at `keep_bulk_idx`, enumerate Pauli webs, and return (S, L, …).

    Parameters
    ----------
    g : pyzx.Graph
    gauge : None | "plus" | "zero"
        If None: do not project any bulk legs (no-gauge code, all bulk
        qubits are logicals).
        "plus":  apply |+⟩ to every bulk leg except `keep_bulk_idx`.
        "zero":  apply |0⟩ to every bulk leg except `keep_bulk_idx`.
    keep_bulk_idx : int
        Which bulk input to leave open as a logical qubit.

    Returns a dict:
        S, L   : (m, 2*N_bdry), (k, 2*N_bdry) symplectic binary matrices
        H_raw  : full parity check over bulk+boundary
        bulk_vs, boundary_vs, vmap, stab_webs, logical_webs
    """
    import pyzx as zx
    from pyzx.web import compute_stabilisers

    if gauge not in (None, "plus", "zero"):
        raise ValueError(f"unknown gauge: {gauge}")

    if gauge is not None:
        n_inputs = len(list(g.inputs()))
        if keep_bulk_idx < 0 or keep_bulk_idx >= n_inputs:
            raise ValueError(f"keep_bulk_idx={keep_bulk_idx} out of range")
        projections = ["+" if gauge == "plus" else "0"] * n_inputs
        projections[keep_bulk_idx] = "/"  # "/" = leave this leg open in pyzx
        # pyzx's apply_state takes a single string; "/" marks identity
        # (no projection). Paper uses: "/" + "+"*(n-1) to leave input 0 open.
        state_str = "".join(projections)
        g = g.copy()
        g.apply_state(state_str)

    webs = compute_stabilisers(g)
    H, bulk_vs, boundary_vs, vmap = build_parity_check_from_webs(g, webs)
    S, stab_webs, L, logical_webs = (
        extract_boundary_stabilisers_and_logicals(
            H, bulk_vs, len(vmap), webs, vmap
        )
    )
    return dict(S=S.astype(np.uint8),
                L=L.astype(np.uint8),
                H_raw=H.astype(np.uint8),
                bulk_vs=bulk_vs,
                boundary_vs=boundary_vs,
                vmap=vmap,
                stab_webs=stab_webs,
                logical_webs=logical_webs)
