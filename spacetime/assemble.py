"""Round-stacking assembler.

The spacetime blocks are periodic under translation by one round, so the
decoding data (wire set, detector matrix, logical correlator) for any even
depth K can be assembled from the labelled K=2 export. The detector matrix is
block-banded (two boundary bands and a repeating interior band); the logical
representative is first made periodic by Gaussian elimination modulo the
detectors, then chained. Validation helpers check the assembly against a
directly computed reference by row-span equality.
"""
import ast
import json
import pathlib

import numpy as np

from .core.wt02_pipeline import Gf2Span


def cap_rule(lab):
    """Bottom/top-cap classification of a wire from its end labels."""
    cis = {ci for ci, _ in lab}
    ts = [t for _, (_, _, t) in lab]
    if len(cis) == 2:
        return 'bot' if max(ts) == 0 else 'top'
    return 'int'

def load_labelled(data_dir, n, K):
    z = np.load(pathlib.Path(data_dir) / f'labelled_n{n}_K{K}.npz')
    labs = json.load(open(pathlib.Path(data_dir) / f'labelled_n{n}_K{K}.json'))
    ends = [frozenset(((int(a[0]), ast.literal_eval(a[1])),
                       (int(b[0]), ast.literal_eval(b[1])))) for a, b in labs]
    return z['H'].astype(np.uint8), z['O'].astype(np.uint8), ends


def shift_lab(lab, s):
    return frozenset((ci, (kind, a, t + 2 * s)) for ci, (kind, a, t) in lab)


def classify_wires(W2, W4):
    """valid-shift class per K=2 wire, from the K=4 ground truth:
    'int' translatable at every shift; 'bot' only s=0; 'top' only s=K-2."""
    set4 = set(W4)
    cls = []
    for lab in W2:
        v = [s for s in (0, 1, 2) if shift_lab(lab, s) in set4]
        if v == [0, 1, 2]: cls.append('int')
        elif v == [0]: cls.append('bot')
        elif v == [2]: cls.append('top')
        else: raise AssertionError(f'unexpected shift set {v} for {lab}')
    return cls


def target_wires(W2, cls, K):
    """wire list for depth K (K even): bot at s=0, int at all s, top at s=K-2."""
    assign = {}
    for w, lab in enumerate(W2):
        shifts = ([0] if cls[w] == 'bot' else
                  [K - 2] if cls[w] == 'top' else list(range(K - 1)))
        for s in shifts:
            assign.setdefault(shift_lab(lab, s), None)
    wires = sorted(assign, key=lambda lab: sorted(
        (ci, kind, a, t) for ci, (kind, a, t) in lab))
    return wires, {lab: i for i, lab in enumerate(wires)}


def translate_rows(rows, W2, idx_t, ncol, shifts):
    out = []
    for s in shifts:
        for r in rows:
            v = np.zeros(ncol, np.uint8); ok = True
            for w in np.flatnonzero(r[0::2] | r[1::2]):
                j = idx_t.get(shift_lab(W2[w], s))
                if j is None: ok = False; break
                v[2 * j] = r[2 * w]; v[2 * j + 1] = r[2 * w + 1]
            if ok: out.append(v)
    return out


def periodic_O(H2, O2, W2, cls):
    """O2' = O2 + c.H2 with round-1 pattern == shift of round-0 pattern.
    Constraint pairs: (w, tau(w)) for interior wires w whose +1-shift label is
    itself a K=2 wire."""
    idx2 = {lab: i for i, lab in enumerate(W2)}
    pairs = []
    for w, lab in enumerate(W2):
        if cls[w] != 'int': continue
        j = idx2.get(shift_lab(lab, 1))
        if j is not None: pairs.append((w, j))
    O2p = []
    for oi in range(O2.shape[0]):
        cols, rhs = [], []
        for (w, j) in pairs:
            for off in (0, 1):
                cols.append(H2[:, 2 * w + off] ^ H2[:, 2 * j + off])
                rhs.append(O2[oi, 2 * w + off] ^ O2[oi, 2 * j + off])
        M = np.array(cols, np.uint8)          # (constraints x 204)
        b = np.array(rhs, np.uint8)
        # solve M c = b by elimination on [M|b]
        A = np.hstack([M, b[:, None]])
        r = 0; piv = []
        for c_ in range(M.shape[1]):
            rowsnz = [i for i in range(r, A.shape[0]) if A[i, c_]]
            if not rowsnz: continue
            A[[r, rowsnz[0]]] = A[[rowsnz[0], r]]
            for i in range(A.shape[0]):
                if i != r and A[i, c_]: A[i] ^= A[r]
            piv.append(c_); r += 1
        assert not any(A[i, -1] and not A[i, :-1].any()
                       for i in range(A.shape[0])), 'periodic rep: no solution'
        c = np.zeros(M.shape[1], np.uint8)
        for i, p_ in enumerate(piv): c[p_] = A[i, -1]
        v = O2[oi].copy()
        for k_ in np.flatnonzero(c): v ^= H2[k_]
        # verify periodicity
        for (w, j) in pairs:
            assert v[2*w] == v[2*j] and v[2*w+1] == v[2*j+1]
        O2p.append(v)
    return np.array(O2p, np.uint8)


def chain_O(O2p, W2, cls, idx_t, ncol, K):
    """periodic chain: bot caps from s=0, top caps from s=K-2, interior wires
    take the (shift-independent) O2' value."""
    O = np.zeros((O2p.shape[0], ncol), np.uint8)
    for oi in range(O2p.shape[0]):
        for w, lab in enumerate(W2):
            shifts = ([0] if cls[w] == 'bot' else
                      [K - 2] if cls[w] == 'top' else list(range(K - 1)))
            for s in shifts:
                j = idx_t[shift_lab(lab, s)]
                for off in (0, 1):
                    bit = O2p[oi, 2 * w + off]
                    if O[oi, 2 * j + off] and not bit:
                        raise AssertionError('chain conflict')
                    O[oi, 2 * j + off] |= bit
    return O


def assemble(data_dir, n, K, cls=None):
    H2, O2, W2_ = load_labelled(data_dir, n, 2)
    if cls is None:
        cls = [cap_rule(lab) for lab in W2_]
    wires, idx_t = target_wires(W2_, cls, K)
    ncol = 2 * len(wires)
    rows = translate_rows(H2, W2_, idx_t, ncol, range(K - 1))
    O2p = periodic_O(H2, O2, W2_, cls)
    O = chain_O(O2p, W2_, cls, idx_t, ncol, K)
    H = np.array(rows, np.uint8)
    return H, O, wires, cls


