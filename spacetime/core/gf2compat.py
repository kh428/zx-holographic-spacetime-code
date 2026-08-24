"""Pure-numpy replacements for the packed GF(2) helpers that some
pipeline modules import. Row vectors are packed bit-wise (little-endian)
into uint8 words; all arithmetic is over GF(2)."""
import numpy as np


def _pack(A):
    A = np.ascontiguousarray(np.asarray(A, dtype=np.uint8) % 2)
    rows, cols = A.shape
    return np.packbits(A, axis=1, bitorder="little"), rows, cols


def _rref_inplace(Wp, rows, cols, full_reduce=True):
    piv = []
    r = 0
    for c in range(cols):
        byte, bit = divmod(c, 8)
        mask = np.uint8(1 << bit)
        idx = np.flatnonzero((Wp[r:rows, byte] & mask) != 0)
        if idx.size == 0:
            continue
        p = int(idx[0]) + r
        if p != r:
            Wp[[r, p]] = Wp[[p, r]]
        hit = np.flatnonzero((Wp[:rows, byte] & mask) != 0)
        hit = hit[hit != r]
        Wp[hit] ^= Wp[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return piv


def _unpack(Wp_rows, cols):
    return [np.unpackbits(np.asarray(row, dtype=np.uint8),
                          bitorder="little")[:cols] for row in Wp_rows]


def packed_rref_with_pivots(rows):
    """rows: sequence of 0/1 sequences. Returns (rref rows, pivot columns)."""
    if not len(rows):
        return [], []
    A = np.array(rows, dtype=np.uint8) % 2
    Wp, nr, nc = _pack(A)
    piv = _rref_inplace(Wp, nr, nc, full_reduce=True)
    return _unpack(Wp[:len(piv)], nc), piv


def packed_mul(A, B):
    """GF(2) matrix product of two 0/1 row-lists."""
    A = np.array(A, dtype=np.uint8) % 2
    B = np.array(B, dtype=np.uint8) % 2
    return list((A @ B) % 2)
