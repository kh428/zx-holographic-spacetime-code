"""Rim-adapted check basis for the bond-extended $\\{4,5\\}$ codes.

Re-chooses the generating set of the stabiliser group so that the non-bond
generators are low-weight and supported on the boundary (the group itself is
unchanged). Exact even-Y enumeration at n=1; weight-thinning with even-Y
repair at n>=2.
"""
import numpy as np

from . import e45_bondext as BE
from .core import wt02_pipeline as W

def gf2_nullspace(A):
    A = A.copy() % 2; m, ncol = A.shape; piv = []; rr = 0
    for c_ in range(ncol):
        s_ = np.nonzero(A[rr:, c_])[0]
        if not len(s_): continue
        A[[rr, rr + s_[0]]] = A[[rr + s_[0], rr]]
        h = np.nonzero(A[:, c_])[0]; h = h[h != rr]; A[h] ^= A[rr]
        piv.append(c_); rr += 1
        if rr == m: break
    B = []
    for f in [c_ for c_ in range(ncol) if c_ not in piv]:
        v = np.zeros(ncol, np.uint8); v[f] = 1
        for i_, p_ in enumerate(piv): v[p_] = A[i_, f]
        B.append(v)
    return np.array(B, np.uint8)


def rim_basis_code(n):
    code = BE.code_for_foliation(n)
    c = BE.build(n)
    S, nq = c["S"], c["nq"]
    P = c["layoutP"]
    rim = sorted(4 * i + s for (i, s), j in P["nbr"].items() if j < 0)
    r = len(rim)
    A = np.zeros((len(S), 2 * r), np.uint8)
    for i, row in enumerate(S):
        A[i, :r] = row[nq:][rim]; A[i, r:] = row[:nq][rim]
    N = gf2_nullspace(A)
    reps = c["reps"]
    C2 = np.zeros((2, N.shape[0]), np.uint8)
    for j, v in enumerate(N):
        x = np.zeros(nq, np.uint8); z = np.zeros(nq, np.uint8)
        x[rim] = v[:r]; z[rim] = v[r:]
        C2[0, j] = (int(x @ reps[0][2]) + int(z @ reps[0][1])) % 2
        C2[1, j] = (int(x @ reps[1][2]) + int(z @ reps[1][1])) % 2
    stab = (gf2_nullspace(C2) @ N) % 2
    print(f'n={n}: rim legs={r}, rim-stab dim={stab.shape[0]}', flush=True)

    def qweight(v): return int(np.count_nonzero(v[:r] | v[r:]))
    def ycount(v): return int(np.count_nonzero(v[:r] & v[r:]))

    if n == 1:
        # exact even-Y enumeration over the 2^19 group (d5_build v2)
        bi = [int.from_bytes(np.packbits(v).tobytes(), 'big') for v in stab]
        pool = []
        vals = np.zeros(1 << 19, object)
        for mask in range(1, 1 << 19):
            v = vals[mask & (mask - 1)] ^ bi[(mask & -mask).bit_length() - 1]
            vals[mask] = v
            if bin(v).count('1') <= 12:
                b = np.unpackbits(np.frombuffer(
                    v.to_bytes((2 * r + 7) // 8, 'big'), np.uint8))[:2 * r]
                wq = int(np.count_nonzero(b[:r] | b[r:]))
                yc = int(np.count_nonzero(b[:r] & b[r:]))
                if wq <= 6 and yc % 2 == 0:
                    pool.append((wq, b))
        del vals
        pool.sort(key=lambda t: t[0])
        span = W.Gf2Span(); rim_gens = []
        for wq, b in pool:
            if span.add(b.copy()): rim_gens.append(b)
            if len(rim_gens) == 19: break
        for v in stab:
            if len(rim_gens) == 19: break
            if ycount(v) % 2 == 0 and span.add(v.copy()): rim_gens.append(v)
        assert len(rim_gens) == 19
        final = []
        for v in rim_gens:
            fv = np.zeros(2 * nq, np.uint8)
            fv[:nq][rim] = v[:r]; fv[nq:][rim] = v[r:]
            final.append(fv)
    else:
        # weight-thinning + even-Y repair (d5n2)
        rows = [v.copy() for v in stab]
        changed = True; passes = 0
        while changed and passes < 40:
            changed = False; passes += 1
            for i in range(len(rows)):
                wi = qweight(rows[i])
                for j in range(len(rows)):
                    if i == j: continue
                    cand = rows[i] ^ rows[j]
                    if qweight(cand) < wi:
                        rows[i] = cand; wi = qweight(cand); changed = True
        odd = [i for i, v in enumerate(rows) if ycount(v) % 2]
        for i in odd:
            if ycount(rows[i]) % 2 == 0: continue
            best = None
            for j in range(len(rows)):
                if i == j: continue
                cand = rows[i] ^ rows[j]
                if ycount(cand) % 2 == 0:
                    wq = qweight(cand)
                    if best is None or wq < best[0]: best = (wq, cand)
            assert best is not None
            rows[i] = best[1]
        sp = W.Gf2Span(); final = []
        for v in rows:
            fv = np.zeros(2 * nq, np.uint8)
            fv[:nq][rim] = v[:r]; fv[nq:][rim] = v[r:]
            if sp.add(fv.copy()): final.append(fv)
        for v in stab:
            if len(final) == stab.shape[0]: break
            fv = np.zeros(2 * nq, np.uint8)
            fv[:nq][rim] = v[:r]; fv[nq:][rim] = v[r:]
            if ycount(v) % 2 == 0 and sp.add(fv.copy()): final.append(fv)
        assert len(final) == stab.shape[0]
    assert not any(int((v[:nq] & v[nq:]).sum()) % 2 for v in final), 'odd-Y row'
    orig = code["code_rows"]
    fullsp = W.Gf2Span(); basis = []
    for rw in final:
        assert fullsp.add(rw.copy()); basis.append(rw)
    for rw in orig:
        if fullsp.add(rw.copy()): basis.append(rw)
    basis = np.array(basis, np.uint8)
    assert basis.shape[0] == orig.shape[0], (basis.shape, orig.shape)
    sp3 = W.Gf2Span()
    for rw in basis: sp3.add(rw.copy())
    assert all(not sp3.add(rw.copy()) for rw in orig), 'group mismatch'
    print(f'n={n}: rim-adapted basis complete, {basis.shape[0]} rows', flush=True)
    return dict(code, code_rows=basis, sx=basis[:, :nq].copy(),
                sz=basis[:, nq:].copy(), m=basis.shape[0])


