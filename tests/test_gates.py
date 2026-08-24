"""Validation gates for the shipped constructions and data."""
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data' / 'matrices'


def _rank(rows):
    from spacetime.core.wt02_pipeline import Gf2Span
    sp = Gf2Span()
    return sum(sp.add(np.ascontiguousarray(r).copy()) for r in rows)


def test_stacking_reproduces_direct_k4():
    from spacetime.assemble import assemble, load_labelled
    H4b, O4b, W4b = load_labelled(DATA, 1, 4)
    H4a, O4a, wires4, cls = assemble(DATA, 1, 4)
    assert set(wires4) == set(W4b)
    pos = {lab: i for i, lab in enumerate(wires4)}
    inv = np.empty(len(wires4), np.int64)
    for bi, lab in enumerate(W4b):
        inv[pos[lab]] = bi
    H4a_b = np.zeros_like(H4a)
    for wi in range(len(wires4)):
        H4a_b[:, 2 * inv[wi]] = H4a[:, 2 * wi]
        H4a_b[:, 2 * inv[wi] + 1] = H4a[:, 2 * wi + 1]
    rb, ra = _rank(H4b), _rank(H4a_b)
    ru = _rank(list(H4b) + list(H4a_b))
    assert rb == ra == ru == 408


def test_block_web_counts():
    from spacetime.rim_basis import rim_basis_code
    from spacetime.core import foliated_general as FG
    from pyzx.web import compute_stabilisers
    from pyzx.web.compute import compute_detecting_regions
    c1 = rim_basis_code(1)
    blk = FG.build_general_foliated_block(c1['sx'], c1['sz'], 2)
    g, _ = FG.to_zx(blk)
    assert len(compute_detecting_regions(g)) == 51
    assert len(compute_stabilisers(g)) == 104


def test_tile_stabilisers_and_logicals():
    from fractions import Fraction
    from spacetime import e45
    from pyzx.utils import VertexType
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    Y = 1j * X @ Z
    P = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}

    def op(s):
        m = P[s[0]]
        for c in s[1:]:
            m = np.kron(m, P[c])
        return m

    gt0 = e45.build_zx(0)[0]

    def capped(phase):
        gt = gt0.clone()
        b = list(gt.inputs())[0]
        gt.set_inputs(())
        gt.set_type(b, VertexType.X)
        gt.set_phase(b, phase)
        return np.asarray(gt.to_tensor()).reshape(-1)

    V = np.stack([capped(0), capped(Fraction(1, 1))], axis=1)
    V = V / np.sqrt((V.conj().T @ V)[0, 0].real)
    plus = np.array([1, 1], complex) / np.sqrt(2)
    psi = V @ plus
    psi = psi / np.linalg.norm(psi)
    for s in ('IXIX', 'XIXI', 'YYZZ', 'ZZZZ'):
        assert abs(np.vdot(psi, op(s) @ psi) - 1) < 1e-9
    for s in ('IXIX', 'XIXI', 'YYZZ'):
        assert np.allclose(V.conj().T @ op(s) @ V, I2, atol=1e-9)
    assert np.allclose(V.conj().T @ op('ZZZZ') @ V, X, atol=1e-9)
    assert np.allclose(V.conj().T @ op('XZIZ') @ V, Z, atol=1e-9)


def test_simulation_smoke():
    from spacetime.simulate import run_point
    z = np.load(DATA / 'newcon_n0_K2_rim_mats.npz')
    fails = run_point(z['H'], z['O'], int(z['n_wires']), 0.03,
                      shots=2000, seed=7)
    assert 120 <= fails <= 300      # ~10% failure rate, wide binomial margin
