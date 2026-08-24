"""Smoke tests — fast. Run with:
    cd zxholo && ../venv_testing/bin/pytest tests/
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zxholo as az


def test_imports():
    # If we got this far, __init__ didn't choke.
    assert az.__version__


def test_tile_from_css_r4():
    t = az.ZXTile.from_css(SX=[[1, 0, 1, 0], [0, 1, 0, 1]],
                           LX=[[1, 1, 0, 0]])
    assert t.n_ports == 5           # 4 physical + 1 logical
    assert len(t.bulk_ports) == 1


def test_tile_rotate_preserves_topology():
    t = az.ZXTile.from_css(SX=[[1, 0, 1, 0], [0, 1, 0, 1]],
                           LX=[[1, 1, 0, 0]])
    r = az.rotate_tile(t, 2)
    assert r.ports == [t.ports[(i - 2) % t.n_ports] for i in range(t.n_ports)]
    assert r.graph is t.graph        # topology is unchanged, ports rotated


def test_smooth_rowspace_preserved():
    # build a random full-rank symplectic-ish check, smooth, and verify
    # the row space is unchanged.
    rng = np.random.default_rng(42)
    S = rng.integers(0, 2, size=(15, 40)).astype(np.uint8)
    S_sm = az.smooth_stabiliser_basis(S, passes=200, sample_js=20, seed=0)
    # joint rank = rank(S) = rank(S_sm)  ⇔  same row-span over F_2
    from zxholo.extract import gf2_rref
    r1 = len(gf2_rref(S.astype(int))[1])
    r2 = len(gf2_rref(S_sm.astype(int))[1])
    rj = len(gf2_rref(np.vstack([S, S_sm]).astype(int))[1])
    assert r1 == r2 == rj


def test_apply_gauge_reduces_inputs():
    from zxholo.lattice import build_tiled_codes
    g, _ = build_tiled_codes(4, 5, 3)  # paper n=0 (small)
    n_in0 = len(list(g.inputs()))
    g_plus = az.apply_gauge(g, gauge=az.GAUGE_PLUS, keep_bulk_idx=0)
    assert len(list(g_plus.inputs())) < n_in0 or n_in0 <= 1


def test_decoder_pipeline_end_to_end():
    g, _ = az.build_tiled_codes(4, 5, 4)  # paper n=1
    g = az.apply_gauge(g, az.GAUGE_PLUS, keep_bulk_idx=0)
    out = az.extract_code(g, gauge=None)
    S_sm = az.smooth_stabiliser_basis(out["S"], passes=500, sample_js=100, seed=0)
    H_bp = az.col_swap(S_sm)
    L_bp = az.col_swap(out["L"])
    dec = az.make_decoder(H_bp, osd_order=0)
    rng = np.random.default_rng(0)
    r = az.run_mc(dec, H_bp, L_bp, az.ErasureChannel(p_e=0.35), 150, rng)
    assert 0.0 <= r["LER"] <= 1.0
    # for a {4,5} n=1 code at p_e=0.35, LER should be below the 0.75
    # random-coset ceiling (i.e. the decoder is doing SOMETHING useful).
    assert r["LER"] < 0.6
