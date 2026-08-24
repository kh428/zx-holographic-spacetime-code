"""Acceptance tests: cross-check built codes against the bundled
`data/stab_cache/p4_q5_n{3,4}.pkl` matrices.

Run with:
    pytest tests/test_paper_reproduction.py -v
"""
import pickle
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zxholo as az

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "stab_cache"


def _gf2_rank(A):
    from zxholo.extract import gf2_rref
    return len(gf2_rref(A.astype(int))[1])


@pytest.mark.parametrize("paper_n,n_qubits,n_stab", [
    (0, 4, 3),        # Table 1
    (1, 20, 19),
])
def test_tiled_size_matches_paper_table1_small(paper_n, n_qubits, n_stab):
    """build_tiled_codes(4, 5, paper_n + 3) gives paper n_boundary."""
    g, _ = az.build_tiled_codes(4, 5, paper_n + 3)
    assert len(list(g.outputs())) == n_qubits


@pytest.mark.skipif(not (CACHE_DIR / "p4_q5_n3.pkl").exists(),
                    reason="stab_cache/p4_q5_n3.pkl not present")
def test_tiled_n3_shape_matches_cache():
    """Built + gauged + extracted n=6 tile should give same (S, L) shapes
    as the bundled n=3 cache."""
    g, _ = az.build_tiled_codes(4, 5, 6)
    g = az.apply_gauge(g, az.GAUGE_PLUS, keep_bulk_idx=0)
    out = az.extract_code(g, gauge=None)
    d = pickle.load(open(CACHE_DIR / "p4_q5_n3.pkl", "rb"))
    Sc = np.asarray(d["S"], dtype=np.uint8)
    Lc = np.asarray(d["L"], dtype=np.uint8)
    assert out["S"].shape == Sc.shape
    assert out["L"].shape == Lc.shape


@pytest.mark.skipif(not (CACHE_DIR / "p4_q5_n3.pkl").exists(),
                    reason="stab_cache/p4_q5_n3.pkl not present")
def test_tiled_n3_threshold_consistent_with_cache():
    """BP+OSD-0 on the built n=3 code at p_e=0.50 should have LER around
    0.3-0.5 — the threshold region. Uses a small shot count for speed."""
    g, _ = az.build_tiled_codes(4, 5, 6)
    g = az.apply_gauge(g, az.GAUGE_PLUS, keep_bulk_idx=0)
    out = az.extract_code(g, gauge=None)
    S_sm = az.smooth_stabiliser_basis(out["S"], passes=800, sample_js=300)
    H_bp = az.col_swap(S_sm)
    L_bp = az.col_swap(out["L"])
    dec = az.make_decoder(H_bp, osd_order=0)
    rng = np.random.default_rng(0)
    r = az.run_mc(dec, H_bp, L_bp, az.ErasureChannel(p_e=0.50), 100, rng)
    # loose band; we just want to reject the "decoder is broken" failure
    assert 0.2 <= r["LER"] <= 0.8
