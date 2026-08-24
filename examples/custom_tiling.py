"""Example: experiment with a non-paper (p, q) tiling. Uses the legacy
generic builder `build_zx_holo_generic` — no LEGO_HQEC dependency, no
Table-1 match, but any hyperbolic (p, q) works.

Good smoke test that the decoder side of the pipeline doesn't secretly
bake in assumptions about {4,5} or {5,4}.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zxholo as az


def try_tiling(p, q, layers):
    try:
        g, _ = az.build_zx_holo_generic(p=p, q=q, layers=layers)
        n_in = len(list(g.inputs()))
        n_out = len(list(g.outputs()))
        print(f"{{{p},{q}}} layers={layers}: inputs={n_in}  outputs={n_out}")
        if n_in == 0 or n_out == 0:
            print("    skipping (empty code)")
            return
        g2 = az.apply_gauge(g, gauge=az.GAUGE_PLUS, keep_bulk_idx=0)
        out = az.extract_code(g2, gauge=None)
        S, L = out["S"], out["L"]
        print(f"    S={S.shape}  L={L.shape}")
        if S.shape[0] == 0 or L.shape[0] == 0:
            print("    empty S or L, cannot decode")
            return
        H_bp = az.col_swap(az.smooth_stabiliser_basis(S, passes=500, sample_js=200))
        L_bp = az.col_swap(L)
        dec = az.make_decoder(H_bp, osd_order=0)
        rng = np.random.default_rng(0)
        r = az.run_mc(dec, H_bp, L_bp, az.ErasureChannel(p_e=0.3), 200, rng)
        print(f"    BP+OSD-0 erasure p_e=0.30  LER={r['LER']:.3f}")
    except Exception as e:
        print(f"{{{p},{q}}} layers={layers}: failed — {e.__class__.__name__}: {e}")


def main():
    print("non-paper tilings via build_zx_holo_generic:")
    try_tiling(5, 4, 3)   # pentagon n=1
    try_tiling(4, 5, 3)   # dual-pentagon (different layer policy than tiled)
    try_tiling(6, 4, 2)   # hexagonal-ish
    try_tiling(7, 3, 2)   # heptagonal
    try_tiling(3, 7, 2)   # triangular


if __name__ == "__main__":
    main()
