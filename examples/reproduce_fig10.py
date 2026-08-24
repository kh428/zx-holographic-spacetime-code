"""Reproduce fig 10 (BP+OSD-0 erasure, |+>-gauge, smoothed H) at the
paper's {4,5} n=3 size.

Uses the cached `p4_q5_n3.pkl` bundled under `data/stab_cache/` for
speed; set `USE_CACHE = False` to rebuild from scratch (~30 s).

Run:
    python reproduce_fig10.py           # ~1 min (parallel, 2000 shots)
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zxholo as az

USE_CACHE = True
CACHE_PATH = (Path(__file__).resolve().parent.parent
              / "data" / "stab_cache" / "p4_q5_n3.pkl")


def load_or_build():
    if USE_CACHE and CACHE_PATH.exists():
        print(f"loading cache {CACHE_PATH}")
        d = pickle.load(open(CACHE_PATH, "rb"))
        return np.asarray(d["S"], dtype=np.uint8), np.asarray(d["L"], dtype=np.uint8)
    print("building {4,5} tiled n=6 (paper n=3) from scratch")
    g, _ = az.build_tiled_codes(4, 5, 6)
    g = az.apply_gauge(g, gauge=az.GAUGE_PLUS, keep_bulk_idx=0)
    out = az.extract_code(g, gauge=None)
    return out["S"], out["L"]


def main():
    S, L = load_or_build()
    print(f"code: S={S.shape} L={L.shape}")

    print("smoothing (paper's heuristic, single seed, 8000 passes)…")
    S_sm = az.smooth_stabiliser_basis(S, passes=8000, sample_js=1200,
                                      stop_weight=10, seed=0)
    print(f"  max weight: {S.sum(1).max()} → {S_sm.sum(1).max()}")

    H_bp = az.col_swap(S_sm)
    L_bp = az.col_swap(L)
    p_es = [0.10, 0.20, 0.30, 0.35, 0.40, 0.43, 0.46, 0.48, 0.50, 0.53, 0.56]
    chans = [az.ErasureChannel(p_e=p) for p in p_es]
    print(f"parallel sweep: {len(p_es)} p_e × 2000 shots")
    results = az.run_sweep_parallel(
        H_bp, L_bp, chans, shots_per_point=2000,
        decoder_spec=dict(osd_order=0, max_iter=200),
        n_jobs=-1, seed=42,
    )
    for r in results:
        print(f"  p_e={r['p_e']:.2f}  LER={r['LER']:.4f}  "
              f"fails={r['fails']}/{r['shots']}")

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    az.plot_ler_curves(
        {"BP+OSD-0, smoothed (|+⟩-gauge), n=3": results},
        "Fig 10 — BP+OSD-0 erasure, {4,5} holographic code at n=3",
        out_dir / "fig10_reproduced.png",
        xlabel=r"boundary erasure rate $p_e$",
    )
    print(f"\nsaved {out_dir / 'fig10_reproduced.png'}")


if __name__ == "__main__":
    main()
