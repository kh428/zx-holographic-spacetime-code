"""Command-line entry point. Minimal — exposes the `build + extract +
smooth + decode + sweep` pipeline as one invocation.

Usage:
    zxholo build --p 4 --q 5 --layers 6 --out stab.pkl
    zxholo decode --cache stab.pkl --channel erasure --p-sweep 0.2,0.3,0.4,0.5 \\
                  --shots 2000 --decoder bposd0 --out results.json

For complex multi-task sweeps, prefer a Python script in
`examples/` — the CLI is intentionally minimal.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def _cmd_build(args):
    from .lattice import build_tiled_codes
    from .gauges import apply_gauge
    from .extract import extract_code
    from .cache import save_code

    g, _ = build_tiled_codes(args.p, args.q, args.layers)
    g = apply_gauge(g, gauge=args.gauge, keep_bulk_idx=args.keep_bulk_idx) \
        if args.gauge else g
    out = extract_code(g, gauge=None)  # gauge already applied above
    save_code(args.out, g=g, H=out["H_raw"], S=out["S"], L=out["L"],
              bulk_vertices=out["bulk_vs"], boundary_vertices=out["boundary_vs"],
              vmap=out["vmap"], webs=None)
    print(f"wrote {args.out}  (S={out['S'].shape}, L={out['L'].shape})")


def _parse_sweep(s):
    return [float(x) for x in s.split(",")]


def _cmd_decode(args):
    from .cache import load_code
    from .smooth import smooth_stabiliser_basis, pick_best_smoothed_basis
    from .lattice import col_swap
    from .decode import ErasureChannel, PauliChannel, MixedChannel
    from .parallel import run_sweep_parallel

    d = load_code(args.cache)
    S = np.asarray(d["S"], dtype=np.uint8)
    L = np.asarray(d["L"], dtype=np.uint8)
    if args.smooth:
        if args.smooth == "greedy-best":
            S = pick_best_smoothed_basis(S, L, seeds=range(args.smooth_seeds),
                                         passes=args.smooth_passes,
                                         sample_js=args.smooth_samples,
                                         stop_weight=args.smooth_stop,
                                         verbose=False)
        else:  # single-seed
            S = smooth_stabiliser_basis(S, passes=args.smooth_passes,
                                        sample_js=args.smooth_samples,
                                        stop_weight=args.smooth_stop,
                                        seed=0)
    H_bp = col_swap(S)
    L_bp = col_swap(L)
    osd = {"bp": None, "bposd0": 0, "bposd10": 10}[args.decoder]
    decoder_spec = dict(max_iter=args.max_iter, osd_order=osd)

    pvals = _parse_sweep(args.p_sweep)
    if args.channel == "erasure":
        channels = [ErasureChannel(p_e=p) for p in pvals]
    elif args.channel == "pauli":
        channels = [PauliChannel(p=p) for p in pvals]
    elif args.channel == "mixed":
        pr_list = _parse_sweep(args.pr_sweep)
        channels = [MixedChannel(p_e=pe, p_r=pr) for pe in pvals for pr in pr_list]
    else:
        raise SystemExit(f"unknown channel: {args.channel}")

    results = run_sweep_parallel(H_bp, L_bp, channels,
                                 shots_per_point=args.shots,
                                 decoder_spec=decoder_spec,
                                 n_jobs=args.jobs, verbose=args.verbose)
    json.dump({"results": results,
               "cache": str(args.cache),
               "decoder": args.decoder,
               "smooth": args.smooth,
               "shots": args.shots},
              open(args.out, "w"), indent=2, default=str)
    print(f"wrote {args.out}  ({len(results)} points)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zxholo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a holographic code cache")
    b.add_argument("--p", type=int, required=True)
    b.add_argument("--q", type=int, required=True)
    b.add_argument("--layers", type=int, required=True)
    b.add_argument("--gauge", choices=["plus", "zero"], default=None)
    b.add_argument("--keep-bulk-idx", type=int, default=0)
    b.add_argument("--out", required=True)
    b.set_defaults(func=_cmd_build)

    d = sub.add_parser("decode", help="run Monte Carlo on a cached code")
    d.add_argument("--cache", required=True)
    d.add_argument("--channel", choices=["erasure", "pauli", "mixed"],
                   required=True)
    d.add_argument("--p-sweep", default="0.2,0.3,0.4,0.5")
    d.add_argument("--pr-sweep", default="0.0,0.05,0.10")
    d.add_argument("--decoder", choices=["bp", "bposd0", "bposd10"],
                   default="bposd0")
    d.add_argument("--shots", type=int, default=1000)
    d.add_argument("--max-iter", type=int, default=200)
    d.add_argument("--smooth", choices=["", "single", "greedy-best"], default="")
    d.add_argument("--smooth-passes", type=int, default=8000)
    d.add_argument("--smooth-samples", type=int, default=1200)
    d.add_argument("--smooth-stop", type=int, default=10)
    d.add_argument("--smooth-seeds", type=int, default=20)
    d.add_argument("--jobs", type=int, default=-1)
    d.add_argument("--verbose", type=int, default=0)
    d.add_argument("--out", required=True)
    d.set_defaults(func=_cmd_decode)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
