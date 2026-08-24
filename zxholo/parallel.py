"""Parallel Monte-Carlo drivers (joblib/loky, process-based).

Two callable variants:

1. `run_sweep_parallel(...)` — one joblib task **per (n, p) channel**.
   Simple, low-overhead. Good when `num_channels` >= number of cores.
   Under-utilises cores when `num_channels` is small (e.g. 5 p-values
   on a 14-core machine leaves 9 cores idle).

2. `run_sweep_chunked(...)` — each channel's shots split into `n_chunks`
   sub-tasks. Matches the paper's `hqec_to_zx_corrections.ipynb` cell-9
   pattern: persistent loky pool, shot-level task granularity. Gives
   near-perfect core utilisation on small sweeps, at the cost of a
   little per-task dispatch overhead.

Thread-based drivers are NOT provided here: `ldpc.BpDecoder.decode()`
holds the GIL, so `ThreadPoolExecutor` actually *serialises* the work
(measured 6× slower than process parallelism on this workload). Process
parallelism via loky is strictly better.

Both functions return a list of dicts — one per input channel — with
keys `shots`, `fails`, `LER`, and whichever of `{p_e, p, p_r}` the
channel declared.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
from joblib import Parallel, delayed

from .decode import make_decoder, run_mc


# =========================================================================
# Coarse-grained: one joblib task per channel
# =========================================================================

def _single_point_joblib(H_bp, L_bp, channel, shots, decoder_spec, seed):
    """Run MC for one channel in a worker process."""
    dec = make_decoder(H_bp, **decoder_spec)
    rng = np.random.default_rng(seed)
    res = run_mc(dec, H_bp, L_bp, channel, shots, rng)
    out = dict(res)
    for field in ("p_e", "p", "p_r"):
        v = getattr(channel, field, None)
        if v is not None:
            out[field] = v
    return out


def run_sweep_parallel(
    H_bp: np.ndarray,
    L_bp: np.ndarray,
    channels: Sequence,
    shots_per_point: int,
    decoder_spec: Optional[dict] = None,
    n_jobs: int = -1,
    seed: int = 0,
    verbose: int = 0,
):
    """Process-pool sweep, one worker per channel.

    Preferred when `len(channels) >= n_jobs`. For smaller sweeps use
    `run_sweep_chunked` for better core utilisation.
    """
    decoder_spec = dict(decoder_spec or {})
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=len(channels))
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")
    return parallel(
        delayed(_single_point_joblib)(
            H_bp, L_bp, c, shots_per_point, decoder_spec, int(s))
        for c, s in zip(channels, seeds)
    )


# =========================================================================
# Fine-grained: chunk each channel's shots for better load balance
# =========================================================================

def _chunk_task(H_bp, L_bp, channel, chunk_shots, decoder_spec, seed):
    """Run `chunk_shots` shots of one channel; return fail count."""
    dec = make_decoder(H_bp, **decoder_spec)
    rng = np.random.default_rng(seed)
    n_qubits = H_bp.shape[1] // 2
    fails = 0
    for _ in range(chunk_shots):
        info = channel.sample(rng, n_qubits)
        e_vec = info["e_vec"]
        syndrome = (H_bp @ e_vec) & 1
        dec.update_channel_probs(channel.channel_probs(H_bp.shape[1], info))
        e_hat = dec.decode(syndrome.astype(np.uint8))
        residual = (e_vec ^ e_hat) & 1
        if np.any((L_bp @ residual) & 1):
            fails += 1
    return fails


def run_sweep_chunked(
    H_bp: np.ndarray,
    L_bp: np.ndarray,
    channels: Sequence,
    shots_per_point: int,
    decoder_spec: Optional[dict] = None,
    n_jobs: int = -1,
    n_chunks: Optional[int] = None,
    seed: int = 0,
    verbose: int = 0,
):
    """Process-pool sweep with each channel's shots split into
    `n_chunks` sub-tasks. Delivers near-perfect load balance on
    small sweeps.

    `n_chunks = None` -> `max(1, os.cpu_count() // len(channels))`
    so the total job count is roughly `n_jobs` per channel across the
    whole pool.
    """
    decoder_spec = dict(decoder_spec or {})
    n_ch = len(channels)
    if n_chunks is None:
        cpus = os.cpu_count() or 1
        n_chunks = max(1, cpus // max(1, n_ch))
    # split shots
    base = shots_per_point // n_chunks
    rem = shots_per_point - base * n_chunks
    chunk_sizes = [base + (1 if i < rem else 0) for i in range(n_chunks)]
    # seeds
    ss = np.random.SeedSequence(seed)
    # one seed per (channel, chunk)
    child_seeds = ss.generate_state(n_ch * n_chunks, dtype=np.uint32)
    # build tasks
    tasks = []
    for ci, c in enumerate(channels):
        for k in range(n_chunks):
            s = int(child_seeds[ci * n_chunks + k])
            tasks.append((ci, chunk_sizes[k], c, s))
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")
    fails_chunks = parallel(
        delayed(_chunk_task)(H_bp, L_bp, c, cs, decoder_spec, s)
        for (_, cs, c, s) in tasks
    )
    # reassemble
    out = []
    cursor = 0
    for ci, c in enumerate(channels):
        total_fails = sum(fails_chunks[cursor:cursor + n_chunks])
        cursor += n_chunks
        d = dict(shots=shots_per_point, fails=total_fails,
                 LER=total_fails / shots_per_point)
        for field in ("p_e", "p", "p_r"):
            v = getattr(c, field, None)
            if v is not None:
                d[field] = v
        out.append(d)
    return out
