"""Decoder drivers: erasure, Pauli depolarising, mixed. Uses ldpc.BpDecoder
/ BpOsdDecoder with the paper's interleaved symplectic convention:

    H_bp = col_swap(S)          # swap (x_j, z_j) columns per qubit
    syndrome = H_bp @ e  mod 2
    residual = e ^ e_hat
    logical-fail iff (col_swap(L) @ residual) mod 2 ≠ 0

Channels are dataclasses returning a sampler + a channel-probs fn so the
same MC driver can dispatch to erasure / Pauli / mixed uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


# ----- channels -----------------------------------------------------------

@dataclass
class ErasureChannel:
    p_e: float

    def sample(self, rng, n_qubits):
        E = rng.random(n_qubits) < self.p_e
        erased = np.flatnonzero(E)
        e_vec = np.zeros(2 * n_qubits, dtype=np.uint8)
        if erased.size:
            e_vec[2 * erased] = rng.integers(0, 2, erased.size).astype(np.uint8)
            e_vec[2 * erased + 1] = rng.integers(0, 2, erased.size).astype(np.uint8)
        return dict(e_vec=e_vec, erased=erased)

    def channel_probs(self, n_cols, info):
        ch = np.full(n_cols, 1e-10, dtype=np.float64)
        er = info["erased"]
        if er.size:
            ch[np.concatenate([2 * er, 2 * er + 1])] = 0.5
        return ch


@dataclass
class PauliChannel:
    p: float

    def sample(self, rng, n_qubits):
        e_vec = np.zeros(2 * n_qubits, dtype=np.uint8)
        r = rng.random(n_qubits)
        kind = np.zeros(n_qubits, dtype=np.int8)
        kind[(r >= 1 - self.p) & (r < 1 - 2 * self.p / 3)] = 1  # X
        kind[(r >= 1 - 2 * self.p / 3) & (r < 1 - self.p / 3)] = 2  # Y
        kind[r >= 1 - self.p / 3] = 3  # Z
        e_vec[0::2] = ((kind == 1) | (kind == 2)).astype(np.uint8)
        e_vec[1::2] = ((kind == 2) | (kind == 3)).astype(np.uint8)
        return dict(e_vec=e_vec)

    def channel_probs(self, n_cols, info):
        return np.full(n_cols, 2 * self.p / 3, dtype=np.float64).clip(1e-10, 1 - 1e-10)


@dataclass
class MixedChannel:
    p_e: float
    p_r: float

    def sample(self, rng, n_qubits):
        E = rng.random(n_qubits) < self.p_e
        erased = np.flatnonzero(E)
        non = np.flatnonzero(~E)
        e_vec = np.zeros(2 * n_qubits, dtype=np.uint8)
        if erased.size:
            e_vec[2 * erased] = rng.integers(0, 2, erased.size).astype(np.uint8)
            e_vec[2 * erased + 1] = rng.integers(0, 2, erased.size).astype(np.uint8)
        if non.size:
            r = rng.random(non.size)
            kind = np.zeros(non.size, dtype=np.int8)
            kind[(r >= 1 - self.p_r) & (r < 1 - 2 * self.p_r / 3)] = 1
            kind[(r >= 1 - 2 * self.p_r / 3) & (r < 1 - self.p_r / 3)] = 2
            kind[r >= 1 - self.p_r / 3] = 3
            e_vec[2 * non] = ((kind == 1) | (kind == 2)).astype(np.uint8)
            e_vec[2 * non + 1] = ((kind == 2) | (kind == 3)).astype(np.uint8)
        return dict(e_vec=e_vec, erased=erased)

    def channel_probs(self, n_cols, info):
        ch = np.full(n_cols, 2 * self.p_r / 3, dtype=np.float64)
        er = info["erased"]
        if er.size:
            ch[np.concatenate([2 * er, 2 * er + 1])] = 0.5
        return ch.clip(1e-10, 1 - 1e-10)


# ----- decoder -----------------------------------------------------------

def make_decoder(H_bp: np.ndarray, bp_method: str = "product_sum",
                 max_iter: int = 200, osd_order: Optional[int] = None):
    """Build a BpDecoder or BpOsdDecoder with the paper's defaults."""
    from ldpc import BpDecoder, BpOsdDecoder
    if osd_order is None:
        return BpDecoder(H_bp.astype(np.uint8), error_rate=0.1,
                         bp_method=bp_method, max_iter=max_iter,
                         input_vector_type="syndrome")
    method = "osd_0" if osd_order == 0 else "osd_e"
    return BpOsdDecoder(H_bp.astype(np.uint8), error_rate=0.1,
                        bp_method=bp_method, max_iter=max_iter,
                        osd_method=method, osd_order=osd_order,
                        input_vector_type="syndrome")


def run_mc(decoder, H_bp: np.ndarray, L_bp: np.ndarray, channel,
           shots: int, rng=None) -> dict:
    """Monte-Carlo LER. Returns dict(shots, fails, LER)."""
    if rng is None:
        rng = np.random.default_rng()
    n_qubits = H_bp.shape[1] // 2
    fails = 0
    for _ in range(shots):
        info = channel.sample(rng, n_qubits)
        e_vec = info["e_vec"]
        syndrome = (H_bp @ e_vec) & 1
        decoder.update_channel_probs(
            channel.channel_probs(H_bp.shape[1], info))
        e_hat = decoder.decode(syndrome.astype(np.uint8))
        residual = (e_vec ^ e_hat) & 1
        if np.any((L_bp @ residual) & 1):
            fails += 1
    return dict(shots=shots, fails=fails, LER=fails / shots)
