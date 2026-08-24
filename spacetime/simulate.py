"""Erasure decoding of a spacetime block with BP+OSD-0.

Fault model: every wire of the block is erased independently with probability
p; an erased wire suffers a uniformly random Pauli; the decoder receives the
erased locations as priors (probability 1/2 on both the X and Z column of an
erased wire). A logical failure is recorded when the residual error after
decoding flips the logical correlator.
"""
import numpy as np


def make_decoder(H, q=1e-4):
    import ldpc
    from scipy.sparse import csr_matrix
    return ldpc.BpOsdDecoder(csr_matrix(H), error_channel=[q] * H.shape[1],
                             bp_method="product_sum", max_iter=50,
                             osd_method="osd_cs", osd_order=0), q


def run_point(H, O, n_wires, p, shots=10000, seed=0):
    """Failure count of BP+OSD-0 at erasure rate p over `shots` samples."""
    H = np.asarray(H, dtype=np.uint8)
    O = np.asarray(O, dtype=np.uint8)
    bpd, q = make_decoder(H)
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(shots):
        erased = np.flatnonzero(rng.random(n_wires) < p)
        if erased.size == 0:
            continue
        probs = np.full(H.shape[1], q)
        e = np.zeros(H.shape[1], np.uint8)
        for w in erased:
            probs[2 * w] = probs[2 * w + 1] = 0.5
            pauli = rng.integers(4)
            if pauli in (1, 3):
                e[2 * w] = 1
            if pauli in (2, 3):
                e[2 * w + 1] = 1
        try:
            bpd.update_channel_probs(probs)
        except AttributeError:
            bpd.error_channel = list(probs)
        syndrome = ((H @ e) % 2).astype(np.uint8)
        correction = np.asarray(bpd.decode(syndrome), dtype=np.uint8)
        if ((O @ (e ^ correction)) % 2).any():
            fails += 1
    return fails
