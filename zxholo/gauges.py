"""Gauge projections on a ZX holographic code diagram.

The paper fixes the "gauge" by projecting all-but-one bulk leg onto a
fixed eigenstate (typically |+⟩, sometimes |0⟩). Equivalently, in the
ZX diagram one fuses the bulk legs into X- or Z-spiders:

  `g.apply_state("/" + "+"*(n-1))`   # |+⟩-gauge, keep input 0 open
  `g.apply_state("/" + "0"*(n-1))`   # |0⟩-gauge, keep input 0 open

This module just wraps that idiom so downstream code doesn't have to
re-derive the projection string.
"""
from __future__ import annotations

GAUGE_PLUS = "plus"
GAUGE_ZERO = "zero"
_VALID = (None, GAUGE_PLUS, GAUGE_ZERO)


def apply_gauge(g, gauge=None, keep_bulk_idx: int = 0):
    """Return a copy of `g` with the bulk legs projected as requested.

    Parameters
    ----------
    g : pyzx.Graph
        Graph with inputs = bulk legs, outputs = boundary qubits.
    gauge : None | 'plus' | 'zero'
        None  → no projection (full [[N, k]] code with k = len(inputs)).
        'plus' → project all-but-`keep_bulk_idx` inputs onto |+⟩ → X-spider.
        'zero' → project all-but-`keep_bulk_idx` inputs onto |0⟩ → Z-spider.
    keep_bulk_idx : int
        Which input to leave open as the single bulk (logical) qubit.
    """
    if gauge not in _VALID:
        raise ValueError(f"gauge must be one of {_VALID}, got {gauge!r}")
    if gauge is None:
        return g
    n_in = len(list(g.inputs()))
    if not (0 <= keep_bulk_idx < n_in):
        raise ValueError(
            f"keep_bulk_idx={keep_bulk_idx} out of range (n_inputs={n_in})")
    char = "+" if gauge == GAUGE_PLUS else "0"
    state = ["/" if i == keep_bulk_idx else char for i in range(n_in)]
    g = g.copy()
    g.apply_state("".join(state))
    return g
