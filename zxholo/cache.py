"""Cache (S, L, webs, …) to pickle files in the same format as the
paper's `stab_cache/p4_q5_n{n}.pkl`."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional


def save_code(path, *, g=None, webs=None, H=None, S=None, L=None,
              bulk_vertices=None, boundary_vertices=None, vmap=None):
    """Pickle a code dictionary to `path`. Keys mirror the paper's cache."""
    d = dict(g=g, webs=webs, H=H, S=S, L=L,
             bulk_vertices=bulk_vertices,
             boundary_vertices=boundary_vertices,
             vmap=vmap)
    d = {k: v for k, v in d.items() if v is not None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(d, f)
    return path


def load_code(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
