"""Plotters for typical zxholo experiment outputs.

Two entry points:
    plot_ler_curves(results_by_label, figure_title, out_path, xlabel=None)
    plot_mixed_heatmap(result_grid, out_path, title=None)

Both take the list-of-dicts that `run_mc` / `run_sweep_parallel` returns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pick_xkey(points):
    if not points:
        return "p"
    for k in ("p_e", "p", "p_r"):
        if k in points[0]:
            return k
    return list(points[0].keys())[0]


def plot_ler_curves(
    results_by_label: Dict[str, List[dict]],
    figure_title: str,
    out_path,
    xlabel: Optional[str] = None,
    ylim=(1e-5, 1),
):
    """LER vs. sweep parameter, one curve per key in `results_by_label`."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for label, pts in results_by_label.items():
        if not pts:
            continue
        xkey = _pick_xkey(pts)
        xs = np.array([p[xkey] for p in pts])
        ys = np.array([p["LER"] for p in pts])
        shots = np.array([p["shots"] for p in pts])
        err = np.sqrt(np.clip(ys * (1 - ys), 1e-12, None) / shots)
        ax.errorbar(xs, np.clip(ys, 1e-5, None), yerr=err, marker="o",
                    capsize=3, lw=1.5, label=label)
    ax.set_xlabel(xlabel or xkey)
    ax.set_ylabel("logical error rate")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(figure_title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_mixed_heatmap(
    points: List[dict],
    out_path,
    title: str = "",
    xkey: str = "p_r",
    ykey: str = "p_e",
):
    """Heatmap of LER on a 2D (xkey, ykey) grid from a flat list of dicts."""
    xs = sorted({p[xkey] for p in points})
    ys = sorted({p[ykey] for p in points})
    G = np.full((len(ys), len(xs)), np.nan)
    for p in points:
        G[ys.index(p[ykey]), xs.index(p[xkey])] = p["LER"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(G, origin="lower", cmap="viridis", vmin=0, vmax=1,
                   extent=[min(xs), max(xs), min(ys), max(ys)],
                   aspect="auto")
    ax.set_xlabel(xkey)
    ax.set_ylabel(ykey)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="LER")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
