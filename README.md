# zxholo

A pipeline for building, extracting, and benchmarking holographic
quantum error-correcting codes constructed via ZX-calculus on hyperbolic
tilings.

This repository ships the code required to reproduce the numerical
results of the paper

> *Holographic codes seen through ZX-calculus*,
> K. H. Wan, H. C. W. Price, Q. Yao (2026).

## What's inside

```
zxholo/             core library (build / extract / smooth / decode / plot)
data/stab_cache/    precomputed (S, L) matrices for the paper's {4,5} code at n=3, 4
notebooks/
  paper_plots.ipynb               author's original notebook; hardcoded LER arrays
                                  → pixel-identical paper figures, no MC runtime
  reproduce_paper_figures.ipynb   live-MC reproduction using the cached (S, L)
                                  → agrees with paper_plots arrays within MC noise
  zx_drawings.ipynb               interactive 2D ZX-diagram visualisations
examples/           standalone Python scripts (same content as the notebooks)
tests/              pytest suite (smoke + cached-matrix cross-checks)
docs/
  REPRODUCE.md      step-by-step recipe for each paper figure
  LIMITATIONS.md    non-compliance on non-paper tilings
```

## Two reproduction paths

The bundle ships **two** notebooks for the paper's figures, covering the
two distinct meanings of "exactly reproduce":

1. **Pixel-identical figures** — open `notebooks/paper_plots.ipynb`. This
   is the author's original plotting notebook, with the LER arrays
   hardcoded in-cell (at 16 384 to 786 432 shots per point). Running
   it end-to-end renders the paper's published plots exactly, with zero
   MC wait. Use this when you want to inspect the published figures
   interactively or re-render them at a different dpi / style.
2. **Live MC regeneration** — open `notebooks/reproduce_paper_figures.ipynb`.
   This loads `data/stab_cache/p4_q5_n{3,4}.pkl`, applies the paper's
   row-smoothing heuristic, and runs fresh BP / BP+OSD sweeps at user-
   configurable shot counts. Produces the same curves as
   `paper_plots.ipynb` **within MC noise** (verified: at n=3 p_e=0.475
   on fig 10, 8 192 shots gives LER = 0.2637 vs. author's 0.2698 at
   786 432 shots — 1.2σ apart). Use this when you want to validate the
   pipeline or extend to new parameters.

Bit-identical MC reproduction is not achievable without the author's
original RNG seeds and library versions (BP/OSD has floating-point
non-determinism). Both paths give results statistically indistinguishable
to a physicist.

## Installation

Python 3.11 is required. `LEGO_HQEC` is vendored in this repo under
[`vendor/LEGO_HQEC/`](vendor/LEGO_HQEC/) (Apache 2.0, see
[`NOTICE`](NOTICE)).

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install "pyzx>=0.10"              # from PyPI — has pyzx.web, pyzx.css, pyzx.pauliweb
pip install -e vendor/LEGO_HQEC       # bundled, Apache 2.0
pip install -e .                      # this package: zxholo
```

A `requirements.txt` with pinned versions is provided for reproducing
the paper's environment.

## Quick start

```python
import zxholo as zx
import numpy as np

# Build the paper's {4,5} ZX-holographic code at Table-1 layer n=1:
g, _ = zx.build_tiled_codes(p=4, q=5, n=4)        # layers = paper_n + 3

# Project all-but-one bulk leg onto |+> ("|+>-gauge"):
g = zx.apply_gauge(g, gauge=zx.GAUGE_PLUS, keep_bulk_idx=0)

# Extract the symplectic stabiliser (S) and logical (L) matrices:
out = zx.extract_code(g, gauge=None)

# Apply the paper's row-smoothing heuristic (random subsampling):
S_sm = zx.smooth_stabiliser_basis(out["S"], passes=8000, sample_js=1200,
                                   stop_weight=10, seed=0)

# BP+OSD-0 erasure decoding, parallel over p_e values:
H_bp, L_bp = zx.col_swap(S_sm), zx.col_swap(out["L"])
results = zx.run_sweep_parallel(
    H_bp, L_bp,
    channels=[zx.ErasureChannel(p_e=p) for p in [0.2, 0.3, 0.4, 0.5]],
    shots_per_point=1000,
    decoder_spec=dict(osd_order=0, max_iter=200),
)
for r in results:
    print(r["p_e"], r["LER"])
```

See `notebooks/reproduce_paper_figures.ipynb` for the full paper-figure
pipeline using the precomputed `data/stab_cache/` matrices (avoids
~30 s of redundant code-building per figure).

## Tests

```bash
pytest tests/ -v
```

Ten tests covering: package imports, ZX-tile factories, smoothing-preserves-row-space,
gauge projection, end-to-end decoding, and cross-check against the
precomputed `data/stab_cache/p4_q5_n3.pkl` matrices.

## Reproducing paper figures

See `docs/REPRODUCE.md` for per-figure recipes. TL;DR:

- **fig 9 & 10** (erasure, |+⟩-gauge, BP vs. BP+OSD-0 on smoothed H):
  `notebooks/reproduce_paper_figures.ipynb`, section "Erasure curves"
- **fig 12 & 13** (Pauli depolarising, BP raw vs. BP+OSD-10 smoothed):
  same notebook, section "Pauli curves"
- **fig es_reg** (mixed erasure+Pauli 2D phase diagram):
  same notebook, section "Mixed channel"

The paper's hand-drawn TikZ diagrams and 3-D viewer screenshots are
**not** reproduced here; they live in the arXiv submission's `figures/`
and `nontikz_figures/` directories respectively and were produced by
TikZit (for `.tikz`) and a WebGL viewer external to this package.

## Scope & limitations

This package reproduces the paper's {5,4} pentagon holographic code
and {4,5} ZX-holographic code faithfully (matches Table 1 boundary
counts, row-space equality with `data/stab_cache/p4_q5_n3.pkl`, and
BP+OSD-0 threshold at p_e ≈ 0.5).

For **other hyperbolic Schläfli symbols** — {6,4}, {7,3}, {3,7}, {5,5},
etc. — `build_tiled_codes` runs without error but there are known
caveats around (i) boundary-qubit ordering not being CCW by default
and (ii) leg-dropping for p ≥ 6 in the paper's directed-polygon
labelling. These are detailed in `docs/LIMITATIONS.md`.

## Citation

```bibtex
@article{Wan2026holographic,
  title = {Holographic codes seen through ZX-calculus},
  author = {Wan, Kwok Ho and Price, Henry C. W. and Yao, Qing},
  journal = {arXiv:2601.04467},
  year = {2026},
}
```

## License

`zxholo` is released under the **Apache License, Version 2.0** — see
[LICENSE](LICENSE) for the full text.

Third-party components distributed with this repo:

- `vendor/LEGO_HQEC/` — LEGO_HQEC, Copyright 2024 TU Delft, Apache 2.0.
  See [NOTICE](NOTICE) and
  [vendor/LEGO_HQEC/LICENSE.md](vendor/LEGO_HQEC/LICENSE.md) for
  attribution.

## AI tooling disclosure

Portions of this repository's code and documentation were drafted,
edited, or debugged with the assistance of AI large language models.
All outputs were reviewed, tested, and in many cases substantially
rewritten by the human authors; the scientific results are the work of
the paper's named authors.

# Spacetime ZX-diagram codes

This repository also contains the companion code for the spacetime
(foliated) constructions of the same paper: the bond-extended
$\{4,5\}$ code family, its rim-adapted check basis, the spacetime
decoding matrices, and the erasure simulations. Pauli webs are
computed with the web machinery of [pyzx](https://github.com/zxcalc/pyzx)
(`pyzx.web`), the web-finding algorithm used throughout this work.

## Spacetime (foliated) codes — layout

```
spacetime/          the package: constructions, GF(2) tooling, assembler,
                    BP+OSD-0 erasure harness, 3D viewer rendering
scripts/            entry points (build matrices, assemble deep blocks,
                    run erasure grids, export 3D viewer pages)
data/matrices/      cached parity-check matrices H and logical correlators O
                    for every simulated block, plus the labelled K=2 exports
                    that seed the round-stacking assembler
data/results/       decoder failure counts behind the paper's erasure figures
viewer_html/        standalone interactive 3D pages (three.js via import map)
notebooks/          interactive walkthrough with the pyzx 3D viewer
tests/              validation gates (run `pytest`)
```

### Install (spacetime additions)

```
python -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -e vendor/LEGO_HQEC                     # fused networks (vendored)
```

### Quick start (spacetime additions)

```
# assemble the n=2, K=8 decoding matrices from the labelled K=2 export (seconds)
python scripts/assemble_deep.py 2 8

# decode erasures on any stored block
python scripts/run_erasure.py data/matrices/newcon_n1_K2_rim_mats.npz

# rebuild the fused towers from scratch and check against the cached matrices
python scripts/build_old_towers.py

# export the interactive 3D viewer page
python scripts/make_viewer_html.py
```

`docs/REPRODUCE.md` maps every figure of the paper's spacetime section to the
command that regenerates its data. `pytest` runs the validation gates:
round-stacking versus directly computed matrices, web counts, tile stabiliser
algebra, and a simulation smoke test.

