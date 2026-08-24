# Reproducing the paper's figures

Each recipe below starts from a fresh Python environment and assumes
you've run `pip install -e .` in this repo root (after the two local
editable prerequisites — see `README.md`).

## Interactive route (recommended)

```bash
jupyter lab notebooks/reproduce_paper_figures.ipynb
```

Run all cells top-to-bottom. The notebook:

1. Loads the precomputed `(S, L)` matrices from `data/stab_cache/`.
2. Applies the paper's row-smoothing heuristic.
3. Runs BP / BP+OSD Monte Carlo sweeps (parallel via joblib).
4. Renders LER-vs-p curves and a mixed-channel heatmap.

Default shot counts are modest (~1000 per point) to keep runtime under
5 minutes. For publication-grade statistics raise them to 10 000 on
n=3 and 2500 on n=4; expect ~3 h total.

## Scripted route

`examples/reproduce_fig10.py` is a standalone script for the erasure
BP+OSD-0 curve at paper n=3. Adapt for other figures by changing the
decoder spec and channel class.

## What maps to what

| Paper figure                                                   | Channel     | Decoder     | Generator basis | Cache file        |
|----------------------------------------------------------------|-------------|-------------|-----------------|-------------------|
| fig 9  `54zx_holo_plus_gauge_BP_better_new.png`                | erasure     | BP          | smoothed        | `p4_q5_n{3,4}.pkl`|
| fig 10 `54zx_holo_plus_gauge_BPosd_order0_better_new.png`      | erasure     | BP+OSD-0    | smoothed        | `p4_q5_n{3,4}.pkl`|
| fig 12 `45_pauli_new.png`                                      | Pauli depol | BP          | raw             | `p4_q5_n{3,4}.pkl`|
| fig 13 `45_pauli_better_new.png`                               | Pauli depol | BP+OSD-10   | smoothed        | `p4_q5_n{3,4}.pkl`|
| `es_reg_new.png`                                               | mixed       | BP+OSD-10   | smoothed        | `p4_q5_n{3,4}.pkl`|

Paper's fig 7 (`54zx_holo_no_gauge`) and fig 8 (`54zx_holo_0_gauge_BP`)
require no-gauge and |0⟩-gauge matrices respectively; these are **not**
included in `data/stab_cache/`. They can be regenerated from scratch
via:

```python
import zxholo as zx
g, _ = zx.build_tiled_codes(4, 5, 6)             # paper n=3
# no-gauge:  do not call apply_gauge
# |0>-gauge:
g0 = zx.apply_gauge(g, gauge=zx.GAUGE_ZERO, keep_bulk_idx=0)
out = zx.extract_code(g0, gauge=None)
```

Full (S, L) extraction takes ~10 s for n=3 and ~60 s for n=4.

## Per-figure detail

### Fig 10 (BP+OSD-0 erasure, smoothed)

```python
import numpy as np, pickle, zxholo as zx

S = pickle.load(open("data/stab_cache/p4_q5_n3.pkl","rb"))["S"].astype(np.uint8)
L = pickle.load(open("data/stab_cache/p4_q5_n3.pkl","rb"))["L"].astype(np.uint8)
S_sm = zx.smooth_stabiliser_basis(S, passes=8000, sample_js=1200, stop_weight=10, seed=0)
H_bp, L_bp = zx.col_swap(S_sm), zx.col_swap(L)

channels = [zx.ErasureChannel(p_e=p) for p in np.linspace(0.10, 0.56, 12)]
results = zx.run_sweep_parallel(H_bp, L_bp, channels,
    shots_per_point=2000,
    decoder_spec=dict(osd_order=0, max_iter=200),
    n_jobs=-1)
zx.plot_ler_curves({"BP+OSD-0 n=3": results}, "fig10", "out/fig10.png")
```

### Fig 13 (BP+OSD-10 Pauli, smoothed)

Same template, with:

```python
channels = [zx.PauliChannel(p=p) for p in np.linspace(0.01, 0.15, 10)]
decoder_spec = dict(osd_order=10, max_iter=200)
```

### es_reg (mixed channel, BP+OSD-10)

```python
pe_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
pr_grid = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
channels = [zx.MixedChannel(p_e=pe, p_r=pr) for pe in pe_grid for pr in pr_grid]
results = zx.run_sweep_parallel(H_bp, L_bp, channels,
    shots_per_point=800,
    decoder_spec=dict(osd_order=10, max_iter=200))
zx.plot_mixed_heatmap(results, "out/es_reg.png", title="mixed channel n=3")
```

## Rebuild `data/stab_cache/` from scratch

The shipped caches are time-savers, not primary data. To regenerate
from scratch:

```python
import pickle, zxholo as zx

for paper_n, filename in [(3, "p4_q5_n3.pkl"), (4, "p4_q5_n4.pkl")]:
    g, _ = zx.build_tiled_codes(4, 5, paper_n + 3)
    g = zx.apply_gauge(g, gauge=zx.GAUGE_PLUS, keep_bulk_idx=0)
    out = zx.extract_code(g, gauge=None)
    zx.save_code(f"data/stab_cache/{filename}",
                 g=g, H=out["H_raw"], S=out["S"], L=out["L"],
                 bulk_vertices=out["bulk_vs"],
                 boundary_vertices=out["boundary_vs"],
                 vmap=out["vmap"])
```

Expect ~10 s for n=3 and ~1 min for n=4. Resulting files should be
bit-similar to the shipped caches (Pauli-web ordering is
implementation-dependent, but row-spans are equal).

---

# Reproducing the spacetime-section results

All commands run from the repository root with the environment of
`requirements.txt` (plus LEGO_HQEC for the fused towers).

## Constructions and matrices

| object | command | cached copy |
|---|---|---|
| fused towers, n=1,2 | `python scripts/build_old_towers.py` | `data/matrices/oldcon_n{1,2}_mats.npz` |
| bond-extended block, n, K (direct fold) | `python scripts/build_new_matrices.py <n> <K>` | `data/matrices/labelled_n*_K2.*`, `newcon_*_rim_mats.npz` |
| deep blocks by round stacking | `python scripts/assemble_deep.py <n> <K>` | `data/matrices/newcon_n2_K{6,8}_stacked_mats.npz` |

The direct fold is pure Python and slow beyond K=2 at n=2; the round-stacking
assembler reproduces its output exactly (validated at n=1, K=4 by row-span
equality; see `tests/test_gates.py`) in seconds for any even K.

Matrix files store `H` (detector matrix), `O` (logical correlator rows) and
`n_wires`; columns come in (X, Z) pairs per wire. Labelled exports also store
the wire end labels used by the assembler.

## Erasure simulations (BP+OSD-0)

`python scripts/run_erasure.py <matrices.npz> [shots]` decodes the standard
grid. The decoder is BP+OSD-0 (product-sum belief propagation, 50 iterations,
combination-sweep OSD of order 0) with the erased locations supplied as
priors. Cached failure counts for the paper's figures are in `data/results/`:

- `erasure_K2_and_towers.json` — fixed-depth K=2 family and the fused towers
- `erasure_n1_K4.json`, `erasure_n2_K8.json` — depth-matched family
- `reference_points.json` — two points re-simulated with this repository's
  code in a clean environment, for direct comparison

## Webs and viewers

Closed webs (detectors) and open webs (logical correlators) of a block are
computed with `pyzx.web` (`compute_detecting_regions`, `compute_stabilisers`).
`python scripts/make_viewer_html.py` renders the n=0,1,2 blocks as a
standalone interactive 3D page; `viewer_html/` also contains the
HaPPY-code viewer pages (`happy_3d_viewer.html`, `bh_wh_3d_viewer.html`).
