# Limitations & scope

## What's validated: the paper's constructions

`zxholo` is validated against the numerical results of the paper on:

| tiling   | builder call                       | coverage                                     |
|----------|------------------------------------|----------------------------------------------|
| {5,4}    | `build_tiled_codes(5, 4, n+3)`     | pentagon holographic code (HaPPY)            |
| {4,5}    | `build_tiled_codes(4, 5, n+3)`     | ZX-holographic code — fig 9, 10, 12, 13, es_reg |

For these two tilings:
- Boundary qubit counts match Table 1 of the paper exactly
  (4, 20, 76, 284, 1060 for paper's n = 0..4).
- The extracted stabiliser and logical matrices are row-space equivalent
  to the cached `data/stab_cache/p4_q5_n{3,4}.pkl`.
- BP+OSD-0 erasure threshold sits at p_e ≈ 0.5 (matching fig 10/11).
- Pauli depolarising decoder behaviour — n=4 does not strictly
  out-perform n=3 — reproduces the "decoder-limited" observation of
  fig 12/13.

## What's NOT validated: other hyperbolic tilings

`build_tiled_codes(p, q, n)` accepts any Schläfli symbol `(p, q)` with
`1/p + 1/q < 1/2`, and `build_zx_holo_generic(p, q, layers)` provides
an alternate builder. Both run without error on e.g. {6,4}, {7,3},
{3,7}, {5,5}, producing codes with sensible rank structure and
commuting stabilisers. **However**, there are three known issues that
prevent these being treated as drop-in replacements for the paper's
{5,4} / {4,5} cases:

### 1. Boundary qubit ordering is not CCW by default

The boundary-qubit column order of the extracted `(S, L)` is determined
by `hypertiling`'s internal vertex traversal order, which is not
cyclic around the outermost ring for larger layer counts. Measured
CCW coherence (ratio `2π / N` / mean angular gap between consecutive
indices; 1.0 = perfect CCW, lower = scrambled) from the `testing/`
diagnostic scripts:

| tiling   | paper n | CCW coherence |
|----------|---------|---------------|
| {4,5}    | 0       | 1.000         |
| {4,5}    | 1       | 1.000         |
| {4,5}    | 2       | 0.730         |
| {4,5}    | 3       | 0.613         |

A post-processing pass that re-orders boundary columns by angle around
the centroid of the outermost-ring xy-positions brings coherence to
exactly 1.000 and **preserves the code** (row-space invariant — a
column permutation can't change stabiliser products). This is useful
for producing human-interpretable stabiliser matrices and for
geometry-aware decoders, but does **not** change BP or BP+OSD
performance (weights are unchanged under a column permutation).

If this matters for your application, apply:

```python
import numpy as np
# boundary_xy is (N_boundary, 2); extract from g.outputs() after gauge fix
centroid = boundary_xy.mean(axis=0)
angles = np.arctan2(boundary_xy[:,1] - centroid[1],
                    boundary_xy[:,0] - centroid[0])
perm = np.argsort(angles)   # CCW order
# apply to columns of S, L:
cols = np.empty(2*len(perm), dtype=int)
cols[0::2], cols[1::2] = 2*perm, 2*perm + 1
S_ccw, L_ccw = S[:, cols], L[:, cols]
```

### 2. Legs can be dropped for `p ≥ 6`

The paper's `gen_tiled_codes` uses a 4-slot per-cell directed-polygon
labelling (back / left / front / right). For a cell with `p` edge
neighbours where p > 4, this only accounts for four of them; the
remaining `p − 4` cells-per-edge are silently dropped from the tensor
construction. Concretely:

- For {5,4} pentagons (p = 5): typically 4 directed slots suffice
  because most cells have 1 back + 2 same-layer + ≤ 1 front neighbour.
- For {4,5} squares (p = 4): exactly covered.
- For {6,4} hexagons (p = 6) and larger: cells that would have
  e.g. 1 back + 2 same + 3 front neighbours lose 2 of those legs.

This produces a **valid stabiliser code** (commuting, full-rank) but
one that is "less connected than intended" — a different code than the
one that would be produced by a correct p-port tensor network.

A fully correct arbitrary-(p, q) builder would use angle-sorted cyclic
port indexing and reciprocal-slot gluing; see `docs/DESIGN_NOTES.md`
for a sketch. This is not implemented here.

### 3. Port-level vs. cell-level boundary contraction

At the outermost ring, the paper's convention is:
- **Each max-layer cell becomes one BOUNDARY vertex** in the `pyzx`
  graph, regardless of how many interior neighbours it has.

An alternative convention is:
- **Each dangling port of each max-layer cell becomes one BOUNDARY
  vertex** (no contraction at the boundary).

The first gives Table 1 sizes (4, 20, 76, 284, 1060). The second
produces much larger codes (~50× more qubits at a given `n`); it is a
geometrically distinct code family with its own merits but is **not**
the paper's construction.

`zxholo` uses the paper's convention throughout. A prototype of the
port-level convention was implemented during development
(`testing/t03_principled_scratch.py` in the internal development
repo) but is not shipped.

## Legacy generic builder

`build_zx_holo_generic(p, q, layers)` is a parameterisation of the
paper's pentagon-code builder to arbitrary (p, q). Its final-layer
cleanup assumes outermost cells have degree 1 or 2, which holds for
{5,4} but breaks for:

- {7,3} → **inverts bulk/boundary ratio**, e.g. 29 bulk / 28 boundary
  at `layers=4` (any valid holographic code must have bulk ≪ boundary).
- {6,4}, {5,5} → distorted code sizes.

Use `build_tiled_codes` (the paper-canonical builder) for anything
non-{5,4}. The legacy builder is retained for pentagon-code experiments
and as the `chunk=` hook entry point for custom ZX-tile decorations.

## What this means in practice

- **Reproducing paper figures**: use the defaults. No configuration
  knob required.
- **Experimenting on non-paper tilings**: prefer `build_tiled_codes`,
  apply the CCW post-process if you need interpretable stabilisers,
  and be aware that for `p ≥ 6` you are working with a code whose
  connectivity is less than the tensor-network ideal.
- **Publishing new numerical claims on non-paper tilings**: write a
  correct arbitrary-(p, q) builder first. The sketch in
  `docs/DESIGN_NOTES.md` is a starting point (not shipped code).

---

# Limitations — spacetime (foliated) codes

- The erasure model is phenomenological: wires and measurement events of the
  drawn diagram are the fault locations; there is no circuit-level structure
  inside the measurement events.
- Only erasure noise is simulated. Pauli-noise performance may be more
  sensitive to the choice of web basis and is not covered here.
- The fixed-depth and depth-matched comparisons are finite-size studies of
  n = 0, 1, 2; no size-independent threshold is claimed.
- The direct fold (`scripts/build_new_matrices.py`) is pure Python and slow
  for large blocks; use the round-stacking assembler for deep blocks.
- The round-stacking assembler requires even K and a labelled K=2 export of
  the same code.

- `data/stab_cache/*.pkl` are Python pickle files; load them only from a trusted checkout of this repository (pickle deserialisation executes code by design).
