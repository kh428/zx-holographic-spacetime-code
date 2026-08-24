# Vendored: LEGO_HQEC

This directory contains a vendored copy of the `LEGO_HQEC` Python
package. It is redistributed here under its original Apache License
2.0; see `LICENSE.md` in this directory for the full text.

## Why vendored

`zxholo.lattice.build_tiled_codes` imports
`LEGO_HQEC.OperatorPush.TensorToolbox` for the directed-polygon tensor
bookkeeping that the paper's construction uses. The package is not on
PyPI, so bundling it here avoids having users track down a separate
upstream to reproduce the paper's results.

## Installation

From the repo root:

```bash
pip install -e vendor/LEGO_HQEC
```

After this, `import LEGO_HQEC` works and `zxholo.build_tiled_codes`
functions end-to-end. `pyproject.toml` inside `vendor/LEGO_HQEC/`
handles the usual setuptools install.

## Changes from upstream

None — the files in this directory are a verbatim copy of
`LEGO_HQEC`'s upstream package source (`LEGO_HQEC/`, `pyproject.toml`,
`LICENSE.md`, `README.md`, `requirements.txt`). No patches, no
rebranding. The only omissions are `Examples/`, `build/`,
`lego_hqec.egg-info/`, and `readme_pics/` — these are not needed to
import the package and would inflate the bundle.

## Upstream

Original copyright 2024 TU Delft, licensed Apache 2.0. If you use
LEGO_HQEC in your own work independently of `zxholo`, prefer installing
from the upstream source.
