"""Helper to programmatically construct zx_drawings.ipynb as valid JSON.

Run once:
    python _build_zx_drawings.py

It's kept as a .py helper so the notebook stays regeneratable and we
don't hand-edit JSON.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


CELLS = [
    md([
        "# 2D ZX-diagrams of holographic codes\n",
        "\n",
        "Minimal notebook — uses `pyzx.zx.draw(...)` directly on graphs built by\n",
        "`zxholo`.\n",
        "\n",
        "Covered:\n",
        "1. {5,4} Pentagon HaPPY code — small layers\n",
        "2. {4,5} ZX-holographic code — paper Table-1 sizes\n",
        "3. |+⟩-gauge fixed versions\n",
        "4. Pauli-web stabiliser overlays\n",
        "5. Other hyperbolic tilings ({6,4}, {7,3}, {3,7})\n",
        "\n",
        "Requires `venv_testing` (pyzx Pauli-webs fork + LEGO_HQEC).\n",
        "\n",
        "**Run inside Jupyter** — the fork's `zx.draw` uses the D3 (JavaScript)\n",
        "backend by default in notebooks, which accepts both `scale=` and\n",
        "`pauli_web=` kwargs. A plain Python shell falls back to the\n",
        "matplotlib backend, which only accepts `figsize=` and ignores\n",
        "`pauli_web=`.",
    ]),
    code([
        "import sys\n",
        "from pathlib import Path\n",
        "# repo layout: notebooks/ sits next to the package.\n",
        "sys.path.insert(0, str(Path.cwd().parent))\n",
        "\n",
        "import zxholo as az\n",
        "import pyzx as zx\n",
        "from pyzx.webs import compute_stabilisers\n",
        "\n",
        "# pyzx.draw renders inline; matplotlib is the backend.\n",
        "%matplotlib inline",
    ]),

    md(["## 1. {5,4} Pentagon HaPPY code\n",
        "\n",
        "The original holographic code. `layers=2` gives the central\n",
        "pentagon only (a [[5,1,3]] perfect tensor), `layers=3` adds one ring."]),
    code([
        "g, _ = az.build_zx_holo_generic(p=5, q=4, layers=2)\n",
        "print(f'{{5,4}} layers=2: {len(list(g.inputs()))} bulk, {len(list(g.outputs()))} boundary')\n",
        "zx.draw(g, labels=True, scale=20)",
    ]),
    code([
        "g, _ = az.build_zx_holo_generic(p=5, q=4, layers=3)\n",
        "print(f'{{5,4}} layers=3: {len(list(g.inputs()))} bulk, {len(list(g.outputs()))} boundary')\n",
        "zx.draw(g, labels=False, scale=15)",
    ]),

    md(["## 2. {4,5} ZX-holographic code (paper Table 1)\n",
        "\n",
        "`build_tiled_codes(4, 5, n)` uses the paper-canonical layer policy.\n",
        "Paper's Table-1 layer `n_paper = n - 3`:\n",
        "\n",
        "| call                       | paper `n` | `N_boundary` |\n",
        "|----------------------------|-----------|--------------|\n",
        "| `build_tiled_codes(4,5,3)` | 0         | 4            |\n",
        "| `build_tiled_codes(4,5,4)` | 1         | 20           |\n",
        "| `build_tiled_codes(4,5,5)` | 2         | 76           |\n",
        "| `build_tiled_codes(4,5,6)` | 3         | 284          |"]),
    code([
        "g, _ = az.build_tiled_codes(p=4, q=5, n=3)  # paper n=0\n",
        "print(f'{{4,5}} n=3 (paper n=0): {len(list(g.inputs()))} bulk, {len(list(g.outputs()))} boundary')\n",
        "zx.draw(g, labels=True, scale=25)",
    ]),
    code([
        "g, _ = az.build_tiled_codes(p=4, q=5, n=4)  # paper n=1\n",
        "print(f'{{4,5}} n=4 (paper n=1): {len(list(g.inputs()))} bulk, {len(list(g.outputs()))} boundary')\n",
        "zx.draw(g, labels=False, scale=18)",
    ]),
    code([
        "g, _ = az.build_tiled_codes(p=4, q=5, n=5)  # paper n=2\n",
        "print(f'{{4,5}} n=5 (paper n=2): {len(list(g.inputs()))} bulk, {len(list(g.outputs()))} boundary')\n",
        "zx.draw(g, labels=False, scale=10)",
    ]),

    md(["## 3. |+⟩-gauge fixed\n",
        "\n",
        "Project all bulk legs except `keep_bulk_idx=0` onto |+⟩. This is what\n",
        "the paper's fig 10/13 use for decoder benchmarks."]),
    code([
        "g, _ = az.build_tiled_codes(p=4, q=5, n=4)\n",
        "print(f'before gauge: {len(list(g.inputs()))} bulk inputs')\n",
        "g_gauged = az.apply_gauge(g, gauge=az.GAUGE_PLUS, keep_bulk_idx=0)\n",
        "print(f'after  gauge: {len(list(g_gauged.inputs()))} bulk inputs')\n",
        "zx.draw(g_gauged, labels=False, scale=18)",
    ]),

    md(["## 4. Pauli-web stabiliser overlays\n",
        "\n",
        "`pyzx.webs.compute_stabilisers` returns every Pauli web of the diagram;\n",
        "`zx.draw(g, pauli_web=w)` colours edges by each web's Pauli pattern\n",
        "(X = red, Z = green, Y = blue). We show the first 3 to keep the notebook small."]),
    code([
        "g, _ = az.build_tiled_codes(p=4, q=5, n=4)\n",
        "g_gauged = az.apply_gauge(g, gauge=az.GAUGE_PLUS, keep_bulk_idx=0)\n",
        "webs = compute_stabilisers(g_gauged)\n",
        "print(f'found {len(webs)} Pauli webs')\n",
        "for i, web in enumerate(webs[:3]):\n",
        "    print(f'\\nweb {i}:')\n",
        "    zx.draw(g_gauged, pauli_web=web, scale=18, labels=False)",
    ]),

    md(["## 5. Other hyperbolic tilings\n",
        "\n",
        "Any Schläfli symbol with `1/p + 1/q < 1/2` is a valid hyperbolic\n",
        "tessellation. `build_tiled_codes` (paper-canonical) handles all of\n",
        "them cleanly; the legacy `build_zx_holo_generic` only gives sensible\n",
        "output for {5,4} / {4,5} and degrades at larger `n` on other tilings.\n",
        "\n",
        "**Bulk / boundary counts at `n=4`:**\n",
        "\n",
        "| tile   | `build_tiled_codes(p,q,4)` | `build_zx_holo_generic(p,q,4)` |\n",
        "|--------|----------------------------|--------------------------------|\n",
        "| {5,4}  | 11 bulk / 25 bdry          | 6 / 20                          |\n",
        "| {4,5}  | 13 / 20                    | 13 / 20                         |\n",
        "| {6,4}  | 13 / 42                    | 31 / 114  *(distorted)*         |\n",
        "| {7,3}  | 8 / 21                     | **29 / 28  (inverted!)**        |\n",
        "| {3,7}  | 16 / 12                    | 10 / 18                         |\n",
        "| {5,5}  | 16 / 40                    | 26 / 100  *(distorted)*         |\n",
        "\n",
        "The {7,3} inversion is a bug in the legacy builder's final-layer\n",
        "cleanup: it only handles outermost cells of degree 1 or 2 (fine for\n",
        "{5,4} pentagons) and silently fails on higher-degree cases. Use\n",
        "`build_tiled_codes` below for anything non-pentagon."]),
    code([
        "for (p, q, n) in [(6, 4, 4), (7, 3, 4), (3, 7, 4), (5, 5, 4)]:\n",
        "    g, _ = az.build_tiled_codes(p=p, q=q, n=n)\n",
        "    ni, no = len(list(g.inputs())), len(list(g.outputs()))\n",
        "    print(f'{{{p},{q}}} build_tiled_codes n={n}: {ni} bulk, {no} boundary')\n",
        "    zx.draw(g, labels=False, scale=15)",
    ]),
    md(["### For comparison — legacy generic builder at n=4\n",
        "\n",
        "Included to **show** the distortion, not for use. Compare these to the\n",
        "cell above."]),
    code([
        "for (p, q, L) in [(6, 4, 4), (7, 3, 4), (3, 7, 4)]:\n",
        "    g, _ = az.build_zx_holo_generic(p=p, q=q, layers=L)\n",
        "    ni, no = len(list(g.inputs())), len(list(g.outputs()))\n",
        "    print(f'{{{p},{q}}} build_zx_holo_generic layers={L}: {ni} bulk, {no} boundary')\n",
        "    zx.draw(g, labels=False, scale=12)",
    ]),

    md(["## 6. Pentagon code Pauli webs (for completeness)\n",
        "\n",
        "Same as §4 but on the {5,4} pentagon HaPPY side. Only works at small\n",
        "layers because the number of webs grows fast."]),
    code([
        "g, _ = az.build_zx_holo_generic(p=5, q=4, layers=2)\n",
        "g_gauged = az.apply_gauge(g, gauge=az.GAUGE_PLUS, keep_bulk_idx=0)\n",
        "webs = compute_stabilisers(g_gauged)\n",
        "print(f'{{5,4}} pentagon n=0: {len(webs)} webs')\n",
        "for i, web in enumerate(webs[:2]):\n",
        "    print(f'\\nweb {i}:')\n",
        "    zx.draw(g_gauged, pauli_web=web, scale=25, labels=True)",
    ]),
]

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "venv_testing",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.13",
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "zx_drawings.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out}")
