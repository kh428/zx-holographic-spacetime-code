"""Example: specify a small CSS seed code as a ZXTile, then (optionally)
assemble it onto a {4,5} lattice.

Two paths:
    (a) build a tile from a CSS check-matrix pair (SX, LX) via
        `ZXTile.from_css`.
    (b) place that tile on the hyperbolic lattice via `assemble`.

Path (b) is experimental: port-matching across shared edges has not
been validated against `build_tiled_codes`' row-space. For paper-
reproducing work, use `build_tiled_codes`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zxholo as az


def main():
    # --- (a) tile from CSS seed -----------------------------------
    # The paper's r4 tensor: [[4,1,2]] CSS code. X-stabilisers + logical.
    tile = az.ZXTile.from_css(
        SX=[[1, 0, 1, 0], [0, 1, 0, 1]],
        LX=[[1, 1, 0, 0]],
        normal_form="Z-X",
    )
    print("tile from CSS:")
    print(f"  name         = {tile.name!r}")
    print(f"  n_ports      = {tile.n_ports}  (expect 5 = 4 boundary + 1 bulk)")
    print(f"  n_bulk       = {len(tile.bulk_ports)}")
    print(f"  n_gluable    = {len(tile.gluable_ports)}")
    print(f"  graph.vertices: {len(list(tile.graph.vertices()))}")
    print(f"  graph.edges:    {len(list(tile.graph.edges()))}")

    # --- (b) assemble onto a small lattice ------------------------
    # WARNING: assemble() is experimental — shapes should be sensible
    # but row-space equality with `build_tiled_codes` is not yet proven.
    try:
        g, _ = az.assemble(p=4, q=5, layers=3, tile=tile)
        print()
        print("assembled {4,5} layers=3 with tile:")
        print(f"  inputs  = {len(list(g.inputs()))}")
        print(f"  outputs = {len(list(g.outputs()))}")
        print(f"  vertices = {len(list(g.vertices()))}")
    except Exception as e:
        print(f"\nassemble failed: {e.__class__.__name__}: {e}")
        print("(this is the experimental path; fall back to build_tiled_codes)")


if __name__ == "__main__":
    main()
