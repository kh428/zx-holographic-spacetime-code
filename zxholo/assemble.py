"""`assemble` — place user-specified `ZXTile`s on a {p, q} hyperbolic
lattice and glue them together. Designed to produce a `pyzx.Graph`
compatible with the rest of the pipeline (`extract_code`, `smooth`,
`decode`).

Mirrors the directed-polygon loop from `lattice.build_tiled_codes` but
replaces the hardcoded Z-spider polygon with a user-supplied tile.

NOTE: still experimental — the orientation / port-matching logic across
shared edges is the tricky bit. Validation tests live in
`tests/test_assemble.py`; until they pass, prefer `build_tiled_codes`
for paper-reproducing work.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import networkx as nx

from .tile import ZXTile, rotate_tile


TileDispatch = Union[ZXTile, Callable[[int, int], ZXTile], Callable[[int], ZXTile]]


def _resolve_tile(dispatch: TileDispatch, cell_id: int, layer: int) -> ZXTile:
    """Accept a ZXTile, a fn(layer), or a fn(cell_id, layer)."""
    if isinstance(dispatch, ZXTile):
        return dispatch
    if not callable(dispatch):
        raise TypeError(f"tile dispatch must be ZXTile or callable, got {type(dispatch)}")
    # Try 2-arg; fall back to 1-arg
    try:
        return dispatch(cell_id, layer)
    except TypeError:
        return dispatch(layer)


def assemble(
    p: int,
    q: int,
    layers: int,
    *,
    tile: TileDispatch,
    cross_edge=None,   # pyzx.EdgeType, default = tile.default_cross_edge
    add_bulk_boundaries: bool = True,
):
    """Place `tile` on every cell of the {p, q} hyperbolic tiling.

    Walks the same directed-polygon structure as `lattice.build_tiled_codes`
    (back / left / front / right labelling from `gen_tiled_codes`) and
    stitches user-supplied tiles in place of the paper's Z-spider
    polygons.

    Parameters
    ----------
    p, q, layers : int
        Schläfli symbol + hypertiling layer count. For the paper's
        {4,5} ZX-holographic code at Table-1 layer n, pass layers = n + 3.
    tile : ZXTile or callable
        The per-cell tile template. If a callable, it's invoked as
        `tile(cell_id, layer)` or `tile(layer)` per cell — see
        `_resolve_tile`.
    cross_edge : pyzx.EdgeType, optional
        Edge type used to glue one tile's port to its neighbour's
        opposing port. Defaults to the tile's `default_cross_edge`
        (Hadamard — paper convention).
    add_bulk_boundaries : bool
        If True (default), each tile's `bulk_ports` get a BOUNDARY-type
        vertex appended (making them logical legs). If False, leaves
        them open for the caller.

    Returns
    -------
    Gzx : pyzx.Graph
    Gnx : networkx.Graph
        The underlying lattice graph (same as `build_tiled_codes`).
    """
    import pyzx as zx
    from hypertiling import HyperbolicTiling, TilingKernels
    from LEGO_HQEC.OperatorPush.TensorToolbox import (
        TensorLeg, Tensor, get_tensor_from_id,
    )

    # Re-derive the lattice adjacency exactly as build_tiled_codes does.
    tiling_obj = HyperbolicTiling(p, q, layers,
                                  kernel=TilingKernels.StaticRotationalGraph)

    def get_xy(pid):
        xy = tiling_obj.get_center(pid)
        return float(np.real(xy)), float(np.imag(xy))

    layers_info = {i: tiling_obj.get_layer(i) for i in range(len(tiling_obj))}

    class DP:
        __slots__ = ("back", "left", "right", "front", "all_front")

        def __init__(self):
            self.back = None
            self.left = None
            self.right = None
            self.front = None
            self.all_front = []

    def share_edge(a, b):
        return len(set(tiling_obj.get_nbrs(a)) & set(tiling_obj.get_nbrs(b))) == 2 * (q - 2)

    def shared_nbrs(a):
        return [nb for nb in tiling_obj.get_nbrs(a) if share_edge(a, nb)]

    def direct(pid):
        dp = DP()
        shared = shared_nbrs(pid)
        lvl = layers_info[pid]
        same = [nb for nb in shared if layers_info[nb] == lvl]
        up = [nb for nb in shared if layers_info[nb] < lvl]
        down = [nb for nb in shared if layers_info[nb] > lvl]
        if up:
            dp.back = up[0]
        if down:
            if len(down) == 1:
                dp.front = down[0]
            else:
                dp.all_front = down
        if same:
            dp.left, dp.right = same[0], same[-1]
        return dp

    directed = {pid: direct(pid) for pid in layers_info}

    # Build nx skeleton (for plotting / debugging)
    Gnx = nx.Graph()
    for pid in layers_info:
        Gnx.add_node(pid, layer=layers_info[pid])
    seen_edges = set()
    for pid in layers_info:
        dp = directed[pid]
        for nbr in [dp.back, dp.left, dp.front, dp.right] + list(dp.all_front):
            if nbr is None:
                continue
            e = tuple(sorted((pid, nbr)))
            if e in seen_edges:
                continue
            seen_edges.add(e)
            Gnx.add_edge(*e)
    for pid in Gnx.nodes:
        x, y = get_xy(pid)
        Gnx.nodes[pid]["x"] = x
        Gnx.nodes[pid]["y"] = y

    # ---- assemble ----
    Gzx = zx.Graph()
    # For each cell, record: cell_id -> (port_vertex_ids_in_Gzx, bulk_ids)
    cell_ports: dict = {}

    def orient_tile_for_cell(dp: DP, tile_obj: ZXTile):
        """Return a list of length `p` of Gzx-vertex ids, one per
        directional neighbour slot [back, left, front, right, front_extras…].
        Tile `rotate` may be applied to align port 0 with direction[0].
        For now we take the user's tile as-is and assume port[0] is
        already "back"."""
        # Copy the tile's graph into Gzx with fresh ids, track port mapping.
        id_map = {}
        scale = 30
        cx, cy = 0.0, 0.0  # populated by caller
        for v in tile_obj.graph.vertices():
            ty = tile_obj.graph.type(v)
            q0 = tile_obj.graph.qubit(v)
            r0 = tile_obj.graph.row(v)
            # Gzx coords get a per-cell offset added later.
            new_v = Gzx.add_vertex(ty=ty, qubit=q0, row=r0)
            id_map[v] = new_v
            ph = tile_obj.graph.phase(v)
            if ph:
                Gzx.set_phase(new_v, ph)
        for (a, b) in tile_obj.graph.edges():
            Gzx.add_edge((id_map[a], id_map[b]),
                         tile_obj.graph.edge_type((a, b)))
        port_ids = [id_map[v] for v in tile_obj.ports]
        bulk_ids = [id_map[v] for v in tile_obj.bulk_ports]
        return port_ids, bulk_ids

    for pid in layers_info:
        dp = directed[pid]
        t_obj = _resolve_tile(tile, pid, layers_info[pid])
        ports_g, bulk_g = orient_tile_for_cell(dp, t_obj)
        cell_ports[pid] = dict(
            ports=ports_g,
            bulk=bulk_g,
            tile=t_obj,
            neighbours=[dp.back, dp.left, dp.front, dp.right],
            all_front=dp.all_front,
        )

    # Glue neighbouring tiles across shared edges.
    edge_type_default = None
    glued = set()
    for pid, info in cell_ports.items():
        nbrs = info["neighbours"]
        for dir_idx, nbr in enumerate(nbrs):
            if nbr is None:
                continue
            e = tuple(sorted((pid, nbr)))
            if e in glued:
                continue
            glued.add(e)
            # Which port of `pid` faces `nbr`? dir_idx (0=back, 1=left,
            # 2=front, 3=right)
            my_port_v = info["ports"][dir_idx] if dir_idx < len(info["ports"]) else None
            # And the reciprocal port of `nbr`?  If nbr has pid as
            # back/left/front/right in their directed polygon, take that
            # port slot.
            nbr_info = cell_ports[nbr]
            try:
                back_idx = nbr_info["neighbours"].index(pid)
                nbr_port_v = nbr_info["ports"][back_idx] if back_idx < len(nbr_info["ports"]) else None
            except ValueError:
                # `pid` is in `nbr`'s all_front; punt to unordered match.
                nbr_port_v = nbr_info["ports"][0] if nbr_info["ports"] else None
            if my_port_v is None or nbr_port_v is None:
                continue
            ce = cross_edge or info["tile"].default_cross_edge
            Gzx.add_edge((my_port_v, nbr_port_v), ce)

    # Bulk boundary vertices (logical legs)
    bulk_vs = []
    if add_bulk_boundaries:
        for pid, info in cell_ports.items():
            for bv in info["bulk"]:
                nv = Gzx.add_vertex(ty=zx.VertexType.BOUNDARY,
                                    qubit=Gzx.qubit(bv),
                                    row=Gzx.row(bv) + 1)
                bulk_vs.append(nv)
                Gzx.add_edge((nv, bv))
    Gzx.set_inputs(bulk_vs)
    # Everything still type BOUNDARY in the tile copies and not used as
    # bulk becomes a physical output.
    outputs = []
    all_bulk = {b for info in cell_ports.values() for b in info["bulk"]}
    for v in Gzx.vertices():
        if Gzx.type(v) == zx.VertexType.BOUNDARY and v not in bulk_vs and v not in all_bulk:
            outputs.append(v)
    Gzx.set_outputs(outputs)

    return Gzx, Gnx
