"""Paper-canonical tiling builder — `build_tiled_codes(p, q, n)`.

This is the constructor that matches Table 1 of the paper
(N_boundary = 4, 20, 76, 284, 1060 for n = 0..4 on the {4,5} tiling)
and produces the cached matrices under `data/stab_cache/`.

Mechanics:
  - uses `hypertiling.HyperbolicTiling` with the `StaticRotationalGraph`
    kernel;
  - uses `LEGO_HQEC.OperatorPush.TensorToolbox` to build a
    directed-polygon (back / left / front / right) adjacency before
    expansion;
  - drops "final-layer issues" (dangling cells and all-same-layer
    cliques at the outermost ring) before expanding each interior cell
    into a polygon of Z-spiders;
  - returns `(Gzx, nxG)` where `Gzx.inputs()` = bulk (logical) legs and
    `Gzx.outputs()` = boundary (physical) qubits.

The returned graph is compatible with the paper's gauge idiom
`Gzx.apply_state("/" + "+"*(len(Gzx.inputs())-1))` (equivalent to
`zxholo.apply_gauge(Gzx, gauge='plus', keep_bulk_idx=0)`).
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
import networkx as nx


def build_tiled_codes(p: int, q: int, n: int):
    """Build the paper's ZX holographic code on the {p, q} tiling using
    the `gen_tiled_codes` layer policy. `n` is the hypertiling `layers`
    argument here (NOT the paper's Table-1 layer counter — that one maps
    to n = Table1_n + 3 for {4,5}; see audit notes).

    Returns `(Gzx, G_nx)` where `Gzx` is a `pyzx.Graph` whose inputs are
    bulk (logical) legs and outputs are boundary (physical) legs.
    """
    from hypertiling import HyperbolicTiling, TilingKernels
    from LEGO_HQEC.OperatorPush.TensorToolbox import (
        TensorLeg, Tensor, get_tensor_from_id,
    )
    import pyzx as zx

    tiling_obj = HyperbolicTiling(p, q, n,
                                  kernel=TilingKernels.StaticRotationalGraph)

    def get_xy(poly_id):
        xy = tiling_obj.get_center(poly_id)
        return np.real(xy), np.imag(xy)

    layers_info = {i: tiling_obj.get_layer(i) for i in range(len(tiling_obj))}

    # ---- directed-polygon adjacency ----
    class DirectedPolygon:
        def __init__(self, poly_id):
            self.poly_id = poly_id
            self.back = None
            self.left = None
            self.right = None
            self.front = None
            self.left_front = None
            self.right_front = None
            self.all_front = []

    def share_common_edge(poly_id1, poly_id2, q):
        nbrs1 = set(tiling_obj.get_nbrs(poly_id1))
        nbrs2 = set(tiling_obj.get_nbrs(poly_id2))
        return len(nbrs1.intersection(nbrs2)) == 2 * (q - 2)

    def get_shared_edge_neighbors(poly_id):
        return [nb for nb in tiling_obj.get_nbrs(poly_id)
                if share_common_edge(poly_id, nb, q)]

    def determine_directed_neighbors(poly_id, layers_info, q=5):
        dp = DirectedPolygon(poly_id)
        shared = get_shared_edge_neighbors(poly_id)
        current_layer = layers_info[poly_id]
        same_layer = [nb for nb in shared if layers_info[nb] == current_layer]
        upper_layer = [nb for nb in shared if layers_info[nb] < current_layer]
        lower_layer = [nb for nb in shared if layers_info[nb] > current_layer]
        if upper_layer:
            dp.back = upper_layer[0]
        if lower_layer:
            if len(lower_layer) == 1:
                dp.front = lower_layer[0]
            else:
                dp.all_front = lower_layer
        if same_layer:
            dp.left, dp.right = same_layer[0], same_layer[-1]
        return dp

    directed = {pid: determine_directed_neighbors(pid, layers_info, q)
                for pid in layers_info}
    poly_id_mapping = {pid: pid for pid in layers_info}

    # ---- tensor list with LEGO_HQEC bookkeeping ----
    def has_any_neighbor(poly_id):
        dp = directed[poly_id]
        return any([dp.back, dp.left, dp.right, dp.front,
                    dp.left_front, dp.right_front, dp.all_front])

    def make_tensor(poly_id, lst):
        if not has_any_neighbor(poly_id):
            return
        dp = directed[poly_id]
        t = Tensor(poly_id_mapping[poly_id], 0)
        if poly_id == 0:
            for f in dp.all_front:
                t.add_leg(TensorLeg('I', (poly_id_mapping[f], None)))
        else:
            for dir_name in ['back', 'left', 'front', 'right']:
                nb = getattr(dp, dir_name, None)
                if nb is not None:
                    t.add_leg(TensorLeg('I', (poly_id_mapping[nb], None)))
                else:
                    t.add_leg(TensorLeg('I', None))
        t.layer = layers_info[poly_id]
        lst.append(t)

    tensor_list = []
    for pid in layers_info:
        make_tensor(pid, tensor_list)

    # resolve leg→leg connections
    for t in tensor_list:
        for leg in t.legs:
            if leg.connection is None:
                continue
            nb = get_tensor_from_id(tensor_list, leg.connection[0])
            if nb is None:
                continue
            for idx, nl in enumerate(nb.legs):
                if nl.connection is not None and nl.connection[0] == t.tensor_id:
                    leg.connection = (nb.tensor_id, idx)
                    break

    # ---- base networkx graph ----
    G = nx.Graph()
    for t in tensor_list:
        G.add_node(t.tensor_id, layer=t.layer)
    seen = set()
    for t in tensor_list:
        for leg in t.legs:
            if leg.connection is None:
                continue
            u, v = t.tensor_id, leg.connection[0]
            e = tuple(sorted((u, v)))
            if e not in seen:
                seen.add(e)
                G.add_edge(u, v)
    for nid in G.nodes:
        x, y = get_xy(nid)
        G.nodes[nid]['x'] = float(x)
        G.nodes[nid]['y'] = float(y)

    # ---- drop final-layer issues ----
    OG_nodes = deepcopy(G.nodes())
    layer_of = {i: G.nodes[i]['layer'] for i in OG_nodes}
    max_layer = max(layer_of.values())
    to_remove = []
    for i in OG_nodes:
        if layer_of[i] == max_layer:
            nbrs = list(G.neighbors(i))
            if nbrs and all(G.nodes[j]['layer'] == max_layer for j in nbrs):
                to_remove.append(i)
    G.remove_nodes_from(to_remove)
    for i in G.nodes():
        if G.nodes[i]['layer'] == max_layer:
            for j in list(G.neighbors(i)):
                if G.nodes[j]['layer'] == max_layer:
                    G.remove_edge(i, j)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

    # ---- polygon expansion (same as build.py but retains parent-sorted map) ----
    def expand_node(G, v, pos, node_counter, t=0.3):
        nbrs = list(G.neighbors(v))
        expanded = []
        for u in nbrs:
            new_label = node_counter[0]
            node_counter[0] += 1
            x0, y0 = pos[v]
            x1, y1 = pos[u]
            node_pos = (x0 * (1 - t) + x1 * t, y0 * (1 - t) + y1 * t)
            G.add_node(new_label, pos=node_pos, x=node_pos[0], y=node_pos[1],
                       layer=-1, parent=v, type='expanded', original_edge=u)
            pos[new_label] = node_pos
            expanded.append((new_label, u))
            if G.has_edge(v, u):
                G.remove_edge(v, u)
            G.add_edge(v, new_label, parent=v)
            G.add_edge(new_label, u, parent=v)
        x0, y0 = pos[v]
        expanded.sort(key=lambda pr: np.arctan2(pos[pr[0]][1] - y0,
                                                pos[pr[0]][0] - x0))
        sorted_nodes = [n for n, _ in expanded]
        n_ = len(sorted_nodes)
        for i in range(n_):
            a, b = sorted_nodes[i], sorted_nodes[(i + 1) % n_]
            G.add_edge(a, b, parent=v)

    def expand_all(G, pos, skip):
        ex = []
        for nid in G.nodes():
            try:
                ex.append(int(nid))
            except ValueError:
                pass
        counter = [max(ex, default=-1) + 1]
        originals = [v for v in list(G.nodes()) if v not in skip]
        for v in originals:
            if G.has_node(v):
                expand_node(G, v, pos, counter, t=0.3)

    pos = {i: (G.nodes[i]['x'], G.nodes[i]['y']) for i in G.nodes()}
    max_lvl_nodes = [i for i in G.nodes()
                     if G.nodes[i].get('layer', -1) == max_layer]
    expand_all(G, pos, skip=max_lvl_nodes)

    # ---- pyzx graph ----
    Gzx = zx.Graph()
    scale = 30
    for i in G.nodes():
        vt = (zx.VertexType.BOUNDARY
              if G.nodes[i].get('layer', -1) == max_layer
              else zx.VertexType.Z)
        Gzx.add_vertex(index=i, ty=vt,
                       qubit=scale * G.nodes[i]['x'],
                       row=scale * G.nodes[i]['y'])
    for a, b in G.edges():
        pa = G.nodes[a].get('parent', -1)
        pb = G.nodes[b].get('parent', -1)
        if pa == pb:
            Gzx.add_edge((a, b), zx.EdgeType.HADAMARD)
        elif pa != -1 and b == pa:
            Gzx.add_edge((a, b), zx.EdgeType.HADAMARD)
        elif pb != -1 and a == pb:
            Gzx.add_edge((a, b), zx.EdgeType.HADAMARD)
        elif (Gzx.type(a) != zx.VertexType.BOUNDARY
              and Gzx.type(b) != zx.VertexType.BOUNDARY):
            Gzx.add_edge((a, b), zx.EdgeType.HADAMARD)
        else:
            Gzx.add_edge((a, b), zx.EdgeType.SIMPLE)

    boundary_nodes = [v for v in Gzx.vertices()
                      if Gzx.type(v) == zx.VertexType.BOUNDARY]
    bulk_nodes = []
    for i in G.nodes():
        if (G.nodes[i].get('type', -1) == -1
                and Gzx.type(i) != zx.VertexType.BOUNDARY):
            nv = Gzx.add_vertex(ty=zx.VertexType.BOUNDARY,
                                qubit=Gzx.qubit(i),
                                row=Gzx.row(i) + 1)
            bulk_nodes.append(nv)
            Gzx.add_edge((nv, i))
    Gzx.set_inputs(bulk_nodes)
    Gzx.set_outputs(boundary_nodes)

    return Gzx, G


# =========================================================================
# Legacy generic builder. Parameterises the paper's pentagon-code
# construction to arbitrary (p, q). Does NOT match Table 1 sizes — use
# `build_tiled_codes` above for that. Useful for pentagon-code experiments
# and as the `chunk`-hook entry point for custom ZX-tile decorations.
# =========================================================================

from collections import defaultdict
from typing import Callable, Optional


def col_swap(M: np.ndarray) -> np.ndarray:
    """Swap columns 2j ↔ 2j+1 (interleaved symplectic layout)."""
    out = np.empty_like(M)
    out[:, 0::2] = M[:, 1::2]
    out[:, 1::2] = M[:, 0::2]
    return out


# ----- default polygon-expansion chunk (paper's {5,4} HaPPY-style) ---------

def _build_networkx_graph(adjacent_matrix, center_coords, p, colors=None):
    G = nx.Graph()
    n_centers = center_coords.shape[0]
    for y in range(len(adjacent_matrix)):
        if y >= n_centers:
            sector = (y - 1) // (n_centers - 1)
            index = (y - 1) % (n_centers - 1) + 1
            rot = center_coords[index] * np.exp(1j * sector * 2 * np.pi / p)
            x = float(np.real(rot))
            y_pos = float(np.imag(rot))
        else:
            z = center_coords[y]
            x = float(np.real(z))
            y_pos = float(np.imag(z))
        node_data = {"pos": (x, y_pos)}
        if colors is not None:
            node_data["node_color"] = colors[y]
        G.add_node(y, **node_data)
    for y, row in enumerate(adjacent_matrix):
        for index in row:
            if index >= len(adjacent_matrix):
                continue
            G.add_edge(y, index)
    return G


def _expand_node_along_edges(G, v, pos, node_counter, t=0.3, extra_attrs=None):
    if extra_attrs is None:
        extra_attrs = {}
    neighbors = list(G.neighbors(v))
    expanded_nodes = []
    for u in neighbors:
        new_label = node_counter[0]
        node_counter[0] += 1
        x0, y0 = pos[v]
        x1, y1 = pos[u]
        node_pos = (x0 * (1 - t) + x1 * t, y0 * (1 - t) + y1 * t)
        node_attr = {'parent': v, 'type': 'expanded', 'original_edge': u}
        node_attr.update(extra_attrs)
        G.add_node(new_label, pos=node_pos, **node_attr)
        pos[new_label] = node_pos
        expanded_nodes.append((new_label, u))
        if G.has_edge(v, u):
            G.remove_edge(v, u)
        G.add_edge(v, new_label, parent=v)
        G.add_edge(new_label, u, parent=v)
    x0, y0 = pos[v]

    def ang(pair):
        node, _ = pair
        x, y = pos[node]
        import math
        return math.atan2(y - y0, x - x0)

    expanded_nodes.sort(key=ang)
    sorted_nodes = [n for n, _ in expanded_nodes]
    n_ = len(sorted_nodes)
    for i in range(n_):
        u_, w_ = sorted_nodes[i], sorted_nodes[(i + 1) % n_]
        G.add_edge(u_, w_, parent=v)
    return sorted_nodes


def _expand_all(G, pos, skip_nodes=None, t=0.3, extra_attrs=None):
    if skip_nodes is None:
        skip_nodes = []
    if extra_attrs is None:
        extra_attrs = {}
    existing_ints = []
    for n in G.nodes():
        try:
            existing_ints.append(int(n))
        except ValueError:
            if '_' in str(n):
                existing_ints.append(int(str(n).split('_')[0]))
    counter = [max(existing_ints, default=-1) + 1]
    original_nodes = [v for v in list(G.nodes()) if v not in skip_nodes]
    for v in original_nodes:
        if G.has_node(v):
            _expand_node_along_edges(G, v, pos, counter, t, extra_attrs)


def _default_polygon_chunk(gzx, nxG, OG_nodes, vertices_to_level, level_to_vertices):
    """Default chunk: the paper's per-cell pattern — Z-spider polygon via the
    expanded network, plus one Z-spider per cell connected to a BOUNDARY
    vertex labelled `logical` (the bulk leg)."""
    import pyzx as zx
    scale = 30
    for ii in nxG.nodes():
        xy = nxG.nodes[ii]["pos"]
        xy = (scale * xy[0], scale * xy[1])
        gzx.add_vertex(ty=zx.VertexType.Z, qubit=xy[0], row=xy[1])
    for _ in nxG.edges():
        gzx.add_edges(nxG.edges(), zx.EdgeType.HADAMARD)

    max_lvl = max(level_to_vertices)
    for ii in list(nxG.nodes()):
        if ii in OG_nodes and vertices_to_level[ii] == max_lvl:
            deg = len(list(gzx.neighbors(ii)))
            if deg == 1:
                gzx.set_type(ii, zx.VertexType.BOUNDARY)
                gzx.set_vdata(ii, "logical_physcial", "physical")
            elif deg == 2:
                n0 = list(gzx.neighbors(ii))[0]
                n1 = list(gzx.neighbors(ii))[1]
                nnew0 = zx.rules.unspider(gzx, [ii, [n0]])
                nnew1 = zx.rules.unspider(gzx, [ii, [n1]])
                gzx.set_qubit(nnew0,
                              q=gzx.qubit(n0) - 0.5 * (gzx.qubit(n0) - gzx.qubit(nnew0)))
                gzx.set_row(nnew0,
                            r=gzx.row(n0) - 0.5 * (gzx.row(n0) - gzx.row(nnew0)))
                gzx.set_qubit(nnew1,
                              q=gzx.qubit(n1) - 0.5 * (gzx.qubit(n1) - gzx.qubit(nnew1)))
                gzx.set_row(nnew1,
                            r=gzx.row(n1) - 0.5 * (gzx.row(n1) - gzx.row(nnew1)))
                gzx.remove_vertex(ii)
                gzx.set_type(nnew0, zx.VertexType.BOUNDARY)
                gzx.set_vdata(nnew0, "logical_physcial", "physical")
                gzx.set_type(nnew1, zx.VertexType.BOUNDARY)
                gzx.set_vdata(nnew1, "logical_physcial", "physical")

    for ii in OG_nodes:
        if ii not in level_to_vertices[max_lvl]:
            gzx.set_type(ii, zx.VertexType.Z)
            nn = gzx.add_vertex(ty=zx.VertexType.BOUNDARY,
                                qubit=gzx.qubit(ii),
                                row=gzx.row(ii) + 1)
            gzx.set_vdata(nn, "logical_physcial", "logical")
            gzx.add_edge((ii, nn), zx.EdgeType.SIMPLE)

    for ii in gzx.edges():
        a, b = ii
        pa = nxG.nodes.get(a, {}).get("parent", None)
        pb = nxG.nodes.get(b, {}).get("parent", None)
        if pa is not None and pb is not None and pa != pb:
            gzx.set_edge_type(ii, zx.EdgeType.SIMPLE)
        elif (gzx.vdata(a, "logical_physcial") == "physical"
              or gzx.vdata(b, "logical_physcial") == "physical"):
            gzx.set_edge_type(ii, zx.EdgeType.SIMPLE)

    inp, out = [], []
    for ii in gzx.vertices():
        role = gzx.vdata(ii, "logical_physcial")
        if role == "physical":
            out.append(ii)
        elif role == "logical":
            inp.append(ii)
    gzx.set_inputs(inp)
    gzx.set_outputs(out)


# ----- top-level -----------------------------------------------------------

ChunkFn = Callable[..., None]


def build_zx_holo_generic(
    p: int,
    q: int,
    layers: int,
    kernel: str = "GRG",
    chunk: Optional[ChunkFn] = None,
    expand_t: float = 0.3,
):
    """Build a ZX-diagram holographic code on the {p, q} hyperbolic tiling.

    Parameters
    ----------
    p, q : int
        Schläfli symbol of the hyperbolic tiling. Paper uses (5, 4) for
        the pentagon holographic code and (4, 5) for the {4,5} ZX
        holographic code. Any (p, q) with 1/p + 1/q < 1/2 is a valid
        hyperbolic tessellation.
    layers : int
        Number of hypertiling rings. Maps to the paper's layer counter
        `n` via  n = layers - 2  (so layers=2 -> n=0).
    kernel : str
        hypertiling kernel. 'GRG' is used here to match the cached
        matrices; 'GRC' is the current hypertiling replacement but
        lacks the `center_coords` attribute this routine needs.
    chunk : callable or None
        A function `chunk(gzx, nxG, OG_nodes, v2lvl, lvl2v)` that decorates
        the `pyzx.Graph` `gzx` after the NetworkX skeleton `nxG` has been
        expanded. Default is the paper's Z-spider polygon + bulk Z-spider
        chunk.
    expand_t : float
        Polygon expansion parameter (0 = at cell centre, 1 = at cell
        vertices). Paper uses 0.3.

    Returns
    -------
    gzx : pyzx.Graph
        ZX-diagram with `inputs()` = bulk/logical legs and `outputs()` =
        boundary/physical legs.
    nxG : networkx.Graph
        The intermediate expanded tiling graph (useful for plotting).
    """
    import pyzx as zx
    from hypertiling import HyperbolicGraph

    if 1 / p + 1 / q >= 0.5:
        raise ValueError(f"{{{p},{q}}} is not a hyperbolic tiling "
                         f"(require 1/p + 1/q < 1/2)")
    if layers < 2:
        raise ValueError("layers must be >= 2 (hypertiling requirement)")

    G = HyperbolicGraph(p, q, layers, kernel=kernel)
    adjacent_matrix = G.get_nbrs_list()
    center_coords = G.center_coords

    palette = ["#81b29a", "#f2cc8f", "#e07a5f"]
    colors = [palette[G.get_reflection_level(i) % len(palette)]
              for i in range(len(adjacent_matrix))]

    nxG = _build_networkx_graph(adjacent_matrix, center_coords, p, colors)
    OG_nodes = deepcopy(nxG.nodes())
    pos = nx.get_node_attributes(nxG, "pos")

    G.get_nbrs_list_sector()
    vertices_to_level = {}
    for ii in nxG.nodes():
        vertices_to_level[ii] = int(G.get_reflection_level(ii))
    level_to_vertices = defaultdict(list)
    for k, v in vertices_to_level.items():
        level_to_vertices[v].append(k)

    _expand_all(nxG, pos,
                skip_nodes=level_to_vertices[max(level_to_vertices)],
                t=expand_t)

    gzx = zx.Graph()
    if chunk is None:
        chunk = _default_polygon_chunk
    chunk(gzx, nxG, OG_nodes, vertices_to_level, level_to_vertices)

    return gzx, nxG
