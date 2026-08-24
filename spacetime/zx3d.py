"""
zx3d.py — 3-D rendering of PyZX ZX-diagrams with Pauli-web overlays, three back-ends:

  draw_graph_3d_pw   interactive three.js viewer (zx_viewer_3D.js, WebGL, orbit/drag)
  draw_graph_3d_mpl  static matplotlib 3-D fallback (no WebGL / no internet needed)
  to_tikz3d          TikZiT-style .tikz export of the SAME diagram (fixed oblique
                     projection, previewable with `preview` before committing to LaTeX)

Shared conventions
  coordinates : x = g.row(v), y = g.qubit(v), z = g.vdata(v, 'z', 0)
                ((x, y) = spatial plane, z = the third / foliation / time axis)
  Pauli webs  : any pyzx.pauliweb.PauliWeb, or a LIST of them. Colours follow the
                RGB = XZY convention (X = red, Z = green, Y = blue). Webs are always
                drawn exactly ON their wires; a list is overlaid with nested tube
                radii so co-located webs stay distinguishable (web_offset > 0 opts
                in to a lateral spread instead).
  extensions  : vertex type 7 renders as a red box (an "X-box", the red mirror of
                pyzx's Z_BOX = 6); edge type 3 renders dashed black (handy for
                check-box membership edges in code diagrams).
"""
from __future__ import annotations
import cmath
import json
import os
import random
import string
from typing import Optional, Union, List

try:
    from IPython.display import display, HTML
except ImportError:                       # headless use: capture html_str only
    class HTML:                            # noqa: N801
        def __init__(self, data):
            self.data = data

    def display(_):
        pass
from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.pauliweb import PauliWeb
from pyzx.utils import (VertexType, EdgeType, phase_to_s, get_z_box_label,
                        get_h_box_label, hbox_has_complex_label, settings)

JS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zx_viewer_3D.js")

Webs = Optional[Union[PauliWeb, List[PauliWeb]]]


def _phase_str(g: BaseGraph[VT, ET], v: VT) -> str:
    """Phase label of a vertex as a JSON-safe string (mirrors pyzx's drawing.graph_json).

    Z-boxes and H-boxes can carry an arbitrary complex label instead of a phase, so
    show that label. Everything else goes through phase_to_s, which also handles
    Fraction phases that json.dumps cannot serialize."""
    ty = g.type(v)
    if ty == VertexType.Z_BOX:
        return str(get_z_box_label(g, v))
    if ty == VertexType.H_BOX and hbox_has_complex_label(g, v):
        label = get_h_box_label(g, v)
        if cmath.isclose(label, -1):
            return ''  # a plain Hadamard box
        return str(label)
    try:
        return phase_to_s(g.phase(v), ty, poly_with_pi=True)
    except Exception:
        return str(g.phase(v))


def _as_web_list(pauli_web: Webs) -> List[PauliWeb]:
    if pauli_web is None:
        return []
    webs = list(pauli_web) if isinstance(pauli_web, (list, tuple)) else [pauli_web]
    # drop empty webs: they would still consume an offset slot and shift the
    # remaining webs off their wires
    return [w for w in webs if w is not None and w.half_edges()]


def _xyz(g, v):
    return float(g.row(v)), float(g.qubit(v)), float(g.vdata(v, 'z', 0.0))


# ================= interactive three.js viewer =================
def draw_graph_3d_pw(
    g: BaseGraph[VT, ET],
    pauli_web: Webs = None,
    labels: bool = False,
    node_size: float = 0.2,
    edge_cyli_radius: float = 0.05,
    camera_zoom: float = 1.0,
    web_offset: float = 0.0,
    camera_dist: Optional[float] = None,  # deprecated, ignored: camera auto-fits now
):
    """Draw a PyZX graph in 3D with optional Pauli-web highlighting (three.js).

    pauli_web: a single PauliWeb or a LIST of webs — always drawn exactly ON their
    wires; overlaid webs are kept distinguishable by nested tube radii. Set
    web_offset > 0 (e.g. 1.0) to spread a list laterally instead. The camera
    auto-centers on the graph and auto-fits its distance; camera_zoom scales that
    distance (1.0 = fitted, <1 closer, >1 further). Interaction: left-drag orbits,
    scroll zooms, dragging a vertex moves it."""
    if camera_dist is not None:
        print("note: camera_dist is deprecated and ignored — the camera now "
              "auto-fits the graph; use camera_zoom (1.0 = fitted) instead.")

    graph_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    vs = list(g.vertices())

    minx = min((g.row(v) for v in vs), default=0)
    maxx = max((g.row(v) for v in vs), default=0)
    miny = min((g.qubit(v) for v in vs), default=0)
    maxy = max((g.qubit(v) for v in vs), default=0)
    minz = min((g.vdata(v, "z", default=0) for v in vs), default=0)
    maxz = max((g.vdata(v, "z", default=0) for v in vs), default=0)
    coords = {v: (g.row(v) - (minx + maxx) / 2,
                  g.qubit(v) - (miny + maxy) / 2,
                  g.vdata(v, "z", default=0) - (minz + maxz) / 2) for v in vs}

    nodes = [{
        'name': str(v),
        'x': float(coords[v][0]),
        'y': float(coords[v][1]),
        'z': float(coords[v][2]),
        't': int(g.type(v)),
        'phase': _phase_str(g, v),
        'ground': g.is_ground(v),
    } for v in vs]

    links = []
    counts = {}
    for e in g.edges():
        s, t = str(g.edge_s(e)), str(g.edge_t(e))
        i = counts.get((s, t), 0)
        links.append({'source': s, 'target': t, 't': int(g.edge_type(e)), 'index': i})
        counts[(s, t)] = i + 1

    pw_edges = []
    for k, web in enumerate(_as_web_list(pauli_web)):
        for (s, t), p in web.half_edges().items():
            pw_edges.append({'source': str(s), 'target': str(t), 't': p, 'web': k})

    graph_json_str = json.dumps({'nodes': nodes, 'links': links, 'pauli_web': pw_edges})
    # Double-encode for embedding: json.dumps of the STRING yields a valid JS string
    # literal (quotes/backslashes/newlines escaped at the JS layer, so JSON.parse sees
    # the intact JSON text — a bare '...' literal would eat the JSON escaping).
    # <-escape '<' so a label can never form '</script>' inside the HTML.
    js_payload = json.dumps(graph_json_str).replace('<', '\\u003c')

    with open(JS_PATH) as f:
        library_code = f.read()

    # The import map resolves the viewer's bare "three" / "three/addons/" imports
    # to CDN URLs (OrbitControls itself does `import ... from "three"`, which
    # fails without it).
    html_str = f"""
    <div style="overflow:auto; background-color: white" id="graph-output-3d-{graph_id}"></div>
    <script type="importmap">
    {json.dumps(settings.javascript_importmap)}
    </script>
    <script type="module">
    {library_code}
    showGraph3D('graph-output-3d-{graph_id}', JSON.parse({js_payload}), {node_size}, {edge_cyli_radius}, {camera_zoom}, {'true' if labels else 'false'}, {web_offset});
    </script>
    """
    display(HTML(html_str))


# ================= static matplotlib fallback =================
def draw_graph_3d_mpl(g: BaseGraph[VT, ET], pauli_web: Webs = None, node_size: float = 45,
                      figsize=(7, 7), elev: float = 22, azim: float = -60,
                      web_lw: float = 3.0):
    """No-WebGL 3-D fallback: render the graph with matplotlib (works in any notebook
    frontend or headless, no GPU / no internet needed). Static, but rotatable with
    %matplotlib widget. Colours match the viewer: Z = green, X = red, boundary = grey,
    H-box gold, Z-box green square, X-box (type 7) red square; Hadamard edges blue,
    edge type 3 dashed; web X = red / Z = green / Y = blue."""
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    pos = {v: _xyz(g, v) for v in g.vertices()}
    colmap = {0: '0.6', 1: (0.20, 0.78, 0.20), 2: (0.90, 0.30, 0.30),
              3: 'gold', 4: 'black', 5: 'black',
              6: (0.20, 0.78, 0.20), 7: (0.90, 0.30, 0.30)}
    boxy = {3, 6, 7}                                  # drawn as squares
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for e in g.edges():
        s, t = g.edge_s(e), g.edge_t(e)
        et = int(g.edge_type(e))
        col = '#0088ff' if et == EdgeType.HADAMARD else '0.45'
        ax.plot(*[[pos[s][i], pos[t][i]] for i in range(3)], '-', color=col,
                lw=0.6, alpha=0.55, dashes=(2, 1.5) if et == 3 else ())
    wcol = {'X': 'red', 'Z': 'green', 'Y': 'blue'}
    for k, w in enumerate(_as_web_list(pauli_web)):
        lw = max(1.2, web_lw - 0.9 * k)          # nested widths: shared wires stay visible
        for (a, b), p in w.half_edges().items():
            mid = [(pos[a][i] + pos[b][i]) / 2 for i in range(3)]
            ax.plot(*[[pos[a][i], mid[i]] for i in range(3)], '-',
                    color=wcol.get(str(p).upper()[:1], 'k'), lw=lw, alpha=0.5)
    for ty in sorted(set(int(g.type(v)) for v in g.vertices())):
        vs = [v for v in g.vertices() if int(g.type(v)) == ty]
        ax.scatter([pos[v][0] for v in vs], [pos[v][1] for v in vs],
                   [pos[v][2] for v in vs],
                   s=node_size * (0.4 if ty == 0 else 1.0),
                   marker='s' if ty in boxy else 'o',
                   color=colmap.get(ty, '0.5'),
                   edgecolors='k', linewidths=0.3, depthshade=True)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    plt.tight_layout()
    import matplotlib
    if 'agg' not in matplotlib.get_backend().lower():
        plt.show()


# ================= TikZ export (fixed oblique projection) =================
def project(x, y, z, depth=0.62, tilt_deg=26.0, zscale=1.0):
    """Oblique axonometric projection: x horizontal, y receding (angle tilt_deg,
    foreshortened by `depth`), z vertical (scaled by `zscale`). Returns (u, v) page
    coordinates. Used identically by `preview` and `to_tikz3d`, so the matplotlib
    preview is a faithful proof of the TikZ output."""
    import numpy as np
    a = np.deg2rad(tilt_deg)
    u = x + depth * np.cos(a) * y
    v = depth * np.sin(a) * y + zscale * z
    return u, v


def _web_edges(g, web):
    """Collapse a PauliWeb's half-edges to {frozenset((s,t)): 'X'|'Z'|'Y'}."""
    acc = {}
    for (a, b), p in web.half_edges().items():
        key = frozenset((int(a), int(b)))
        p = str(p).upper()[:1]
        if key in acc and acc[key] != p:
            acc[key] = 'Y'                            # X and Z on the same wire -> Y
        else:
            acc[key] = acc.get(key, p)
    return acc


_WEBCOL = {'X': (0.86, 0.20, 0.20), 'Z': (0.20, 0.65, 0.25), 'Y': (0.27, 0.40, 0.85)}
_WEBTIKZ = {'X': 'xweb', 'Z': 'zweb', 'Y': 'yweb'}


def _perp_offset(pa, pb, mag):
    """Page-space unit perpendicular to the projected edge, scaled by mag."""
    import numpy as np
    d = np.array([pb[0] - pa[0], pb[1] - pa[1]])
    n = np.hypot(*d)
    if n < 1e-9 or not mag:
        return (0.0, 0.0)
    return (-d[1] / n * mag, d[0] / n * mag)


def preview(g, webs: Webs = None, path='preview.png', depth=0.62, tilt_deg=26.0,
            zscale=1.0, node_size=26, figsize=(7, 9), dpi=150, title=None,
            time_edges=None, time_alpha=0.5, draw_boundary=True, web_sep=0.0):
    """Draw the projected diagram with matplotlib so the viewing angle can be
    eyeballed before committing to LaTeX — the SAME projection as `to_tikz3d`.

    webs: a PauliWeb or list of PauliWebs to overlay (web_sep > 0 spreads a list
    perpendicular to each edge, page units, mirroring the interactive viewer).
    time_edges: a set of frozenset({a, b}) endpoint pairs drawn faint (alpha
    time_alpha) as foliation/time bonds. draw_boundary=False leaves BOUNDARY
    spiders as bare line stubs."""
    import numpy as np
    # render on an explicit Agg canvas: preview only ever saves a file, so it must
    # not touch the process-global backend (a pyplot import here could lock a fresh
    # session to Agg and silently suppress later draw_graph_3d_mpl display)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    webs = _as_web_list(webs)
    P = {v: project(*_xyz(g, v), depth=depth, tilt_deg=tilt_deg, zscale=zscale)
         for v in g.vertices()}
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    TE = time_edges or set()
    for e in g.edges():
        s, t = g.edge_s(e), g.edge_t(e)
        (ux, uy), (vx, vy) = P[s], P[t]
        et = int(g.edge_type(e))
        had = (et == EdgeType.HADAMARD)
        is_time = frozenset((int(s), int(t))) in TE
        a = time_alpha if is_time else 1.0
        col = '#4488ff' if had else ('0.4' if is_time else '0.25')
        ax.plot([ux, vx], [uy, vy], '-', lw=0.8, color=col, alpha=a,
                dashes=(2, 1.5) if (had or et == 3) else (), zorder=1)
    nweb = len(webs)
    for k, web in enumerate(webs):
        mag = (k - (nweb - 1) / 2) * web_sep
        for key, p in _web_edges(g, web).items():
            a, b = tuple(key)
            (ux, uy), (vx, vy) = P[a], P[b]
            ox, oy = _perp_offset(P[a], P[b], mag)
            ax.plot([ux + ox, vx + ox], [uy + oy, vy + oy], '-', lw=4.5,
                    color=_WEBCOL[p], alpha=0.55, solid_capstyle='round', zorder=2)
    order = sorted(g.vertices(), key=lambda v: P[v][1])
    for v in order:
        t = int(g.type(v))
        mk = 'o'
        if t == VertexType.Z:
            c, ec = '#ddffdd', 'k'
        elif t == VertexType.X:
            c, ec = '#ff8888', 'k'
        elif t == VertexType.H_BOX:
            c, ec, mk = 'gold', 'k', 's'
        elif t == VertexType.Z_BOX:
            c, ec, mk = '#ddffdd', 'k', 's'
        elif t == 7:                                  # X-box (red mirror of Z_BOX)
            c, ec, mk = '#ff8888', 'k', 's'
        else:
            if not draw_boundary:                     # leave in/out stubs as bare lines
                continue
            c, ec = '0.6', '0.3'                      # boundary grey
        x, y = P[v]
        ax.plot([x], [y], mk, ms=np.sqrt(node_size), mfc=c, mec=ec, mew=0.6, zorder=3)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    return path


def to_tikz3d(g, webs: Webs = None, depth=0.62, tilt_deg=26.0, zscale=1.0, scale=1.0,
              boundary_style='none', web_behind=True, time_edges=None,
              time_opt='draw=black, opacity=0.22', draw_boundary=True, web_sep=0.0):
    """Return a TikZiT-style `.tikz` string of the 3-D diagram, using the SAME
    projection as `preview`. Node styles `Z dot` / `X dot`, `hadamard edge`, and web
    styles `xweb`/`zweb`/`yweb` come from the bundled `zx3d.tikzstyles` (or your own
    quantum.tikzstyles). time_edges (set of frozenset({a, b})) are drawn faint with
    `time_opt`; draw_boundary=False leaves BOUNDARY spiders invisible so in/out
    stubs read as bare lines. webs may be a list; web_sep > 0 spreads overlaid webs
    perpendicular to each edge (in projected page units) like the 3-D viewer —
    with web_sep=0 web edges stay anchored to the nodes (nicer for TikZiT editing)."""
    webs = _as_web_list(webs)
    TE = time_edges or set()
    verts = list(g.vertices())
    P = {v: project(*_xyz(g, v), depth=depth, tilt_deg=tilt_deg, zscale=zscale)
         for v in verts}
    nid = {v: i for i, v in enumerate(verts)}
    fmt = lambda p: f"({scale * p[0]:.3f}, {scale * p[1]:.3f})"

    def _vstyle(v):
        t = int(g.type(v))
        if t == VertexType.Z:
            return 'Z dot'
        if t == VertexType.X:
            return 'X dot'
        if t == VertexType.H_BOX:
            return 'hadamard'
        if t == VertexType.Z_BOX:
            return 'Z box'
        if t == 7:             # X-box (red mirror of Z_BOX)
            return 'X box'
        return 'none'          # BOUNDARY / other -> blank node

    L = ["\\begin{tikzpicture}", "\t\\begin{pgfonlayer}{nodelayer}"]
    # nodes — ALWAYS emitted (so edges can reference them); BOUNDARY spiders use the
    # invisible `none` style unless a visible boundary_style is requested.
    for v in verts:
        st = _vstyle(v)
        if st == 'none' and draw_boundary and boundary_style != 'none':
            st = boundary_style
        L.append(f"\t\t\\node [style={st}] ({nid[v]}) at {fmt(P[v])} {{}};")
    L.append("\t\\end{pgfonlayer}")
    L.append("\t\\begin{pgfonlayer}{edgelayer}")
    weblines = []
    nweb = len(webs)
    for k, web in enumerate(webs):
        mag = (k - (nweb - 1) / 2) * web_sep
        for key, p in _web_edges(g, web).items():
            a, b = tuple(key)
            if mag:
                ox, oy = _perp_offset(P[a], P[b], mag)
                pa = (P[a][0] + ox, P[a][1] + oy)
                pb = (P[b][0] + ox, P[b][1] + oy)
                weblines.append(f"\t\t\\draw [style={_WEBTIKZ[p]}] {fmt(pa)} to {fmt(pb)};")
            else:
                weblines.append(f"\t\t\\draw [style={_WEBTIKZ[p]}] ({nid[a]}) to ({nid[b]});")
    if web_behind:
        L += weblines
    for e in g.edges():
        s, t = g.edge_s(e), g.edge_t(e)
        if frozenset((int(s), int(t))) in TE:
            opt = time_opt
        elif g.edge_type(e) == EdgeType.HADAMARD:
            opt = 'style=hadamard edge'
        elif int(g.edge_type(e)) == 3:
            opt = 'style=dashed edge'
        else:
            opt = ''
        sstr = f"[{opt}] " if opt else ""
        L.append(f"\t\t\\draw {sstr}({nid[s]}) to ({nid[t]});")
    if not web_behind:
        L += weblines
    L.append("\t\\end{pgfonlayer}")
    L.append("\\end{tikzpicture}")
    return "\n".join(L)


def write_tikz3d(g, path, webs: Webs = None, **kw):
    s = to_tikz3d(g, webs=webs, **kw)
    with open(path, 'w') as f:
        f.write(s)
    return path
