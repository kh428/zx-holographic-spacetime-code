"""Viewers for the {4,5} patch: purple tensor boxes + standalone HTML.

The stock tensor_box_view.coarse_encoder assumes 5-legged HaPPY tiles and
boxes per BULK leg — useless for an X-gauged patch (one bulk leg total). Here:
one Z_BOX per tile at the tile's own disk position, an H edge per bond, grey
stubs on open boundary slots, and the single bulk leg on the central box.
draw_boxes / _patch from tensor_box_view are reused for the purple styling.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np


def coarse_patch(info):
    """Purple-box graph of a built patch (from wan_patch.build_patch info)."""
    import pyzx as zx
    from pyzx.utils import EdgeType, VertexType

    G = zx.Graph()
    box = []
    for i in range(info["T"]):
        x, y = info["xy"][i]
        v = G.add_vertex(ty=VertexType.Z_BOX)
        G.set_row(v, y)
        G.set_qubit(v, x)
        box.append(v)
    for (i, s, j, s2) in info["bond_list"]:
        G.add_edge((box[i], box[j]), EdgeType.HADAMARD)
    outs = []
    rho = 0.55 * (np.hypot(*(info["xy"][1] - info["xy"][0]))
                  if info["T"] > 1 else 1.0)
    for (i, s) in info["open_slots"]:
        dx, dy = info["slots"][i][s]
        x, y = info["xy"][i]
        b = G.add_vertex(ty=VertexType.BOUNDARY, qubit=x + rho * dx,
                         row=y + rho * dy)
        G.add_edge((box[i], b), EdgeType.SIMPLE)
        outs.append(b)
    if info["gauge"] != "none":
        x, y = info["xy"][0]
        b = G.add_vertex(ty=VertexType.BOUNDARY, qubit=x + 0.25, row=y + 0.25)
        G.add_edge((box[0], b), EdgeType.SIMPLE)
        G.set_inputs((b,))
    G.set_outputs(tuple(outs))
    return G


def capture(g, **kw):
    import zx3d
    box, orig = {}, zx3d.display
    zx3d.display = lambda h: box.__setitem__("html", h.data)
    try:
        zx3d.draw_graph_3d_pw(g, None, **kw)
    finally:
        zx3d.display = orig
    return box["html"]


PAYLOAD = re.compile(r'JSON\.parse\((".*?")\),', re.S)


def write_html(scenes, info_rows, out_path, title):
    """Standalone multi-scene viewer; purple-box patch applied to the global JS."""
    import zx3d
    from tensor_box_view import _patch, BOX_COLOR, BOX_SCALE

    JS = _patch(pathlib.Path(zx3d.JS_PATH).read_text(), BOX_SCALE, BOX_COLOR)
    assert BOX_COLOR in JS and f"node_size * {BOX_SCALE}" in JS, \
        "purple-box patch did not land in the JS the page runs"
    IM = json.dumps(zx3d.settings.javascript_importmap)
    btns = "\n".join(
        f'<button class="cbtn" data-k="{k}">{k.replace("_", " ")}</button>'
        for k in scenes)
    divs = "\n".join(
        f'<div class="scene" id="s-{k}" style="display:none"></div>'
        for k in scenes)
    pay = ",\n".join(f'  "{k}": {v["payload"]}' for k, v in scenes.items())
    sz = json.dumps({k: [v["ns"], v["er"]] for k, v in scenes.items()})
    rows = "\n".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td>"
        f"<td>{r[4]}</td></tr>" for r in info_rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 18px;
        color: #1a1a1a; background: #fff; }}
 h2 {{ margin: 0 0 6px; font-size: 19px; }}
 p.sub {{ margin: 0 0 12px; color: #555; font-size: 13.5px; max-width: 74em; }}
 button {{ font: inherit; padding: 6px 14px; margin-right: 6px; cursor: pointer;
          border: 1px solid #bbb; background: #f4f4f4; border-radius: 5px; }}
 button.on {{ background: #1b6ca8; color: #fff; border-color: #1b6ca8; }}
 table {{ border-collapse: collapse; font-size: 13px; margin: 10px 0 14px; }}
 th, td {{ border: 1px solid #ddd; padding: 3px 9px; text-align: left; }}
 th {{ background: #f7f7f7; }}
 .scene {{ border: 1px solid #e2e2e2; border-radius: 6px; }}
</style></head><body>
<h2>{title}</h2>
<p class="sub">The {{4,5}} ZX-holographic code of arXiv:2601.04467, X-gauged
(every non-central bulk leg is |+&gt;, i.e. fused away). Full ZX: green =
Z spiders, all internal edges Hadamard (blue). Purple boxes: one tensor box
per tile, H edge per bond, grey = boundary legs, the single input = the
central bulk qubit. Drag to orbit, scroll to zoom.</p>
<table><tr><th>scene</th><th>[[N,k]]</th><th>certified d</th><th>tiles/bonds</th>
<th>fold wires (K=2)</th></tr>{rows}</table>
<div>{btns}</div>
{divs}
<script type="importmap">{IM}</script>
<script type="module">
{JS}
window.showGraph3D = showGraph3D;
window.__SC = {{
{pay}
}};
window.__SZ = {sz};
window.__built = {{}};
window.__show = function (k) {{
  document.querySelectorAll('.scene').forEach(d => d.style.display = 'none');
  const host = document.getElementById('s-' + k);
  host.style.display = '';
  if (!window.__built[k]) {{
    host.innerHTML = '<div id="g-' + k + '"></div>';
    const s = window.__SZ[k];
    showGraph3D('g-' + k, JSON.parse(window.__SC[k]), s[0], s[1], 1.0, false, 0.0);
    window.__built[k] = true;
  }}
}};
document.querySelectorAll('.cbtn').forEach(b => b.onclick = () => {{
  document.querySelectorAll('.cbtn').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  window.__show(b.dataset.k);
}});
document.querySelector('.cbtn').click();
</script></body></html>"""
    out_path.write_text(html)
    return out_path


def embed_foliated(g, vmap, blk, xy, r_data=2.6, t_scale=1.2):
    """Foliated block standing on the {4,5} disk: data worldlines at their
    own boundary-leg positions, each check ancilla at the centroid of its
    support, time along the row axis. (Self-contained copy of the validated
    st_viewer.embed_single logic — st_viewer's module-level boot is unwanted.)"""
    import math
    from .core import wt02_pipeline as W

    xy = np.asarray(xy, float)
    xy = xy - xy.mean(axis=0)
    scale = r_data / max(float(np.hypot(*xy.T).max()), 1e-12)
    xy = xy * scale
    anc = {c: p * scale for c, p in W.check_positions(blk, xy / scale).items()}
    groups = {}
    for c, p in anc.items():
        groups.setdefault((round(float(p[0]), 6), round(float(p[1]), 6)),
                          []).append(c)
    spread = 0.075 * r_data
    for (px, py), mem in groups.items():
        if len(mem) == 1:
            continue
        for j, c in enumerate(sorted(mem)):
            th = 2 * math.pi * j / len(mem)
            anc[c] = np.array([px + spread * math.cos(th),
                               py + spread * math.sin(th)])
    seen = {}
    for key, v in vmap.items():
        kind, idx, t = key
        p = xy[idx] if kind == "data" else anc[idx]
        g.set_row(v, t * t_scale)
        g.set_qubit(v, float(p[0]))
        g.set_vdata(v, "z", float(p[1]))
        pos = (round(float(p[0]), 4), round(float(p[1]), 4),
               round(t * t_scale, 4))
        assert pos not in seen, f"spiders {seen[pos]} and {v} coincide"
        seen[pos] = v
    for b in list(g.inputs()) + list(g.outputs()):
        nb = next(iter(g.neighbors(b)))
        g.set_qubit(b, g.qubit(nb))
        g.set_vdata(b, "z", g.vdata(nb, "z", default=0.0))
        g.set_row(b, -t_scale if b in g.inputs()
                  else 2 * blk.rounds * t_scale)
    return dict(scale=scale, anc=anc, data_xy=xy)


def checks_as_boxes(g, vmap):
    """Clone with every ANCILLA spider retyped Z_BOX: each syndrome
    measurement event renders as a purple box, repeating every round —
    the spacetime checks, literally."""
    from pyzx.utils import VertexType

    g2 = g.clone()
    n_box = 0
    for key, v in vmap.items():
        if key[0] == "ancilla":
            g2.set_type(v, VertexType.Z_BOX)
            n_box += 1
    return g2, n_box
