"""Standalone interactive 3D viewer pages for the bond-extended blocks
(n = 0, 1, 2 at K = 2), rendered with the pyzx 3D viewer
(https://github.com/kh428/pyzx_3d_viewer). The page loads three.js via an
import map, so a browser needs internet access on first view.

Usage: python scripts/make_viewer_html.py [output_dir]
"""
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from spacetime import zx3d
from spacetime import e45
from spacetime import wan_view as WV
from spacetime.core import foliated_general as FG
from spacetime.rim_basis import rim_basis_code

OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path(__file__).resolve().parent.parent / 'viewer_html'
K = 2


def capture(g, webs, **kw):
    grabbed = {}
    orig = zx3d.display
    zx3d.display = lambda h: grabbed.__setitem__('html', h.data)
    try:
        zx3d.draw_graph_3d_pw(g, webs, **kw)
    finally:
        zx3d.display = orig
    return grabbed['html']


scenes = []
for n in (0, 1, 2):
    code2 = rim_basis_code(n)
    nq = code2['sx'].shape[1]
    blk = FG.build_general_foliated_block(code2['sx'], code2['sz'], K)
    g, vm = FG.to_zx(blk)
    P = e45.patch_layers(n)
    txy = np.asarray(P['xy'], float)
    xyG = np.zeros((nq, 2))
    for i in range(P['T']):
        c = txy[i]
        base = np.arctan2(c[1], c[0]) if np.hypot(*c) > 1e-9 else 0.0
        for s_ in range(4):
            th = base + 2 * np.pi * (s_ + 0.5) / 4
            xyG[4 * i + s_] = c + 0.14 * np.array([np.cos(th), np.sin(th)])
    WV.embed_foliated(g, vm, blk, xyG)
    frag = capture(g, None, node_size=[0.14, 0.10, 0.07][n],
                   edge_cyli_radius=[0.05, 0.035, 0.025][n])
    if scenes:
        frag = re.sub(r'<script type="importmap">.*?</script>', '', frag,
                      flags=re.S)
    scenes.append((f'Bond-extended spacetime block, n = {n}, K = {K}', frag))

body = "\n".join(f'<h2>{t}</h2>\n{f}' for t, f in scenes)
html = ('<!doctype html><html><head><meta charset="utf-8">'
        '<title>Bond-extended spacetime blocks</title></head><body>'
        '<h1>Bond-extended spacetime blocks ({4,5} family)</h1>'
        '<p>Drag to orbit, scroll to zoom. Needs internet once for the '
        'three.js import map.</p>' + body + '</body></html>')
OUT.mkdir(exist_ok=True)
(OUT / 'spacetime_blocks_3d_viewer.html').write_text(html)
print('wrote', OUT / 'spacetime_blocks_3d_viewer.html',
      f'({len(html)/1e6:.1f} MB)')
