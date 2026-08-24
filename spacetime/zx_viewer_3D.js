// zx_viewer_3D.js
//
// 3D viewer for PyZX graphs with optional Pauli-web overlays.
//
// three.js is resolved through the import map that the notebook injects
// ("three" / "three/addons/", see pyzx settings.javascript_importmap);
// if no import map is present, it falls back to jsdelivr's self-contained
// +esm bundles. Either way an internet connection is required the first
// time a plot is rendered.
//
// Entry point (called from the notebook's draw_graph_3d_pw):
//
//   showGraph3D(tag, graph, node_size, edgeRadius, camera_zoom, show_labels)
//
//     tag         id of the container <div>
//     graph       {nodes, links, pauli_web} JSON produced by draw_graph_3d_pw
//     node_size   sphere radius of spiders (world units)
//     edgeRadius  cylinder radius of edges (world units)
//     camera_zoom 1.0 = graph exactly fits the view; <1 closer, >1 further
//     show_labels draw a "name : phase" sprite above every vertex
//
// The camera auto-centers on the graph's bounding box and auto-fits its
// distance to the bounding sphere, so no manual camera distance is needed.
//
// Vertex types (PyZX VertexType values):
//   0 BOUNDARY  grey sphere        3 H_BOX     yellow box
//   1 Z         green sphere       4 W_INPUT   small black sphere
//   2 X         red sphere         5 W_OUTPUT  black cone
//   6 Z_BOX     green box          7 X_BOX     red box  (red version of the Z-box)
//
// Edge types: 1 SIMPLE black, 2 HADAMARD blue, 3 dashed black.
// Pauli web colors: Z = green, X = red, Y = blue.

export function showGraph3D(tag, graph, node_size = 0.75, edgeRadius = 0.1, camera_zoom = 1.0, show_labels = false, webOffset = 0.0) {

    const container = document.getElementById(tag);
    container.innerHTML = '';
    container.style.width = '100%';
    container.style.height = '600px';

    Promise.all([
        import('three'),
        import('three/addons/controls/OrbitControls.js'),
    ]).catch(() => Promise.all([
        import('https://cdn.jsdelivr.net/npm/three@0.172.0/+esm'),
        import('https://cdn.jsdelivr.net/npm/three@0.172.0/examples/jsm/controls/OrbitControls.js/+esm'),
    ])).then(([THREE, controlsMod]) => {

        const OrbitControls = controlsMod.OrbitControls;

        /* ================= Scene ================= */

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);

        // the container can report 0x0 if the output is rendered before it
        // is laid out (e.g. VSCode webviews); fall back and let the resize
        // observer below correct it once real sizes arrive
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 600;

        /* ---------- camera: auto-center & auto-fit ---------- */

        const bbox = new THREE.Box3();
        graph.nodes.forEach(n => bbox.expandByPoint(new THREE.Vector3(n.x, n.y, n.z)));
        if (bbox.isEmpty()) bbox.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(1, 1, 1));
        const center = bbox.getCenter(new THREE.Vector3());
        const sphere = bbox.getBoundingSphere(new THREE.Sphere());
        // pad so node spheres / web cylinders at the hull are not cut off
        const radius = Math.max(sphere.radius, 1e-3) + 3 * node_size;

        const fov = 45;
        const aspect = width / height;
        // distance at which a sphere of `radius` exactly fills the frustum,
        // checked against both the vertical and the horizontal field of view
        const vDist = radius / Math.sin(THREE.MathUtils.degToRad(fov / 2));
        const hFov = 2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(fov / 2)) * aspect);
        const hDist = radius / Math.sin(hFov / 2);
        const dist = 1.15 * Math.max(vDist, hDist) * camera_zoom;   // 15% breathing room

        const camera = new THREE.PerspectiveCamera(fov, aspect, dist / 100, (dist + 4 * radius) * 4);
        // gentle 3/4 view; change to (0,0,1) for a head-on view
        const viewDir = new THREE.Vector3(0.3, 0.25, 1).normalize();
        camera.position.copy(center).addScaledVector(viewDir, dist);

        // --- robust WebGL context handling ---
        // Browsers cap live WebGL contexts (~16). Many 3D outputs / re-runs exhaust them and
        // new renderers fail with "Error creating WebGL context". Keep a small shared pool and
        // dispose the oldest context when needed. (Works within one window, e.g. JupyterLab; in
        // VS Code each output may be its own frame, so also clear outputs / restart the kernel.)
        const ZX = (window.__ZX_VIEW = window.__ZX_VIEW || { pool: [], cap: 8 });
        function zxDisposeOldest() {
            const old = ZX.pool.shift();
            if (!old) return;
            try { old.forceContextLoss(); } catch (e) {}
            try { old.dispose(); } catch (e) {}
            if (old.domElement && old.domElement.parentNode) {
                const p = old.domElement.parentNode; old.domElement.remove();
                const note = document.createElement('div');
                note.style.cssText = 'color:#888;font:12px sans-serif;padding:8px';
                note.textContent = '(3D view released to free a WebGL context — re-run this cell to view it again)';
                p.appendChild(note);
            }
        }
        const rendererOpts = { antialias: true, powerPreference: 'low-power',
                               failIfMajorPerformanceCaveat: false };
        let renderer;
        if (ZX.pool.length >= ZX.cap) zxDisposeOldest();
        while (true) {
            try { renderer = new THREE.WebGLRenderer(rendererOpts); break; }
            catch (e) { if (ZX.pool.length === 0) throw e; zxDisposeOldest(); }
        }
        ZX.pool.push(renderer);
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.target.copy(center);
        controls.update();

        /* ================= Materials ================= */

        const nodeMaterials = {
            0: new THREE.MeshBasicMaterial({ color: 0x999999 }),                              // boundary
            1: new THREE.MeshBasicMaterial({ color: new THREE.Color("rgb(216,248,216)") }),   // Z
            2: new THREE.MeshBasicMaterial({ color: new THREE.Color("rgb(232,165,165)") }),   // X
            3: new THREE.MeshBasicMaterial({ color: 0xffff00 }),                              // H-box
            4: new THREE.MeshBasicMaterial({ color: 0x000000 }),                              // W input
            5: new THREE.MeshBasicMaterial({ color: 0x000000 }),                              // W output
            6: new THREE.MeshBasicMaterial({ color: new THREE.Color("rgb(216,248,216)") }),   // Z-box (green)
            7: new THREE.MeshBasicMaterial({ color: new THREE.Color("rgb(232,165,165)") }),   // X-box (red) — red version of the Z-box
        };

        const outlineMaterial = new THREE.MeshBasicMaterial({
            color: 0x000000,
            side: THREE.BackSide
        });

        /* ================= Nodes ================= */

        const nodes = {};
        const draggable = [];

        function makeLabelSprite(text) {
            const fontSize = 48;
            const pad = 8;
            const canvas = document.createElement('canvas');
            let ctx = canvas.getContext('2d');
            ctx.font = `${fontSize}px sans-serif`;
            canvas.width = Math.ceil(ctx.measureText(text).width) + 2 * pad;
            canvas.height = fontSize + 2 * pad;
            ctx = canvas.getContext('2d');   // resizing the canvas resets the context
            ctx.font = `${fontSize}px sans-serif`;
            ctx.fillStyle = 'black';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, pad, canvas.height / 2);

            const texture = new THREE.CanvasTexture(canvas);
            const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
            const sprite = new THREE.Sprite(material);
            const h = node_size * 1.6;
            sprite.scale.set(h * canvas.width / canvas.height, h, 1);
            sprite.position.set(0, node_size * 2.0, 0);
            return sprite;
        }

        function makeNodeMesh(n) {
            let geom;
            switch (n.t) {
                case 3: geom = new THREE.BoxGeometry(node_size * 1.5, node_size * 1.5, node_size * 1.5); break;
                case 4: geom = new THREE.SphereGeometry(node_size * 0.4, 16, 16); break;
                case 5: geom = new THREE.ConeGeometry(node_size, node_size, 3); break;
                case 6: geom = new THREE.BoxGeometry(node_size * 1.5, node_size * 1.5, node_size * 1.5); break;
                case 7: geom = new THREE.BoxGeometry(node_size * 1.5, node_size * 1.5, node_size * 1.5); break;  // X-box (red)
                case 0: geom = new THREE.SphereGeometry(node_size * 0.5, 20, 20); break;
                default: geom = new THREE.SphereGeometry(node_size, 20, 20);
            }

            const mesh = new THREE.Mesh(geom, nodeMaterials[n.t] || nodeMaterials[1]);

            const outline = new THREE.Mesh(
                geom.clone().scale(1.12, 1.12, 1.12),
                outlineMaterial
            );

            const group = new THREE.Group();
            group.add(outline);
            group.add(mesh);

            if (show_labels) {
                const text = (n.phase !== undefined && n.phase !== null && String(n.phase) !== '')
                    ? `${n.name} : ${n.phase}`
                    : String(n.name);
                group.add(makeLabelSprite(text));
            }

            group.position.set(n.x, n.y, n.z);

            scene.add(group);
            draggable.push(group);

            return group;
        }

        graph.nodes.forEach(n => {
            nodes[n.name] = makeNodeMesh(n);
        });

        /* ================= Edges ================= */

        const edges = [];

        function makeCylinder(radius, color, opacity = 1) {
            const geom = new THREE.CylinderGeometry(radius, radius, 1, 16);
            const mat = new THREE.MeshBasicMaterial({
                color, transparent: opacity < 1, opacity
            });
            const mesh = new THREE.Mesh(geom, mat);
            scene.add(mesh);
            return mesh;
        }

        function updateCylinder(mesh, a, b) {
            const dir = new THREE.Vector3().subVectors(b, a);
            const len = dir.length();
            mesh.scale.set(1, len, 1);
            mesh.position.copy(a.clone().add(b).multiplyScalar(0.5));
            mesh.quaternion.setFromUnitVectors(
                new THREE.Vector3(0, 1, 0),
                dir.normalize()
            );
        }

        // parallel edges (multigraph backends): fan them out laterally using the
        // payload's per-pair `index`, symmetric around the straight line, so they
        // don't render coincident. Single edges (the usual case) get zero offset.
        const pairCount = {};
        graph.links.forEach(l => {
            const k = l.source + '|' + l.target;
            pairCount[k] = (pairCount[k] || 0) + 1;
        });
        const edgeSpacing = edgeRadius * 3;

        graph.links.forEach(l => {
            const A = nodes[l.source];
            const B = nodes[l.target];
            const cnt = pairCount[l.source + '|' + l.target];
            const idx = (l.index === undefined ? 0 : l.index);
            const eOff = cnt > 1 ? (idx - (cnt - 1) / 2) * edgeSpacing : 0;
            if (l.t === 3) {   // dashed black line (e.g. X-check box -> its data qubits)
                const nDash = 6, frac = 0.55, segs = [];
                for (let i = 0; i < nDash; i++)
                    segs.push({ mesh: makeCylinder(edgeRadius * 0.6, 0x000000),
                                t0: i / nDash, t1: (i + frac) / nDash });
                edges.push({ A, B, dash: true, segs, eOff });
                return;
            }
            const isH = l.t === 2;   // Hadamard edges drawn in blue
            const mesh = makeCylinder(edgeRadius, isH ? 0x0088ff : 0x000000);
            edges.push({ mesh, A, B, eOff });
        });

        /* ================= Pauli web(s) ================= */
        // Half-edges: drawn from the source vertex to the edge midpoint, as a
        // thicker translucent cylinder over the underlying edge.
        //
        // Several webs can be overlaid at once (e.g. the X-check family G_X and
        // the Z-check family G_Z of a foliated code). Each half-edge may carry a
        // `web` index; webs are spread apart by a small lateral offset (computed
        // perpendicular to the edge) so co-located X and Z webs do not z-fight
        // but sit side by side. Entries with no `web` field default to web 0, so
        // a single-web call renders exactly as before.

        const webIds = graph.pauli_web.map(l => (l.web === undefined ? 0 : l.web));
        const nWebs = webIds.length ? (Math.max(...webIds) + 1) : 1;
        // Webs draw ON their wires by default; overlaid webs are distinguished by
        // nested tube radii (web k slightly thinner). webOffset > 0 opts back in to
        // a lateral spread of webOffset * 5 * edgeRadius between webs.
        const webSpacing = edgeRadius * 5 * webOffset;

        graph.pauli_web.forEach(l => {
            const A = nodes[l.source];
            const B = nodes[l.target];

            const color =
                l.t === 'Z' ? 0x00cc00 :
                l.t === 'X' ? 0xff0000 :
                              0x0000ff;   // Y

            const w = (l.web === undefined ? 0 : l.web);
            // symmetric offset: webs centered around 0 so the bundle stays on-edge
            const offsetMag = (w - (nWebs - 1) / 2) * webSpacing;
            // on-wire overlays: nested radii so webs sharing a wire stay visible
            const radius = nWebs > 1 ? edgeRadius * Math.max(2.2, 5.5 - 1.6 * w)
                                     : edgeRadius * 6;

            const mesh = makeCylinder(radius, color, nWebs > 1 ? 0.5 : 0.3);
            edges.push({ mesh, A, B, half: true, offsetMag });
        });

        // a world-space "up" used to build a stable perpendicular for offsetting
        const WEB_UP = new THREE.Vector3(0, 0, 1);
        const WEB_UP_ALT = new THREE.Vector3(0, 1, 0);

        function halfEdgeOffset(a, b, mag) {
            if (!mag) return new THREE.Vector3();
            const dir = new THREE.Vector3().subVectors(b, a).normalize();
            // perpendicular to the edge; fall back if the edge is parallel to WEB_UP
            let perp = new THREE.Vector3().crossVectors(dir, WEB_UP);
            if (perp.lengthSq() < 1e-6) perp.crossVectors(dir, WEB_UP_ALT);
            return perp.normalize().multiplyScalar(mag);
        }

        function updateEdges() {
            edges.forEach(e => {
                if (e.dash) {
                    const off = halfEdgeOffset(e.A.position, e.B.position, e.eOff || 0);
                    e.segs.forEach(s => updateCylinder(s.mesh,
                        e.A.position.clone().lerp(e.B.position, s.t0).add(off),
                        e.A.position.clone().lerp(e.B.position, s.t1).add(off)));
                } else if (e.half) {
                    const mid = e.A.position.clone().lerp(e.B.position, 0.5);
                    const off = halfEdgeOffset(e.A.position, e.B.position, e.offsetMag);
                    updateCylinder(e.mesh,
                        e.A.position.clone().add(off),
                        mid.add(off));
                } else {
                    const off = halfEdgeOffset(e.A.position, e.B.position, e.eOff || 0);
                    updateCylinder(e.mesh,
                        e.A.position.clone().add(off),
                        e.B.position.clone().add(off));
                }
            });
        }

        /* ================= Dragging ================= */
        // Nodes are dragged in the plane perpendicular to the view ray,
        // at the depth where they were picked.

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        let dragged = null;
        const offset = new THREE.Vector3();
        let dragDepth = 0;

        function getMouse(event) {
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        }

        renderer.domElement.addEventListener('pointerdown', e => {
            getMouse(e);
            raycaster.setFromCamera(mouse, camera);
            const hits = raycaster.intersectObjects(draggable, true);
            if (hits.length) {
                dragged = hits[0].object.parent;
                controls.enabled = false;
                dragDepth = camera.position.distanceTo(hits[0].point);
                offset.copy(hits[0].point).sub(dragged.position);
            }
        });

        renderer.domElement.addEventListener('pointermove', e => {
            if (!dragged) return;
            getMouse(e);
            raycaster.setFromCamera(mouse, camera);
            const newPos = new THREE.Vector3();
            raycaster.ray.at(dragDepth, newPos);
            dragged.position.copy(newPos.sub(offset));
        });

        function endDrag() {
            dragged = null;
            controls.enabled = true;
        }
        renderer.domElement.addEventListener('pointerup', endDrag);
        renderer.domElement.addEventListener('pointerleave', endDrag);

        /* ================= Render ================= */

        function onResize() {
            const w = container.clientWidth;
            const h = container.clientHeight;
            if (w === 0 || h === 0) return;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }
        window.addEventListener('resize', onResize);
        if (typeof ResizeObserver !== 'undefined') {
            new ResizeObserver(onResize).observe(container);
        }

        function animate() {
            requestAnimationFrame(animate);
            updateEdges();
            controls.update();
            renderer.render(scene, camera);
        }
        animate();

    }).catch(err => {
        // distinguish "no WebGL at all" from "ran out of WebGL contexts"
        let hasGL = false;
        try { const c = document.createElement('canvas');
              hasGL = !!(c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl')); }
        catch (e) { hasGL = false; }
        const msg = (err && err.message ? err.message : String(err));
        const hint = /context/i.test(msg)
            ? (hasGL
                ? 'Too many live WebGL contexts (browsers cap ~16). Clear other 3D cell outputs / '
                  + 'restart the kernel, or view one plot at a time. In JupyterLab the viewer now '
                  + 'auto-frees old contexts; in VS Code each output is separate, so clear outputs.'
                : 'This notebook frontend has WebGL disabled/unavailable. Open the notebook in a '
                  + 'browser (JupyterLab) or enable hardware acceleration.')
            : 'three.js is loaded from cdn.jsdelivr.net via the import map (with a +esm fallback) — '
              + 'this needs internet access and a frontend that allows remote scripts in outputs.';
        container.innerHTML = '<pre style="color:#b00; white-space:pre-wrap;">'
            + 'zx_viewer_3D.js could not render: ' + msg + '\n' + hint + '</pre>';
        console.error('zx_viewer_3D.js:', err);
    });
}
