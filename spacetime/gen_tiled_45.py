"""Verbatim copy of hqec_to_zx.ipynb cell 18 (gen_tiled_codes for the
{p,q} ZX-holographic codes). Only addition: the pyzx import, which the
original cell inherited from its notebook session."""
import pyzx as zx

import numpy as np
import networkx as nx
from hypertiling import HyperbolicTiling, TilingKernels
from LEGO_HQEC.OperatorPush.TensorToolbox import TensorLeg, Tensor, get_tensor_from_id, swap_tensor_legs

def gen_tiled_codes(p, q, n):
    # ---------------------------
    # 1️⃣ Generate tiling
    # ---------------------------
    # p, q, n = 4, 5, 6
    tiling_obj = HyperbolicTiling(p, q, n, kernel=TilingKernels.StaticRotationalGraph)

    # Extract coordinates function
    def get_xy(poly_id):
        xy = tiling_obj.get_center(poly_id)  # returns complex number
        return np.real(xy), np.imag(xy)

    layers_info = {i: tiling_obj.get_layer(i) for i in range(len(tiling_obj))}  # layer info from SRG

    # ---------------------------
    # 2️⃣ Directed polygon structure
    # ---------------------------
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

    def generate_poly_id_mapping(layers_info):
        return {pid: pid for pid in layers_info}

    def share_common_edge(poly_id1, poly_id2, q):
        nbrs1 = set(tiling_obj.get_nbrs(poly_id1))
        nbrs2 = set(tiling_obj.get_nbrs(poly_id2))
        common_nbrs = nbrs1.intersection(nbrs2)
        return len(common_nbrs) == 2*(q-2)

    def get_shared_edge_neighbors(poly_id):
        return [nbr for nbr in tiling_obj.get_nbrs(poly_id) if share_common_edge(poly_id, nbr, q)]

    def determine_directed_neighbors(poly_id, layers_info, q=5):
        dp = DirectedPolygon(poly_id)
        shared_edge_neighbors = get_shared_edge_neighbors(poly_id)
        current_layer = layers_info[poly_id]
        same_layer = [nbr for nbr in shared_edge_neighbors if layers_info[nbr]==current_layer]
        upper_layer = [nbr for nbr in shared_edge_neighbors if layers_info[nbr]<current_layer]
        lower_layer = [nbr for nbr in shared_edge_neighbors if layers_info[nbr]>current_layer]

        if upper_layer: dp.back = upper_layer[0]
        if lower_layer:
            if len(lower_layer)==1: dp.front = lower_layer[0]
            else: dp.all_front = lower_layer

        if same_layer:
            dp.left, dp.right = same_layer[0], same_layer[-1]
        return dp

    directed_polygons = {pid: determine_directed_neighbors(pid, layers_info, q) for pid in layers_info}
    poly_id_mapping = generate_poly_id_mapping(layers_info)

    # ---------------------------
    # 3️⃣ Tensor creation
    # ---------------------------
    def has_any_neighbor(poly_id):
        dp = directed_polygons[poly_id]
        return any([dp.back, dp.left, dp.right, dp.front, dp.left_front, dp.right_front, dp.all_front])

    def generate_tensor_with_legs(poly_id, tensor_list):
        if not has_any_neighbor(poly_id): return
        dp = directed_polygons[poly_id]
        tensor = Tensor(poly_id_mapping[poly_id], 0)
        if poly_id==0:
            for f in dp.all_front: tensor.add_leg(TensorLeg('I', (poly_id_mapping[f], None)))
        else:
            order = ['back','left','front','right']
            for dir_name in order:
                nbr = getattr(dp, dir_name, None)
                if nbr is not None: tensor.add_leg(TensorLeg('I', (poly_id_mapping[nbr], None)))
                else: tensor.add_leg(TensorLeg('I', None))
        tensor.layer = layers_info[poly_id]
        tensor_list.append(tensor)

    tensor_list = []
    for pid in layers_info: generate_tensor_with_legs(pid, tensor_list)

    # Update connections
    for tensor in tensor_list:
        for leg in tensor.legs:
            if leg.connection is None: continue
            nbr_tensor = get_tensor_from_id(tensor_list, leg.connection[0])
            if nbr_tensor is None: continue
            for idx, nbr_leg in enumerate(nbr_tensor.legs):
                if nbr_leg.connection is not None and nbr_leg.connection[0] == tensor.tensor_id:
                    leg.connection = (nbr_tensor.tensor_id, idx)
                    break

    # ---------------------------
    # 4️⃣ Build NetworkX graph
    # ---------------------------
    G = nx.Graph()
    for t in tensor_list: G.add_node(t.tensor_id, layer=t.layer)
    seen_edges = set()
    for t in tensor_list:
        for leg in t.legs:
            if leg.connection is None: continue
            u, v = t.tensor_id, leg.connection[0]
            edge = tuple(sorted((u,v)))
            if edge not in seen_edges:
                seen_edges.add(edge)
                G.add_edge(u,v)

    # attach coordinates
    for tensor_id in G.nodes:
        x, y = get_xy(tensor_id)
        G.nodes[tensor_id]['x'] = float(x)
        G.nodes[tensor_id]['y'] = float(y)


    # remove final layer issues
    from copy import deepcopy 
    Gnodes_OG = deepcopy(G.nodes())

    G_node_to_layer = {}
    for ii in Gnodes_OG:
        G_node_to_layer.update({ii:(G.nodes[ii]['layer'])})
    max_layer = max(G_node_to_layer.values())


    remove_nodes_lst = []
    for ii in Gnodes_OG:
        ii_layer = (G.nodes[ii]['layer']) 
        if ii_layer == max_layer:
            nei_lst = list(G.neighbors(ii))
            nei_layers = [G.nodes[jj]['layer'] for jj in nei_lst]

            nei_layers_arr = np.array(nei_layers)
            if np.all(nei_layers_arr == max_layer):
                remove_nodes_lst.append(ii)

    G.remove_nodes_from(remove_nodes_lst)

    for ii in G.nodes():
        ii_layer = (G.nodes[ii]['layer']) 
        if ii_layer == max_layer:
            nei_lst = list(G.neighbors(ii))
            nei_layers = [G.nodes[jj]['layer'] for jj in nei_lst]

            for ll in nei_lst:
                if G.nodes[ll]['layer'] == max_layer:
                    if G.has_edge(ii,ll):
                        G.remove_edge(ii,ll)

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

    # ---------------------------
    # 5️⃣ Export JSON
    # ---------------------------
    # import json
    export = {
        "nodes": {str(n): {"x": G.nodes[n]["x"], "y": G.nodes[n]["y"], "layer": G.nodes[n]["layer"]} for n in G.nodes},
        "edges": [{"source": u, "target": v} for u,v in G.edges],
        "meta": {"schlafli":"{5,4}", "layers": n, "geometry":"hyperbolic"}
    }
    # with open("lego_tensor_graph.json","w") as f: json.dump(export,f,indent=2)

    # print("LEGO tensor graph exported:", len(G.nodes), "nodes,", len(G.edges), "edges")

    # # old 
    def expand_node_along_edges_polygon_correct(G, v, pos, node_counter, t=0.3, extra_attrs=None):
        """
        Expand node v into one node per neighbor along edges.
        Expanded nodes are connected to parent AND form a polygon according to neighbor angles.
        """
        if extra_attrs is None:
            extra_attrs = {}

        neighbors = list(G.neighbors(v))
        expanded_nodes = []

        # Step 1: create expanded nodes along edges
        for u in neighbors:
            new_label = node_counter[0]
            node_counter[0] += 1

            x0, y0 = pos[v]
            x1, y1 = pos[u]
            node_pos = (x0*(1-t) + x1*t, y0*(1-t) + y1*t)

            node_attr = {'parent': v, 'type': 'expanded', 'original_edge': u}
            node_attr.update(extra_attrs)
            # G.add_node(new_label, pos=node_pos, **node_attr)
            G.add_node(new_label, pos=node_pos, x=node_pos[0],y=node_pos[1],layer = -1, **node_attr)
            pos[new_label] = node_pos
            expanded_nodes.append((new_label, u))  # keep track of corresponding neighbor

            # Replace original edge with edges through expanded node
            if G.has_edge(v, u):
                G.remove_edge(v, u)
            G.add_edge(v, new_label, parent=v)
            G.add_edge(new_label, u, parent=v)

        # Step 2: order expanded nodes counter-clockwise around parent
        x0, y0 = pos[v]
        def angle_to_parent(pair):
            node, u = pair
            x, y = pos[node]
            return np.arctan2(y - y0, x - x0)
        expanded_nodes.sort(key=angle_to_parent)
        sorted_nodes = [n for n, _ in expanded_nodes]

        # Step 3: connect expanded nodes in polygon according to counter-clockwise order
        n = len(sorted_nodes)
        for i in range(n):
            u, w = sorted_nodes[i], sorted_nodes[(i + 1) % n]
            G.add_edge(u, w, parent=v)
        

        return sorted_nodes, v

    def expand_graph_edges_polygon_correct(G, pos, skip_nodes=None, t=0.3, extra_attrs=None):
        """
        Expand all original nodes into edge-aligned polygons.
        Correctly orders expanded nodes counter-clockwise to maintain proper polygon edges.
        """
        if skip_nodes is None:
            skip_nodes = []
        if extra_attrs is None:
            extra_attrs = {}

        # Determine starting counter for new node IDs
        existing_ints = []
        for n in G.nodes():
            try:
                existing_ints.append(int(n))
            except ValueError:
                if '_' in str(n):
                    existing_ints.append(int(str(n).split('_')[0]))
        node_counter = [max(existing_ints, default=-1) + 1]

        original_nodes = [v for v in list(G.nodes()) if v not in skip_nodes]

        par_to_sorted = {}
        for v in original_nodes:
            if G.has_node(v):
                sorted_nodes, vv = expand_node_along_edges_polygon_correct(G, v, pos, node_counter, t, extra_attrs)
                par_to_sorted.update({vv:sorted_nodes})
        
        return par_to_sorted

    pos = {}
    for ii in G.nodes():
        pos.update({ii:(G.nodes[ii]['x'],G.nodes[ii]['y'])})

    max_level_nodes = []
    for ii in G.nodes():
        # if G.nodes[ii]['layer']==max_layer:
        if G.nodes[ii].get('layer', -1)==max_layer:
            max_level_nodes.append(ii)

    par_to_sorted = expand_graph_edges_polygon_correct(G, pos, skip_nodes=max_level_nodes, t=0.3)


    # # generate pyzx graph
    import pyzx as zx
    Gzx = zx.Graph()
    scale = 30
    for ii in G.nodes():
        # print(ii)
        if G.nodes[ii].get('layer', -1) == max_layer:
            vertexty = zx.VertexType.BOUNDARY
        else:
            vertexty = zx.VertexType.Z
        Gzx.add_vertex(index=ii,ty=vertexty,qubit=scale*G.nodes[ii]['x'],row=scale*G.nodes[ii]['y'])
    for ii in G.edges():
        if G.nodes[ii[0]].get('parent', -1) == G.nodes[ii[1]].get('parent', -1):
            Gzx.add_edge(ii,zx.EdgeType.HADAMARD)
        elif G.nodes[ii[0]].get('parent', -1) != -1 and ii[1] == G.nodes[ii[0]].get('parent', -1):
            Gzx.add_edge(ii,zx.EdgeType.HADAMARD)
        elif G.nodes[ii[1]].get('parent', -1) != -1 and ii[0] == G.nodes[ii[1]].get('parent', -1):
            Gzx.add_edge(ii,zx.EdgeType.HADAMARD)
        elif Gzx.type(ii[0]) != zx.VertexType.BOUNDARY and Gzx.type(ii[1]) != zx.VertexType.BOUNDARY:
            Gzx.add_edge(ii,zx.EdgeType.HADAMARD) # contracted legs 
            # Gzx.add_edge(ii,zx.EdgeType.SIMPLE)
        else:
            Gzx.add_edge(ii,zx.EdgeType.SIMPLE)

    boundary_nodes = []
    for ii in Gzx.vertices():
        if Gzx.type(ii) == zx.VertexType.BOUNDARY:
            boundary_nodes.append(ii)

    bulk_nodes = []
    for ii in G.nodes():
        if G.nodes[ii].get('type', -1) == -1 and Gzx.type(ii)!=zx.VertexType.BOUNDARY:
            nv = Gzx.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=Gzx.qubit(ii),row=Gzx.row(ii)+1)
            bulk_nodes.append(nv)
            Gzx.add_edge([nv,ii])

    Gzx.set_inputs(bulk_nodes)
    Gzx.set_outputs(boundary_nodes)

    # zx.draw(Gzx, labels=True, scale=10)   # interactive display only

    return Gzx, 1
