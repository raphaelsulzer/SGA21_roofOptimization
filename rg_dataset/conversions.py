import networkx as nx
import numpy as np

def get_graph_from_verts_and_faces_file(verts_path, faces_path):
    G = nx.Graph()
    verts = []
    with open(verts_path, 'r') as vf:
        for i, line in enumerate(vf):
            line = line.strip()
            if not line:
                continue
            x_str, y_str = line.split(',')
            x, y = float(x_str), float(y_str)
            verts.append((x, y))
            G.add_node(i, pos=(x, y))
    with open(faces_path, 'r') as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                indices = [int(idx) for idx in line.split(',')]
            else:
                indices = [int(idx) for idx in line.split()]
            for i in range(len(indices)):
                v1 = indices[i]
                v2 = indices[(i + 1) % len(indices)]
                G.add_edge(v1, v2)
    return G


def get_graph_from_file(graph_path):
    
    data = np.load(graph_path, allow_pickle=True)
    
    graph = nx.from_numpy_array(data['adjacency'])
    coords = data['coords']
    nx.set_node_attributes(graph, {i: {'pos': tuple(coords[i])} for i in range(len(coords))})
    
    return graph