import pickle
import urllib.parse
import os
import sys
import networkx as nx
import osmnx as ox


import json


import pandas as pd

import unidecode

from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge, substring

import pickle
import sys


sys.path.append(os.getcwd())


def save_list_duplicate_arcs(path_file):
    with open(path_file,'rb') as f:
        g = pickle.load(f)
    
    for u, v, k, _ in g.edges(keys=True, data=True):
        if k > 0:
            print("----------------------------------------------------------------------------")
            print("Node u ")
            print(u)
            print("Node v ")
            print(v)
            print("Printing g[u][v] ")
            print(g[u][v])


def show_duplicate_arcs_subgraph(path_file, g = None):
    """
    Generate the subgraph 
    """
    # Open the file
    if g is None:
        with open(path_file,'rb') as f:
            g = pickle.load(f)
    
    set_id_nodes, list_edges, edge_colors = set(), list(), []
    for u, v, k, data in g.edges(keys = True, data = True):
        # Add first node 'u'
        set_id_nodes.add(u)
        # Add second node 'v'
        set_id_nodes.add(v)
        # Set the edge color
        if len(g[u][v]) > 1:
            if u == v:
                edge_color = "green"
            elif k==0: 
                edge_color = "blue"
            elif k==1:
                edge_color = "red"
            else:
                print("Edge multiplicity higher than expected.")
                #sys.exit()
        else:
            edge_color = "black"
        edge_colors.append(edge_color)
        list_edges.append((u, v, k, data))


    sub_g = nx.MultiDiGraph()
    sub_g.graph['crs'] = g.graph['crs'] 
    sub_g.add_nodes_from([node for node in g.nodes(data=True) if node[0] in set_id_nodes])
    sub_g.add_edges_from(list_edges)

    
    fig, ax = ox.plot_graph(
        sub_g,
        figsize=(60, 60),
        node_color="yellow",
        node_size=5,
        edge_color=edge_colors,
        edge_linewidth=2,
        edge_alpha=0.4,   # transparency (0 = fully transparent, 1 = opaque)
        node_alpha=0.4,
        bgcolor="white",
        show=True,
        close=True
    )


if __name__ == "__main__":
    test_names = {0, 1, 2}
    test_name = 1
    if test_name == 0:
        path_file = "data/original_graphs/versailles.gpickle"
        save_list_duplicate_arcs(path_file)
    
    elif test_name == 1:
        path_file = "data/original_graphs/lieusaint.gpickle"
        show_duplicate_arcs_subgraph(path_file)
    
    elif test_name == 2:
        path_file = "data/original_graphs/lieusaint.gpickle"
        with open(path_file,'rb') as f:
            g = pickle.load(f)
        
        graph_lieu_saint = nx.MultiDiGraph()
        graph_lieu_saint.graph['crs'] = g.graph['crs']
        path_file = "data/original_graphs/localization_data.txt"
        with open(path_file, "r") as file:
            for line in file:
                stripped_line = line.rstrip().split(' ')
                n, y, x = int(stripped_line[0]), float(stripped_line[1]), float(stripped_line[2])
                graph_lieu_saint.add_node(n, x = x, y = y)
        
        path_file = "multi_flow_generation_wei/data/graph_matrice.pypgr"
        with open(path_file, "r") as file:
            for line in file:
                stripped_line = line.rstrip().split(' ')
                graph_lieu_saint.add_edge(int(stripped_line[0]), int(stripped_line[1]))
        show_duplicate_arcs_subgraph(None, graph_lieu_saint)
