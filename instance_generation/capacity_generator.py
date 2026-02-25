import pprint

import random

import sys

import os

sys.path.append(os.getcwd())
from utils.graph_utils import get_nodes, successors, get_arcs, init_graph_arc_attribute_vals


def generate_capacities (graph, base_capacity, factor, print_ = False):
    # Process center
    center = process_center(graph, print_ = print_)
    if print_: print("Center ", center)

    # Initializations
    visited = [False]*len(graph)
    pred = [None]*len(graph)
    
    # Queue initialization
    visited[center] = True
    queue = [center]

    # Main loop
    while len(queue) > 0:
        # Dequeue a node
        node = queue.pop(0)
        
        # # Return the successor of 'node'
        successors_node = successors(node)

        # Browse through the successors of 'node' and if unvisited, update their distance from the source and their visited status
        for succ in successors_node:
            if not visited[succ]:
                visited[succ] = True
                pred[succ] = node 
                queue.append(succ)

    # Initialize the capacities
    dict_attr_params = {"base_capacity":base_capacity,
                        "factor":factor,
                        "pred":pred}
    init_val = lambda u, v, dict_attr_params: dict_attr_params["base_capacity"] * dict_attr_params["factor"] if dict_attr_params["pred"][v]==u or dict_attr_params["pred"][u]==v else dict_attr_params["base_capacity"]
    caps = init_graph_arc_attribute_vals(graph,
                                         init_val = init_val,
                                         dict_attr_params = dict_attr_params)
    
    if print_: print ("Predecessors ", pred)

    return caps, pred


def process_center(adj_mat, print_ = False):
    min_exentricity, centers = float("inf"), []
    # Main loop
    nodes = get_nodes(adj_mat)
    for node in nodes:
        exentricity = process_exentricity (adj_mat, node)
        # Update the min
        if exentricity < min_exentricity: 
            centers = [node] 
            min_exentricity = exentricity
        elif exentricity == min_exentricity:
            centers.append(node)
    # Print
    if print_:
        print("All Centers ", centers)
        return centers[0]
    else:
        return centers[random.randint(0, len(centers)-1)]


def process_exentricity (graph, source):
    # Initializations
    visited = [False]*len(graph)
    distance = [0]*len(graph)

    # Queue initialization
    visited[source] = True
    queue = [source]

    # Main loop
    while len(queue) > 0:
        # Dequeue a node
        node = queue.pop(0)

        # Return the successor of 'node'
        sucessors_node = successors(graph, node)

        # Browse through the successors of 'node' and if unvisited, update their distance from the source and their visited status
        for succ in sucessors_node:
            if not visited[succ]:
                visited[succ] = True
                distance[succ] = distance[node] + 1
                queue.append(succ)

    return distance[node]


if __name__ == "__main__":
    # Lien graphe : https://www.infoforall.fr/act/snt/graphes-et-liaisons-sociales/

    mat = [[0, 1, 1, 1, 0, 0, 0],
           [1, 0, 1, 0, 0, 1, 0],
           [1, 1, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 1, 0, 1, 1],
           [0, 1, 0, 1, 1, 0, 1],
           [0, 0, 0, 0, 1, 1, 0]]
    
    caps= generate_capacities (adj_mat = mat, 
                                base_capacity = 4, 
                                factor = 2, 
                                print_ = True)
    
    pprint.pprint(caps)
    

    
