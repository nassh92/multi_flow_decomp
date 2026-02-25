import sys

import os

import math

sys.path.append(os.getcwd())

from instance_generation.geometric_utils import distance

from utils.graph_utils import init_graph_arc_attribute_vals, get_nb_nodes


def generate_raw_times (graph, nodes, arcs_segs, distance_type = "euclidean"):
    # Initializaitons
    init_val = lambda u, v: 0 if u==v else float("inf")
    raw_transport_times = init_graph_arc_attribute_vals(graph,
                                                        init_val = init_val)
    
    # For each segment in 'arc', the transport tiime is represented as the distance separating both endpoints of the segment
    for seg in arcs_segs:
        node1, node2 = nodes[seg[0]], nodes[seg[1]]
        cost = distance(seg[0], seg[1], distance_type = distance_type)
        if math.isinf(cost) or math.isnan(cost):
            print("Distance is infinity ", seg[0], seg[1])
            return False
        raw_transport_times[node1][node2] = cost
        raw_transport_times[node2][node1] = cost
    return raw_transport_times


def generate_transport_time(graph, raw_transport_times, spanning_tree_pred, fraction):
    # Initializaitons
    init_val = lambda u, v: 0 if u==v else float("inf")
    transport_times = init_graph_arc_attribute_vals(graph,
                                                    init_val = init_val)
    # Main Loop. The 'transport_time' is a fraction of the raw transport time if it is in the spanning_tree
    # else it does not change
    for i in range(len(transport_times)):
        for j in range(len(transport_times)):
            if spanning_tree_pred[j] == i or spanning_tree_pred[i] == j:
                transport_times[i][j] = fraction * raw_transport_times[i][j]
            else:
                transport_times[i][j] = raw_transport_times[i][j]
    return transport_times