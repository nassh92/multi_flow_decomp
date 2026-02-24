import sys

import os

import math

sys.path.append(os.getcwd())

from instance_generation.geometric_utils import distance


def generate_raw_times (nb_nodes, nodes, arcs, distance_type = "euclidean"):
    # Initializaitons
    raw_transport_times = [[0 if i==j else float("inf") for j in range(nb_nodes)] for i in range(nb_nodes)]
    # For each segment in 'arc', the transport tiime is represented as the distance separating both endpoints of the segment
    for seg in arcs:
        node1, node2 = nodes[seg[0]], nodes[seg[1]]
        cost = distance(seg[0], seg[1], distance_type = distance_type)
        if math.isinf(cost) or math.isnan(cost):
            print("Distance is infinity ", seg[0], seg[1])
            return False
        raw_transport_times[node1][node2] = cost
        raw_transport_times[node2][node1] = cost
    return raw_transport_times


def generate_transport_time(raw_transport_times, spanning_tree_pred, fraction):
    # Initializaitons
    transport_times = [[0 if i==j else float("inf") for j in range(len(raw_transport_times))] for i in range(len(raw_transport_times))]
    # Main Loop. The 'transport_time' is a fraction of the raw transport time if it is in the spanning_tree
    # else it does not change
    for i in range(len(transport_times)):
        for j in range(len(transport_times)):
            if spanning_tree_pred[j] == i or spanning_tree_pred[i] == j:
                transport_times[i][j] = fraction * raw_transport_times[i][j]
            else:
                transport_times[i][j] = raw_transport_times[i][j]
    return transport_times