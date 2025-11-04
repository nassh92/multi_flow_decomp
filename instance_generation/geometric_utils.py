import random

import itertools

import sys

from copy import deepcopy

import numpy as np


DISTANCE_TYPES = {"euclidean"}


def generate_nodes(grid_size, nb_nodes):
    """
    Generate the nodes of the graph. 
    grid_size : the size of the grid 
    nb_nodes : Number of nodes to generate
    The algorithm returns a dictionary where the key is the coordinates of the nodes and the value is the number identifier of the node 
    (in the adjacency matrix)
    """
    nodes = {}
    grid_points = list(itertools.product(range(0, grid_size), 
                                         range(0, grid_size)))
    # Chose randomly 'nb_nodes' points from a grid of size 'grid_size' 
    for num_node in range(nb_nodes):
        id_chosen_point = random.randint(0, len(grid_points)-1)
        chosen_point = deepcopy(grid_points[id_chosen_point])
        del grid_points[id_chosen_point]
        nodes[chosen_point] = num_node

    return nodes


def get_near_nodes(P, points, r, distance_type = "euclidean"):
    """
    Get the points at distance r from point P. 
    P : the size of the grid 
    points : List of nodes
    r : real number
    The algorithm returns a dictionary where the key is the coordinates of the nodes and the value is the number identifier of the node 
    (in the adjacency matrix)
    """
    near_points = []
    for i in range(len(points)):
        if points[i] != P and distance(P, points[i], distance_type = distance_type) <= r:
            near_points.append(points[i])
    return near_points


def distance(P, Q, distance_type = "euclidean"):
    """
    Get the points at distance r from point P. 
    P, Q : Points in the grid
    """
    if distance_type not in DISTANCE_TYPES:
        print("Unrecognized distance type ", distance_type)
        sys.exit()
    
    if distance_type == "euclidean": return np.linalg.norm(np.array(P) - np.array(Q))
        