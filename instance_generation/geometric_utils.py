import random

import itertools

import sys

import os

from copy import deepcopy

import numpy as np

sys.path.append(os.getcwd())

from utils.graph_utils import get_neighbours


DISTANCE_TYPES = {"euclidean"}


NEIGHBOURS_SORTING_CRITERIAS = {"distance",
                                "distance_n_nb_neighbours", 
                                "nb_neighbours_n_distance"}


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


def get_near_nodes(P, points, r, distance_type = "euclidean", sort_distances = False):
    """
    Get the points at distance r from point P. 
    P : the size of the grid 
    points : List of nodes
    r : real number
    The algorithm returns a dictionary where the key is the coordinates of the nodes and the value is the number identifier of the node 
    (in the adjacency matrix)
    """
    near_points = []
    if sort_distances: distances_to_P = []
    for i in range(len(points)):
        distance_to_point = distance(P, points[i], distance_type = distance_type) 
        if points[i] != P and distance_to_point <= r:
            near_points.append(points[i])
            if sort_distances: distances_to_P.append(distance_to_point)
    # Sort the distances if 'sort_distances' is True 
    if sort_distances:
        sorted_points = sorted(list(zip(near_points, distances_to_P)),
                               key = lambda x : x[1])
        near_points = [e[0] for e in sorted_points]
    return near_points


def sort_candidate_neighbours(P, 
                              near_points, 
                              distance_type = "euclidean", 
                              sorting_criteria = "distance",
                              opt_params = None):
    if sorting_criteria == "distance":
        sorted_points = sorted(near_points, 
                               key = lambda point : distance(P, 
                                                             point, 
                                                             distance_type = distance_type))
    
    elif sorting_criteria == "distance_n_nb_neighbours":
        adj_mat, nodes = opt_params["adj_mat"], opt_params["nodes"]
        sorted_points = sorted(near_points, 
                               key = lambda point : (distance(P, 
                                                             point, 
                                                             distance_type = distance_type),
                                                    len(get_neighbours(adj_mat, nodes[point]))
                                                    ))
        
    elif sorting_criteria == "nb_neighbours_n_distance":
        adj_mat, nodes = opt_params["adj_mat"], opt_params["nodes"]
        sorted_points = sorted(near_points, 
                               key = lambda point : (len(get_neighbours(adj_mat, nodes[point])),
                                                     distance(P, 
                                                             point, 
                                                             distance_type = distance_type),
                                                    ))
    
    else:
        print("Node sorting criteria unrocognized.")
        sys.exit()
    return sorted_points


def distance(P, Q, distance_type = "euclidean"):
    """
    Get the points at distance r from point P. 
    P, Q : Points in the grid
    """
    if distance_type not in DISTANCE_TYPES:
        print("Unrecognized distance type ", distance_type)
        sys.exit()
    
    if distance_type == "euclidean": return np.linalg.norm(np.array(P) - np.array(Q))
        