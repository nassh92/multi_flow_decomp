import random

import os

import numpy as np

import pprint

import matplotlib.pyplot as plt

import sys

import math

sys.path.append(os.getcwd())
from instance_generation.geometric_utils import DISTANCE_TYPES, generate_nodes, get_near_nodes, distance, sort_candidate_neighbours

from instance_generation.time_costs_generator import generate_raw_times

from instance_generation.intersection_utils import intersect

from instance_generation.connexity import is_connected

from utils.graph_utils import get_neighbours, create_isolated_nodes_graph, add_arc



def generate_random_planar_graph(nb_nodes, 
                                 grid_size, 
                                 r, 
                                 nb_edges, 
                                 max_nb_draws, 
                                 max_nb_tries, 
                                 distance_type = "euclidean",
                                 max_nb_neighbours = None, 
                                 graph_representation = "adjacency_matrix",
                                 print_ = False):
    # Atempt to generate a planar random graph for a maximum of 'max_nb_tries' times
    nb_tries = 0
    while nb_tries < max_nb_tries:
        # Generate the nodes of the graph
        nodes = generate_nodes(grid_size, nb_nodes)
        
        # Construct an adjacency matrix whose entries are intialized to 0
        graph = create_isolated_nodes_graph(nb_nodes, graph_representation = graph_representation)

        # Generate the arcs of the planar random graph
        nb_draws, arcs_segs, points = 0, set(), list(nodes.keys())
        while len(arcs_segs) < nb_edges and nb_draws < max_nb_draws:
            # Affichage du numéro de tirage 'nb_draws'
            if print_ and nb_draws % 10 == 0:
                print("The try number / Number of nodes pairs drawn ", nb_tries, " / ",nb_draws)

            # Select a new candidate arc to be added in 'arcs'
            P = points[random.randint(0, len(points)-1)]
            candidate_neighbours = get_near_nodes(P, points, r, distance_type = distance_type)

            if len(candidate_neighbours) != 0:
                # Select the second point of the arc in case P has neighbours with distance 'r' 
                Q = candidate_neighbours[random.randint(0, len(candidate_neighbours)-1)]

                # Process the condition on the number of neighbours
                nb_neighbours_not_a_problem = True
                if max_nb_neighbours is not None:
                    nb_neighbours = max(len(get_neighbours(graph, nodes[P])), 
                                        len(get_neighbours(graph, nodes[Q])))
                    if nb_neighbours >= max_nb_neighbours:
                        nb_neighbours_not_a_problem = False

                # Add arc [P, Q] to 'arcs' if [P, Q] does not intersect with any of the arcs in 'arcs'
                if nb_neighbours_not_a_problem and (P, Q) not in arcs_segs and (Q, P) not in arcs_segs and not intersect((P, Q), arcs_segs):
                    arcs_segs.add((P, Q))
                    add_arc(graph, nodes[P], nodes[Q])
                    add_arc(graph, nodes[Q], nodes[P])

            nb_draws += 1
        
        # If the graph generated is connected return it as with the transport times and the segments else increment the numbers of tries
        if is_connected (graph):
            raw_transport_times = generate_raw_times (graph, nodes, arcs_segs, distance_type = "euclidean")
            if raw_transport_times: return graph, arcs_segs, nodes, raw_transport_times
        nb_tries += 1
    
    return False, False, False, False



def generate_random_small_world_like_planar_graph(nb_nodes,
                                                  grid_size, 
                                                  r, 
                                                  nb_edges, 
                                                  max_nb_draws, 
                                                  max_nb_tries,
                                                  max_nb_neighbours = None, 
                                                  distance_type = "euclidean",
                                                  sorting_criteria = "distance",
                                                  graph_representation = "adjacency_matrix",
                                                  print_ = False):
    # Atempt to generate a planar random graph for a maximum of 'max_nb_tries' times
    nb_tries = 0
    while nb_tries < max_nb_tries:
        # Generate the nodes of the graph
        nodes = generate_nodes(grid_size, nb_nodes)

        # Construct an adjacency matrix whose entries are intialized to 0
        graph = create_isolated_nodes_graph(nb_nodes, graph_representation = graph_representation)

        # Generate the arcs of the planar random graph
        nb_draws, arcs, points = 0, set(), list(nodes.keys())
        while len(arcs) < nb_arcs and nb_draws < max_nb_draws:
            # Affichage du numéro de tirage 'nb_draws'
            if print_ and nb_draws % 10 == 0:
                print("The try number / Number of nodes pairs drawn ", nb_tries, " / ",nb_draws)

            # Select a new candidate arc to be added in 'arcs'
            P = points[random.randint(0, len(points)-1)]
            candidate_neighbours = get_near_nodes(P, 
                                                  points, 
                                                  r, 
                                                  distance_type = distance_type)
            opt_params = {"adj_mat":graph, "nodes":nodes}
            candidate_neighbours = sort_candidate_neighbours(P, 
                                                             candidate_neighbours, 
                                                             distance_type = distance_type, 
                                                             sorting_criteria = sorting_criteria,
                                                             opt_params = opt_params)

            if len(candidate_neighbours) != 0:
                # Select the second point of the arc in case P has neighbours with distance 'r'
                # Take the closest avaible point
                for ix in range(len(candidate_neighbours)):
                    # Select the the candidate neighbour of index 'ix' 
                    Q = candidate_neighbours[ix]
                    
                    # Process the condition on the number of neighbours
                    nb_neighbours_not_a_problem = True
                    if max_nb_neighbours is not None:
                        nb_neighbours = max(len(get_neighbours(adj_mat, nodes[P])), 
                                            len(get_neighbours(adj_mat, nodes[Q])))
                        if nb_neighbours >= max_nb_neighbours:
                            nb_neighbours_not_a_problem = False

                    # Add arc [P, Q] to 'arcs' if [P, Q] does not intersect with any of the arcs in 'arcs'
                    if nb_neighbours_not_a_problem and (P, Q) not in arcs and (Q, P) not in arcs and not intersect((P, Q), arcs):
                        arcs.add((P, Q))
                        adj_mat[nodes[P]][nodes[Q]] = 1
                        adj_mat[nodes[Q]][nodes[P]] = 1
                        break

            nb_draws += 1
        
        # If the graph generated is connected return it as with the transport times and the segments else increment the numbers of tries
        if is_connected (adj_mat):
            raw_transport_times = generate_raw_times (nb_nodes, nodes, arcs, distance_type = "euclidean") 
            return adj_mat, arcs, nodes, raw_transport_times
        nb_tries += 1
    
    return False, False, False, False



if __name__ == "__main__":
    display = True
    adj_matrice, arcs, nodes, raw_transport_times = generate_random_planar_graph(nb_nodes=6, 
                                                                                grid_size=100, 
                                                                                r=100, 
                                                                                nb_arcs=9, 
                                                                                max_nb_draws=1000, 
                                                                                max_nb_tries=3, 
                                                                                distance_type = "euclidean")
        
    print("Adjacency matrix ")
    pprint.pprint(adj_matrice)

    print("Raw transport times ")
    pprint.pprint(raw_transport_times)

    if display:
        plt.figure()
        plt.title("Test arcs ", fontsize=15)
        for arc in arcs:
            plt.xlabel('x-axis',fontsize=15)
            plt.ylabel('y-axis',fontsize=15)
            p1, p2 = arc
            center = ((p1[0]+p2[0])/2 , (p1[1]+p2[1])/2)
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
            plt.annotate(str(nodes[p1]), xy=p1)
            plt.annotate(str(nodes[p2]), xy=p2)
            plt.annotate(str(round(raw_transport_times[nodes[p1]][nodes[p2]], 2)), xy=center)
            
        plt.grid()
        plt.show()