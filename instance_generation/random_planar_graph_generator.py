import random

import os

import numpy as np

import pprint

import matplotlib.pyplot as plt

import sys

sys.path.append(os.getcwd())
from instance_generation.geometric_utils import DISTANCE_TYPES, generate_nodes, get_near_nodes, distance

from instance_generation.time_costs_generator import generate_raw_times

from instance_generation.intersection_utils import intersect

from instance_generation.connexity import is_connected



def generate_random_planar_graph(nb_nodes, grid_size, r, nb_arcs, max_nb_draws, max_nb_tries, distance_type = "euclidean", print_ = False):
    # Construct an adjacency matrix whose entries are intialized to 0
    adj_mat = [[0 for i in range(nb_nodes)] for j in range(nb_nodes)]

    # Atempt to generate a planar random graph for a maximum of 'max_nb_tries' times
    nb_tries = 0
    while nb_tries < max_nb_tries:
        # Generate the nodes of the graph
        nodes = generate_nodes(grid_size, nb_nodes)

        # Generate the arcs of the planar random graph
        nb_draws, arcs, points = 0, set(), list(nodes.keys())
        while len(arcs) < nb_arcs and nb_draws < max_nb_draws:
            # Affichage du numéro de tirage 'nb_draws'
            if print_ and nb_draws % 10 == 0:
                print("The try number / Number of nodes pairs drawn ", nb_tries, " / ",nb_draws)

            # Select a new candidate arc to be added in 'arcs'
            P = points[random.randint(0, len(points)-1)]
            candidate_neighbours = get_near_nodes(P, points, r, distance_type = distance_type)

            if len(candidate_neighbours) != 0:
                # Select the second point of the arc in case P has neighbours with distance 'r' 
                Q = candidate_neighbours[random.randint(0, len(candidate_neighbours)-1)]

                # Add arc [P, Q] to 'arcs' if [P, Q] does not intersect with any of the arcs in 'arcs'
                if (P, Q) not in arcs and (Q, P) not in arcs and not intersect((P, Q), arcs):
                    arcs.add((P, Q))
                    adj_mat[nodes[P]][nodes[Q]] = 1
                    adj_mat[nodes[Q]][nodes[P]] = 1

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