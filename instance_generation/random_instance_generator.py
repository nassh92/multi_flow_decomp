import numpy as np

import pprint

import os

import sys

import statistics

import itertools

import math

sys.path.append(os.getcwd())
from instance_generation.random_planar_graph_generator import generate_random_planar_graph, generate_random_small_world_like_planar_graph

from instance_generation.capacity_generator import generate_capacities

from instance_generation.time_costs_generator import generate_transport_time

from instance_generation.pairs_utils import (generate_random_origin_destination_pairs, 
                                             generate_origin_destination_pairs_local, 
                                             generate_origin_destination_pairs_global, 
                                             process_weight_pairs, 
                                             PAIRS_GENERATION_TYPES)

from instance_generation.display_utils_gen import display_instance

from instance_generation.real_instances_wei_based.saint_lieu_instance_generator import generate_instance_saint_lieu



############################################################## RANDOM INSTANCE GENERATOR ########################################################
def generate_random_instance(nb_nodes, grid_size, r, nb_arcs, max_nb_draws_gen_graph, 
                             max_nb_tries_gen_graph, base_capacity, 
                             capacity_factor, transport_time_fraction, 
                             nb_pairs, nb_max_draws_pairs,
                             sorting_criteria = "distance",
                             max_nb_neighbours = None, small_world_like = False,
                             distance_type = "euclidean", pairs_generation = "degree", 
                             pairs_criteria = 0, print_ = False):
    # Generate a random planar graph
    if not small_world_like:
        adj_mat, arcs, nodes, raw_transport_times = generate_random_planar_graph(nb_nodes = nb_nodes,
                                                                                    grid_size = grid_size, 
                                                                                    r = r, 
                                                                                    nb_arcs = nb_arcs, 
                                                                                    max_nb_draws = max_nb_draws_gen_graph, 
                                                                                    max_nb_tries = max_nb_tries_gen_graph,
                                                                                    max_nb_neighbours = max_nb_neighbours, 
                                                                                    distance_type = distance_type,
                                                                                    print_ = print_)
    else:
        adj_mat, arcs, nodes, raw_transport_times = generate_random_small_world_like_planar_graph(nb_nodes = nb_nodes,
                                                                                    grid_size = grid_size, 
                                                                                    r = r, 
                                                                                    nb_arcs = nb_arcs, 
                                                                                    max_nb_draws = max_nb_draws_gen_graph, 
                                                                                    max_nb_tries = max_nb_tries_gen_graph,
                                                                                    max_nb_neighbours = max_nb_neighbours,
                                                                                    sorting_criteria = sorting_criteria, 
                                                                                    distance_type = distance_type,
                                                                                    print_ = print_)

    if isinstance(adj_mat, bool):
        print("Random planar graph generation failed.")
        sys.exit()
    if pairs_generation not in PAIRS_GENERATION_TYPES:
        print("Pairs generatio type is not recognized.")
        sys.exit()
    
    # Generate the capacities
    capacities, span_tree_pred = generate_capacities (adj_mat = adj_mat, 
                                                        base_capacity = base_capacity, 
                                                        factor = capacity_factor, 
                                                        print_ = print_)

    # Generate the transport times
    transport_times = generate_transport_time(adj_mat,
                                              raw_transport_times, 
                                              spanning_tree_pred = span_tree_pred, 
                                              fraction = transport_time_fraction)

    # Generate the pairs and the weights following the value of 'pairs_generation'
    if pairs_generation == "random":
        # Generate the weights and the pairs
        pairs = generate_random_origin_destination_pairs (nb_pairs, adj_mat, nb_max_draws = nb_max_draws_pairs)
        weight_pairs = process_weight_pairs(pairs, adj_mat, pairs_selection = pairs_generation)
    
    elif pairs_generation == "degree" or pairs_generation == "capacity":
        # Calculate the matrix which will serve to generate the pairs
        arc_attribute_vals = adj_mat if pairs_generation == "degree" else capacities if pairs_generation == "capacity" else None
        # Generate the weights and the pairs
        pairs = generate_origin_destination_pairs_local (nb_pairs, arc_attribute_vals, nb_max_draws = nb_max_draws_pairs)
        weight_pairs = process_weight_pairs(pairs, 
                                            adj_mat,
                                            arc_attribute_vals = arc_attribute_vals, 
                                            pairs_selection = pairs_generation) # !! optional !!
    
    elif pairs_generation == "min_cut":
        # Generate the pairs and the weights
        pairs = generate_origin_destination_pairs_global (nb_pairs, adj_mat, capacities, nb_max_draws = nb_max_draws_pairs,
                                                          transport_times = transport_times)
        weight_pairs = process_weight_pairs(pairs, 
                                            [adj_mat, capacities, transport_times], 
                                            pairs_selection = pairs_generation) # optional

    elif pairs_generation == "all":
        # Generate the pairs and the weights
        if pairs_criteria == 0:
            pairs = generate_origin_destination_pairs_local (nb_pairs, capacities, nb_max_draws = nb_max_draws_pairs)
        elif pairs_criteria == 1:
            pairs = generate_origin_destination_pairs_global (nb_pairs, adj_mat, capacities, nb_max_draws = nb_max_draws_pairs,
                                                              transport_times = transport_times)
        
        # optional
        weight_pairs_degree = process_weight_pairs(pairs, adj_mat, pairs_selection = "degree")
        weight_pairs_capacity = process_weight_pairs(pairs, 
                                                     adj_mat,
                                                     arc_attribute_vals = capacities, 
                                                     pairs_selection = "capacity")
        weight_pairs_mincut = process_weight_pairs(pairs, [adj_mat, capacities,  transport_times], pairs_selection = "min_cut")
    else:
        print("Pairs generation not recognized.")
        sys.exit()
    
    
    # Construct and return the dict containing all the informations related to the generated instance (follwing the value of 'pairs_generation') 
    return_dict = {"adj_mat":adj_mat,
                   "arcs":arcs,
                   "nodes":nodes,
                   "capacities":capacities,
                   "transport_times":transport_times,
                   "pairs":pairs,
                   # `weights_pairs` below is optional, it will be generated further down the line during multiflow generation
                   "weight_pairs":weight_pairs if pairs_generation != "all" else [weight_pairs_degree, weight_pairs_capacity, weight_pairs_mincut]}
    
    return return_dict


def generate_instances(dir_name, nb_instances, nb_nodes, grid_size, r, nb_arcs, max_nb_draws_gen_graph, max_nb_tries_gen_graph,
                       base_capacity, capacity_factor, transport_time_fraction, nb_pairs, nb_max_draws_pairs,
                       max_nb_neighbours = None, sorting_criteria = "distance",
                       distance_type = "euclidean", pairs_generation = "degree", 
                       pairs_criteria = 0, print_ = False,
                       small_world_like = False):
        for num_instance in range(nb_instances):
            print("Generate instance ", num_instance)
            # Return infos of the generated intance
            infos_instance = generate_random_instance(nb_nodes, grid_size, r, nb_arcs, 
                                                    max_nb_draws_gen_graph, max_nb_tries_gen_graph,
                                                    base_capacity, capacity_factor, 
                                                    transport_time_fraction, nb_pairs, nb_max_draws_pairs, 
                                                    max_nb_neighbours = max_nb_neighbours,
                                                    distance_type = distance_type, 
                                                    sorting_criteria = sorting_criteria,
                                                    pairs_generation = pairs_generation, 
                                                    pairs_criteria = pairs_criteria,
                                                    small_world_like = small_world_like, 
                                                    print_ = print_)
            # Save the instance
            np.save(os.path.join(dir_name, "instance_"+str(num_instance)), infos_instance)


def read_data(file_path, read_all = False):
     # Read the file containing the instance
     instance = np.load(file_path, allow_pickle = True).flat[0]
     # Unpack and return
     if not read_all:
        adj_mat = instance["adj_mat"]
        capacities = instance["capacities"]
        transport_times = instance["transport_times"]
        pairs = instance["pairs"]
        weight_pairs = instance["weight_pairs"]
        return adj_mat, capacities, transport_times, pairs, weight_pairs
     else:
        return instance

"""
Exemple de test pour la génération d'instances : 
generate_instances(dir_name = "instance_generation/instances/", 
                        nb_instances = 10, 
                        nb_nodes = 6, 
                        grid_size = 100, 
                        r = 100, 
                        nb_arcs = 9, 
                        max_nb_draws_gen_graph = 1000, 
                        max_nb_tries_gen_graph = 3,
                        base_capacity = 5, 
                        capacity_factor = 5, 
                        transport_time_fraction = 0.5, 
                        nb_pairs = 2, 
                        nb_max_draws_pairs = 20,
                        distance_type = "euclidean", 
                        print_ = False)
"""

def main():
    test_names = {"generate_instances",
                  "multiple_generate_instances", 
                  "read_instance", 
                  "generate_weights_real_instance"}

    test_name = "generate_instances"

    if test_name not in test_names:
        print("Test name unrecognized.")
        sys.exit()

    if test_name == "generate_instances":
        nb_nodes = 200
        grid_size = 100
        nb_pairs = 20
        calculate_max_nb_edges = lambda x : 3*x-6
        desired_mean_nb_neighbours = 4
        desired_nb_edges = math.ceil((desired_mean_nb_neighbours*nb_nodes)/2)
        generate_instances(dir_name = "data/simulated_data/graph_instances/small_world_neigh_dist_capacity_factor=15_basecapacity=50_time=4/instances_nbnodes=200_pairs=20/", 
                        nb_instances = 100, 
                        nb_nodes = nb_nodes, 
                        grid_size = grid_size, 
                        r = int(grid_size/2), 
                        nb_arcs = min(calculate_max_nb_edges(nb_nodes), desired_nb_edges), 
                        max_nb_draws_gen_graph = 10*calculate_max_nb_edges(nb_nodes), 
                        max_nb_tries_gen_graph = 10,
                        base_capacity = 50, 
                        capacity_factor = 20, 
                        transport_time_fraction = 1/4, 
                        nb_pairs = 20, 
                        nb_max_draws_pairs = nb_pairs*10,
                        max_nb_neighbours = 6,
                        distance_type = "euclidean",
                        sorting_criteria = "nb_neighbours_n_distance",
                        pairs_generation = "capacity",
                        pairs_criteria = 0,
                        print_ = False,
                        small_world_like = True)
    
    elif test_name == "multiple_generate_instances":
        nb_nodes_ls = [55, 75, 95]
        nb_pairs_ls = [10, 15, 20]
        params_names = list(itertools.product(nb_nodes_ls, nb_pairs_ls))
        dir_save_names = ["instances_nbnodes="+str(params_names[i][0])+"_pairs="+str(params_names[i][1])
                                    for i in range(len(params_names))]
        for i in range(len(params_names)):
            if dir_save_names[i] != "instances_nbnodes=75_pairs=15":
                nb_nodes, nb_pairs = params_names[i]
                print("--------------------------------------------------------------------------")
                print(nb_nodes, nb_pairs)
                print("--------------------------------------------------------------------------")
                dir_path = "instance_generation/instances/capacity/"+dir_save_names[i]+"/"
                generate_instances(dir_name = dir_path, 
                                    nb_instances = 100, 
                                    nb_nodes = nb_nodes, 
                                    grid_size = 100, 
                                    r = 25, 
                                    nb_arcs = 250, 
                                    max_nb_draws_gen_graph = 3000, 
                                    max_nb_tries_gen_graph = 5,
                                    base_capacity = 10, 
                                    capacity_factor = 3, 
                                    transport_time_fraction = 0.33, 
                                    nb_pairs = nb_pairs, 
                                    nb_max_draws_pairs = 300,
                                    distance_type = "euclidean",
                                    pairs_generation = "capacity",
                                    pairs_criteria = 0,
                                    print_ = False)
     
    elif test_name == "read_instance":
        file_path = "instance_generation/instances/capacity/instance_0.npy"
        instance = read_data(file_path, read_all = True)
        display_instance (adj_matrice = instance["adj_mat"], 
                          arcs = instance["arcs"],
                          nodes = instance["nodes"], 
                          capacities = instance["capacities"], 
                          transport_times = instance["transport_times"],
                          pairs = instance["pairs"],
                          weight_pairs = instance["weight_pairs"],
                          print_ = False)
    
    elif test_name == "generate_weights_real_instance":
        dir_name = "instance_generation/real_instance/"
        adj_mat, capacities, transport_times, pairs, _ = generate_instance_saint_lieu()
        degres_sortant = [sum(adj_mat[node][:]) for node in range(len(adj_mat))]
        print("Degrès moyen/std_dev ", statistics.mean(degres_sortant), statistics.stdev(degres_sortant))
        weight_pairs = process_weight_pairs(pairs, adj_mat, pairs_generation = "degree")
        #weight_pairs = process_weight_pairs(pairs, capacities, pairs_generation = "capacity")
        #weight_pairs = process_weight_pairs(pairs, [adj_mat, capacities, transport_times], pairs_generation = "min_cut")
        print("Weights of drawn pairs ")
        print(weight_pairs)
        np.save(os.path.join(dir_name, "weights_real_instance"), weight_pairs)
        weight_pairs = list(np.load(os.path.join(dir_name, "weights_real_instance.npy"), allow_pickle = True))
        print("Weights of pairs from file ")
        print(weight_pairs)



if __name__ == "__main__":
     main()