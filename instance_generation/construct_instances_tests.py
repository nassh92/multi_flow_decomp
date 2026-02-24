import os

import sys

import numpy as np

import math

sys.path.append(os.getcwd())
from msmd.multi_flow_desag_instance_utils import construct_instances
from instance_generation.random_instance_generator import generate_random_instance
from instance_generation.multi_flow_instances_generator import generate_multi_flow_instance


def construct_instances_from_files():
    
        """ls_vals = [(55, 10), (55, 15), (55, 20),
        (75, 10), (75, 20),
        (95, 10), (95, 15), (95, 20)]"""
        ls_vals = [(200, 20)]
        for nb_nodes, pairs in ls_vals:

                print("--------------------------------------------------------------")
                print(nb_nodes, pairs)
                print("--------------------------------------------------------------")
                #dir_name_graph_instance = "instance_generation/instances/capacity/"
                dir_name_graph_instance = "data/simulated_data/graph_instances/smal_world_neigh_dist_capacity/"
                dir_name_graph_instance += "instances_nbnodes="+str(nb_nodes)+"_pairs="+str(pairs)+"/"
                #dir_name_multi_flow_instance = "multi_flow_generation/transition_function_instances/"
                dir_name_multi_flow_instance = "data/simulated_data/complete_instances/multi_flow_instances/small_world_neigh_dist_capacity/"
                dir_name_multi_flow_instance += "nb_nodes="+str(nb_nodes)+"_pairs="+str(pairs)+"/"
                dict_instances = construct_instances (
                                                dir_name_graph_instance = dir_name_graph_instance, 
                                                dir_name_multi_flow_instance = dir_name_multi_flow_instance,
                                                nb_instances = 100,
                                                ls_update_transport_time = [True],
                                                ls_update_transition_functions = [True])
                np.save("data/simulated_data/complete_instances/node_pairs/data_instances_small_world_"+str(nb_nodes)+"_"+str(pairs), 
                        dict_instances)



def construct_complete_instances(dict_common_parameters, 
                                 ls_param_vals,
                                 desired_flow_values = None,
                                 min_fl = 1,
                                 pairs_selection = "capacity",
                                 ls_dir_names = [None, None, None]):
        # Fetch the common parametes from the dict 'dict_common_parameters'
        nb_instances = dict_common_parameters["nb_instances"]
        grid_size = dict_common_parameters["grid_size"]
        radius_proportion = dict_common_parameters["radius_proportion"]
        factor_nb_draws_gen_graph = dict_common_parameters["factor_nb_draws_gen_graph"]
        max_nb_tries_gen_graph = dict_common_parameters["max_nb_tries_gen_graph"]
        base_capacity = dict_common_parameters["base_capacity"]
        capacity_factor = dict_common_parameters["capacity_factor"]
        transport_time_fraction = dict_common_parameters["transport_time_fraction"]
        factor_nb_max_draw_pairs = dict_common_parameters["factor_nb_max_draw_pairs"]
        max_nb_neighbours = dict_common_parameters["max_nb_neighbours"]
        distance_type = dict_common_parameters["euclidean"]
        sorting_criteria = dict_common_parameters["nb_neighbours_n_distance"]
        pairs_generation = dict_common_parameters["capacity"]
        pairs_criteria = dict_common_parameters["pairs_criteria"]
        small_world_like = dict_common_parameters["small_world_like"]
    

        for nb_nodes, nb_pairs, desired_mean_nb_neighbours  in ls_param_vals:
                print("--------------------------------------------------------------")
                print(nb_nodes, nb_pairs)
                print("--------------------------------------------------------------")
                #dir_name_graph_instance = "instance_generation/instances/capacity/"
                calculate_max_nb_edges = lambda x : 3*x-6
                desired_nb_edges = math.ceil(desired_mean_nb_neighbours * nb_nodes/2)

                for num_instance in range(nb_instances):
                        print("Generate instance ", num_instance)
                        # Return infos of the generated intance
                        infos_instance = generate_random_instance(nb_nodes = nb_nodes, 
                                                                        grid_size = grid_size, 
                                                                        r = int(grid_size * radius_proportion), 
                                                                        nb_arcs = min(calculate_max_nb_edges(nb_nodes), desired_nb_edges), 
                                                                        max_nb_draws_gen_graph = factor_nb_draws_gen_graph * calculate_max_nb_edges(nb_nodes), 
                                                                        max_nb_tries_gen_graph = max_nb_tries_gen_graph,
                                                                        base_capacity = base_capacity, 
                                                                        capacity_factor = capacity_factor, 
                                                                        transport_time_fraction = transport_time_fraction, 
                                                                        nb_pairs = nb_pairs, 
                                                                        nb_max_draws_pairs = nb_pairs * factor_nb_max_draw_pairs,
                                                                        max_nb_neighbours = max_nb_neighbours,
                                                                        distance_type = distance_type,
                                                                        sorting_criteria = sorting_criteria,
                                                                        pairs_generation = pairs_generation,
                                                                        pairs_criteria = pairs_criteria,
                                                                        print_ = False,
                                                                        small_world_like = small_world_like)
                        

                        adjacency = infos_instance["adj_mat"] 
                        arcs = infos_instance["arcs"]
                        nodes = infos_instance["nodes"]
                        capacities = infos_instance["capacities"]
                        transport_times = infos_instance["transport_times"]
                        pairs = infos_instance["pairs"]
                        weight_pairs = infos_instance["weight_pairs"]
                        
                        if desired_flow_values is None: # If the desired flow values is not None we take the default value of the 'desired_flow_values' 
                                desired_flow_values = [float("inf") for _ in range(len(pairs))]
                        print("Weight pairs ", weight_pairs)
                        return_dict = generate_multi_flow_instance(adjacency, 
                                                                capacities, 
                                                                transport_times, 
                                                                pairs, 
                                                                desired_flow_values, 
                                                                min_fl = min_fl, 
                                                                nb_it_print = None,
                                                                pairs_selection = pairs_selection,
                                                                return_transition_function = True)
                        """
                        Calculate the aggregated flow data. 
                        Keep in the graph, only the arcs which have aggregated flow on it.
                        Delete in the multiflow, the flow values list and the pairs, the elements
                        for which the flow value is zero. 
                        """
                        ls_multi_flow = return_dict["multi_flow"]
                        flow_values = [(pairs[i][0], pairs[i][1], return_dict["flow_values"][i]) for i in range(len(pairs))]
                        transition_functions = return_dict["transition_functions"]            



if __name__ == "__main__":
    test_names = {"construct_instances_from_files", 
                  ""}
    
    test_name = "construct_instances_from_files"
    
    if test_name == "construct_instances_from_files":
        construct_instances_from_files()
    
    elif test_name == "construct_complete_instances":
        # Initialize the dictionary containing the common parameters
        dict_common_parameters = dict()
        dict_common_parameters["nb_instances"] = 100
        dict_common_parameters["grid_size"] = 100
        dict_common_parameters["radius_proportion"] = 1/2
        dict_common_parameters["factor_nb_draws_gen_graph"] = 10
        dict_common_parameters["max_nb_tries_gen_graph"] = 10
        dict_common_parameters["base_capacity"] = 50
        dict_common_parameters["capacity_factor"] = 20
        dict_common_parameters["transport_time_fraction"] = 1/4
        dict_common_parameters["factor_nb_max_draw_pairs"] = 10
        dict_common_parameters["max_nb_neighbours"] = 6
        dict_common_parameters["distance_type"] = "euclidean"
        dict_common_parameters["sorting_criteria"] = "nb_neighbours_n_distance"
        dict_common_parameters["pairs_generation"] = "capacity"
        dict_common_parameters["pairs_criteria"] = 0
        dict_common_parameters["small_world_like"] = True

        # ls_param_vals = [(nb_nodes, nb_pairs, desired_mean_nb_neighbours), ...]
        ls_param_vals = list(zip([100, 150, 200], [10, 15, 20], [3, 4]))

        #
        construct_complete_instances(dict_common_parameters, ls_param_vals)