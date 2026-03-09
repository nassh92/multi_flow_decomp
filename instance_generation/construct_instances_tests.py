import os

import sys

import numpy as np

import math

from copy import deepcopy

import itertools

sys.path.append(os.getcwd())
from msmd.multi_flow_desag_instance_utils import construct_instances
from instance_generation.random_instance_generator import generate_random_instance
from instance_generation.generic_multi_flow_instances_generator import generate_multi_flow_instance
from utils.graph_utils import init_graph_arc_attribute_vals, create_isolated_nodes_graph, get_arcs, add_arc, has_arc
from msmd.multi_flow_desag_instance_utils import MultiFlowDesagInstance



#####################################   Construct instances from files  ##################################### 
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



#####################################   Construct instances from files  #####################################


def post_multi_flow_generation_processing (graph, 
                                     graph_representation, 
                                     transport_times, 
                                     all_pairs, 
                                     all_flow_values, 
                                     all_multi_flow):
        pairs, flow_values, multi_flow = [], [], []
        for i in range(len(all_pairs)):
                pairs.append(all_pairs[i])
                flow_values.append(all_flow_values[i])
                multi_flow.append(all_multi_flow[i])

        dict_params = {"multi_flow":multi_flow}
        init_val = lambda u, v, dict_params : sum(dict_params["multi_flow"][i][u][v] for i in range(len(dict_params["multi_flow"])))
        aggregated_flow = init_graph_arc_attribute_vals(graph, 
                                                        init_val = init_val,
                                                        dict_params = dict_params)
        
        new_graph = create_isolated_nodes_graph(len(graph), 
                                                graph_representation = graph_representation)
        for u, v in get_arcs(graph):
                if aggregated_flow[u][v] > 0: add_arc(new_graph, u, v)

        dict_params = {"transport_times":transport_times, 
                        "new_graph":new_graph,
                        "call_has_arc":has_arc}
        init_val = lambda u, v, dict_params : dict_params["transport_times"][u][v] if dict_params["call_has_arc"](dict_params["new_graph"], u, v) else float("inf")
        new_transport_times = init_graph_arc_attribute_vals(graph,
                                                            init_val = init_val,
                                                            dict_params = dict_params)
        dict_return = {"new_graph":new_graph, 
                       "new_transport_times":new_transport_times, 
                       "pairs":pairs, 
                       "flow_values":flow_values, 
                       "multi_flow":multi_flow, 
                       "aggregated_flow":aggregated_flow}
        return dict_return 



def construct_complete_instances(dict_common_parameters, 
                                 ls_param_vals,
                                 dir_path,
                                 desired_flow_values = None,
                                 min_fl = 1,
                                 pairs_selection = "capacity",
                                 graph_representation = "adjacency_matrix"):
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
        distance_type = dict_common_parameters["distance_type"]
        sorting_criteria = dict_common_parameters["sorting_criteria"]
        pairs_generation = dict_common_parameters["pairs_generation"]
        pairs_criteria = dict_common_parameters["pairs_criteria"]
        small_world_like = dict_common_parameters["small_world_like"]
    

        for nb_nodes, nb_pairs, desired_mean_nb_neighbours  in ls_param_vals:
                print("--------------------------------------------------------------")
                print("nb_nodes ", nb_nodes, " nb_pairs " ,nb_pairs, " des_nb_neigh " ,desired_mean_nb_neighbours)
                print("--------------------------------------------------------------")
                #dir_name_graph_instance = "instance_generation/instances/capacity/"
                calculate_max_nb_edges = lambda x : 3*x-6
                desired_nb_edges = math.ceil(desired_mean_nb_neighbours * nb_nodes/2)

                dict_instances = dict()
                for num_instance in range(nb_instances):
                        #print("Generate instance ", num_instance)
                        # Return infos of the generated intance
                        infos_instance = generate_random_instance(nb_nodes = nb_nodes, 
                                                                grid_size = grid_size, 
                                                                r = int(grid_size * radius_proportion), 
                                                                nb_edges = min(calculate_max_nb_edges(nb_nodes), desired_nb_edges), 
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
                                                                graph_representation = graph_representation,
                                                                print_ = False,
                                                                small_world_like = small_world_like)
                        

                        graph = infos_instance["adj_mat"] 
                        arcs_segs = infos_instance["arcs"]
                        nodes_to_segs = infos_instance["nodes"]
                        capacities = infos_instance["capacities"]
                        transport_times = infos_instance["transport_times"]
                        all_pairs = infos_instance["pairs"]
                        weight_pairs = infos_instance["weight_pairs"]
                        
                        if desired_flow_values is None: # If the desired flow values is not None we take the default value of the 'desired_flow_values' 
                                actual_desired_flow_values = [float("inf") for _ in range(len(all_pairs))]
                        else:
                                actual_desired_flow_values = deepcopy(desired_flow_values)
                        #print("Weight pairs ", weight_pairs)

                        return_dict = generate_multi_flow_instance(deepcopy(graph), 
                                                                deepcopy(capacities), 
                                                                deepcopy(transport_times), 
                                                                all_pairs, 
                                                                actual_desired_flow_values, 
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
                        
                        all_multi_flow = return_dict["multi_flow"]
                        all_flow_values = return_dict["flow_values"]
                        ls_transition_functions = return_dict["transition_functions"]

                        dict_post_processing = post_multi_flow_generation_processing (graph, 
                                                               graph_representation, 
                                                               transport_times, 
                                                               all_pairs, 
                                                               all_flow_values, 
                                                               all_multi_flow)
                        new_graph = dict_post_processing["new_graph"]
                        new_transport_times = dict_post_processing["new_transport_times"]
                        pairs = dict_post_processing["pairs"]
                        flow_values = dict_post_processing["flow_values"]
                        multi_flow = dict_post_processing["multi_flow"]
                        aggregated_flow = dict_post_processing["aggregated_flow"]
                        
                        mfd_instance = MultiFlowDesagInstance(new_graph, 
                                                     aggregated_flow,
                                                     new_transport_times, 
                                                     pairs, 
                                                     flow_values, 
                                                     ls_transition_function = ls_transition_functions,
                                                     update_transport_time = True,
                                                     update_transition_functions = True)
                        dict_instances[(num_instance, True, True)] = (mfd_instance, multi_flow)
                np.save(dir_path+"nb_nodes="+str(nb_nodes)+"_nb_pairs="+str(nb_pairs)+"_nb_neighbours="+str(desired_mean_nb_neighbours), 
                        dict_instances)
        
        print("End of the instances generation.")




if __name__ == "__main__":
    test_names = {"construct_instances_from_files", 
                  "construct_complete_instances"}
    
    test_name = "construct_complete_instances"
    
    if test_name == "construct_instances_from_files":
        construct_instances_from_files()
    
    elif test_name == "construct_complete_instances":
        # Initialize the dictionary containing the common parameters
        dict_common_parameters = dict()
        dict_common_parameters["nb_instances"] = 100
        dict_common_parameters["grid_size"] = 100
        dict_common_parameters["radius_proportion"] = 1/2
        dict_common_parameters["factor_nb_draws_gen_graph"] = 10
        dict_common_parameters["max_nb_tries_gen_graph"] = 100
        dict_common_parameters["base_capacity"] = 50
        dict_common_parameters["capacity_factor"] = 14
        dict_common_parameters["transport_time_fraction"] = 1/4
        dict_common_parameters["factor_nb_max_draw_pairs"] = 10
        dict_common_parameters["max_nb_neighbours"] = 6
        dict_common_parameters["distance_type"] = "euclidean"
        dict_common_parameters["sorting_criteria"] = "nb_neighbours_n_distance"
        dict_common_parameters["pairs_generation"] = "capacity"
        dict_common_parameters["pairs_criteria"] = 0
        dict_common_parameters["small_world_like"] = True

        # ls_param_vals = [(nb_nodes, nb_pairs, desired_mean_nb_neighbours), ...]
        ls_param_vals = list(itertools.product([100, 150, 200], [10, 15, 20], [3, 4]))
        #ls_param_vals = [(200, 15, 4), (200, 20, 3), (200, 20, 4)]

        # The path where the instances are stored
        dir_path = "data/simulated_data/complete_instances/node_pairs/small_world_like/capacity_factor=14/"

        # Generate complete instances
        construct_complete_instances(dict_common_parameters, 
                                     ls_param_vals,
                                     dir_path = dir_path,
                                     desired_flow_values = None,
                                     min_fl = 1,
                                     pairs_selection = "capacity",
                                     graph_representation = "adjacency_list")